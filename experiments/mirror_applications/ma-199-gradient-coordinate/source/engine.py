import io
import math
import struct
import time

import torch
from torch import nn

D, O, TASKS, RANK = 16, 8, 5, 2
METHODS = ('hard_tie', 'mirror_gradient', 'shared_plane_rank2', 'lora_rank1', 'lora_rank2', 'hypernet_rank1', 'independent_full')


def world(seed, condition):
    g = torch.Generator().manual_seed(seed)
    base_w = torch.randn(D, O, generator=g) / math.sqrt(D)
    base_b = torch.randn(O, generator=g) * .03
    raw = torch.randn(D, 2, generator=g)
    basis_start = time.perf_counter()
    plane, _ = torch.linalg.qr(raw, mode='reduced')
    basis_setup_time = time.perf_counter() - basis_start
    angles = torch.rand(TASKS, generator=g) * 1.4 - .7
    vectors = torch.randn(TASKS, O, generator=g) * .25
    if condition == 'aligned':
        deltas = torch.stack([(plane @ torch.stack((torch.cos(angles[t]), torch.sin(angles[t]))))[:, None] @ vectors[t][None, :] for t in range(TASKS)])
    elif condition == 'independent':
        deltas = torch.stack([torch.zeros(D, O)] + [torch.randn(D, O, generator=g) * .18 for _ in range(TASKS - 1)])
        angles.zero_(); vectors.zero_()
    else:
        raise ValueError(condition)
    deltas[0].zero_()
    return {'seed': seed, 'condition': condition, 'base_w': base_w, 'base_b': base_b, 'plane': plane, 'basis_setup_time': basis_setup_time,
            'angles': angles, 'vectors': vectors, 'deltas': deltas}


def teacher(x, w, task):
    return x @ (w['base_w'] + w['deltas'][task]) + w['base_b']


class Hyper(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding = nn.Embedding(TASKS, 8); self.proj = nn.Linear(8, D + O)
        nn.init.normal_(self.embedding.weight, std=.1); nn.init.normal_(self.proj.weight, std=.02); nn.init.zeros_(self.proj.bias)

    def factors(self, task):
        v = self.proj(self.embedding(torch.tensor(task)))
        return v[:D].reshape(D, 1), v[D:].reshape(1, O)


class Learner:
    def __init__(self, method, base_w, base_b, plane, seed, lr):
        torch.manual_seed(seed); self.method, self.seed = method, seed
        self.base_w, self.base_b, self.plane = base_w.detach().clone(), base_b.detach().clone(), plane.detach().clone()
        self.codes, self.adapters, self.private = {}, {}, {}
        self.hyper = Hyper() if method == 'hypernet_rank1' else None
        self.shared_optimizer = torch.optim.AdamW(self.hyper.parameters(), lr=lr, weight_decay=1e-4) if self.hyper else None
        self.optimizers = {}; self.seen = [0]; self.wall = {}; self.examples = {}; self.updates = {}

    def forward(self, x, task):
        if task == 0 or self.method == 'hard_tie': return x @ self.base_w + self.base_b
        if self.method == 'mirror_gradient' and task in self.codes:
            angle, v = self.codes[task]
            coords = torch.stack((torch.cos(angle), torch.sin(angle)))
            u = self.plane @ coords
            return x @ self.base_w + (x @ u[:, None]) @ v[None, :] + self.base_b
        if self.method == 'shared_plane_rank2' and task in self.codes:
            c = self.codes[task]
            return x @ self.base_w + (x @ self.plane) @ c + self.base_b
        if self.method in ('lora_rank1', 'lora_rank2') and task in self.adapters:
            a, b = self.adapters[task]
            return x @ self.base_w + (x @ a) @ b + self.base_b
        if self.method == 'hypernet_rank1' and task > 0:
            a, b = self.hyper.factors(task)
            return x @ self.base_w + (x @ a) @ b + self.base_b
        if self.method == 'independent_full' and task in self.private:
            w, b = self.private[task]
            return x @ w + b
        return x @ self.base_w + self.base_b

    def acquire(self, task, x, y, lr, updates=300):
        if self.method == 'hard_tie': self.seen.append(task); return
        if self.method == 'mirror_gradient':
            angle = nn.Parameter(torch.zeros(())); v = nn.Parameter(torch.randn(O) * .02); self.codes[task] = (angle, v); params = [angle, v]
        elif self.method == 'shared_plane_rank2':
            c = nn.Parameter(torch.zeros(2, O)); self.codes[task] = c; params = [c]
        elif self.method in ('lora_rank1', 'lora_rank2'):
            rank = 1 if self.method == 'lora_rank1' else RANK
            a = nn.Parameter(torch.randn(D, rank) / math.sqrt(D)); b = nn.Parameter(torch.zeros(rank, O)); self.adapters[task] = (a, b); params = [a, b]
        elif self.method == 'hypernet_rank1':
            params = list(self.hyper.parameters())
        elif self.method == 'independent_full':
            w = nn.Parameter(self.base_w.clone()); b = nn.Parameter(self.base_b.clone()); self.private[task] = (w, b); params = [w, b]
        else: raise ValueError(self.method)
        opt = self.shared_optimizer if self.method == 'hypernet_rank1' else torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        if self.method != 'hypernet_rank1': self.optimizers[task] = opt
        start = time.perf_counter()
        for _ in range(updates):
            loss = (self.forward(x, task) - y).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        self.wall[task] = time.perf_counter() - start
        self.examples[task] = updates * len(x); self.updates[task] = updates; self.seen.append(task)

    def records(self):
        rec = [('base_w', self.base_w), ('base_b', self.base_b), ('task_ids', torch.tensor(self.seen, dtype=torch.uint8))]
        if self.method in ('mirror_gradient', 'shared_plane_rank2'):
            rec.append(('gradient_plane', self.plane))
        if self.method == 'mirror_gradient':
            for t, (a, b) in sorted(self.codes.items()): rec.extend([(f'code.{t}.angle', a.detach()), (f'code.{t}.vector', b.detach())])
        elif self.method == 'shared_plane_rank2':
            rec.extend((f'code.{t}.matrix', c.detach()) for t, c in sorted(self.codes.items()))
        for t, (a, b) in sorted(self.adapters.items()): rec.extend([(f'adapter.{t}.A', a.detach()), (f'adapter.{t}.B', b.detach())])
        for t, (a, b) in sorted(self.private.items()): rec.extend([(f'private.{t}.W', a.detach()), (f'private.{t}.b', b.detach())])
        if self.hyper is not None: rec.extend((f'hyper.{k}', v.detach()) for k, v in sorted(self.hyper.state_dict().items()))
        return rec

    def base_records(self):
        rec = [('base_w', self.base_w), ('base_b', self.base_b), ('task_ids', torch.tensor([0], dtype=torch.uint8))]
        if self.method in ('mirror_gradient', 'shared_plane_rank2'): rec.append(('gradient_plane', self.plane))
        return rec

    def serialize(self): return serialize_records(self.method, self.records())
    def inference_bytes(self): return len(self.serialize())
    def base_bytes(self): return len(serialize_records(self.method, self.base_records()))
    def resume_bytes(self):
        states = {t: o.state_dict() for t, o in self.optimizers.items()}
        if self.shared_optimizer: states['shared'] = self.shared_optimizer.state_dict()
        f = io.BytesIO(); torch.save({'state': self.records(), 'optimizers': states}, f); return len(f.getvalue())
    def resume_base_bytes(self):
        f = io.BytesIO(); torch.save({'state': self.base_records(), 'optimizers': {}}, f); return len(f.getvalue())

    @classmethod
    def from_payload(cls, payload):
        method, records = deserialize_records(payload); state = dict(records)
        plane = state.get('gradient_plane', torch.zeros(D, 2))
        m = cls(method, state['base_w'], state['base_b'], plane, 0, .003)
        for name, value in state.items():
            parts = name.split('.')
            if parts[0] == 'code' and len(parts) == 3 and parts[2] in ('angle', 'vector'):
                task = int(parts[1]); m.codes.setdefault(task, [None, None])[0 if parts[2] == 'angle' else 1] = nn.Parameter(value.clone())
            elif parts[0] == 'code' and len(parts) == 3 and parts[2] == 'matrix':
                m.codes[int(parts[1])] = nn.Parameter(value.clone())
            elif parts[0] == 'adapter':
                task = int(parts[1]); m.adapters.setdefault(task, [None, None])[0 if parts[2] == 'A' else 1] = nn.Parameter(value.clone())
            elif parts[0] == 'private':
                task = int(parts[1]); m.private.setdefault(task, [None, None])[0 if parts[2] == 'W' else 1] = nn.Parameter(value.clone())
        m.codes = {k: (tuple(v) if isinstance(v, list) else v) for k, v in m.codes.items()}
        m.adapters = {k: tuple(v) for k, v in m.adapters.items()}
        m.private = {k: tuple(v) for k, v in m.private.items()}
        if m.hyper is not None: m.hyper.load_state_dict({k[len('hyper.'):]: v for k, v in state.items() if k.startswith('hyper.')})
        m.seen = state['task_ids'].tolist()
        return m


def serialize_records(method, records):
    method = method.encode('ascii'); out = bytearray(b'MA199I1\0')
    out.extend(struct.pack('<BHHHH', len(method), len(records), D, O, RANK)); out.extend(method)
    for name, t in records:
        name = name.encode('ascii'); t = t.detach().contiguous().cpu()
        type_id = 1 if t.dtype == torch.uint8 else 0
        if not type_id: t = t.float()
        raw = t.numpy().tobytes()
        out.extend(struct.pack('<H', len(name))); out.extend(name); out.extend(struct.pack('<BB', type_id, t.ndim))
        if t.ndim: out.extend(struct.pack('<' + 'H' * t.ndim, *t.shape))
        out.extend(struct.pack('<I', len(raw))); out.extend(raw)
    return bytes(out)


def deserialize_records(payload):
    view = memoryview(payload)
    if bytes(view[:8]) != b'MA199I1\0': raise ValueError('bad inference magic')
    ml, n, d, o, rank = struct.unpack_from('<BHHHH', view, 8)
    if (d, o, rank) != (D, O, RANK): raise ValueError('dimension mismatch')
    off = 17; method = bytes(view[off:off+ml]).decode('ascii'); off += ml; records = []
    for _ in range(n):
        nl = struct.unpack_from('<H', view, off)[0]; off += 2
        name = bytes(view[off:off+nl]).decode('ascii'); off += nl
        type_id, ndim = struct.unpack_from('<BB', view, off); off += 2
        shape = struct.unpack_from('<'+'H'*ndim, view, off) if ndim else (); off += 2*ndim
        size = struct.unpack_from('<I', view, off)[0]; off += 4
        raw = bytes(view[off:off+size]); off += size
        dtype = torch.uint8 if type_id == 1 else torch.float32
        records.append((name, torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(shape)))
    if off != len(view): raise ValueError('trailing payload bytes')
    return method, records


def evaluate(m, w, seed, upto):
    g = torch.Generator().manual_seed(seed); x = torch.randn(2048, D, generator=g); out = []
    with torch.no_grad():
        for task in range(upto + 1): out.append(float((m.forward(x, task) - teacher(x, w, task)).square().mean()))
    return out


def mac_proxy(method, examples):
    per = D * O
    if method == 'mirror_gradient': per += D + O + 4
    elif method == 'shared_plane_rank2': per += 2 * (D + O)
    elif method.startswith('lora'): per += (1 if method.endswith('1') else RANK) * (D + O)
    elif method == 'hypernet_rank1': per += D + O + 8 * (D + O)
    return examples * per
