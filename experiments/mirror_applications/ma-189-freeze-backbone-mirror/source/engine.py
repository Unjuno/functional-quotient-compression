import io
import math
import struct
import time

import torch
from torch import nn

D, O, RANK = 16, 8, 2
METHODS = ('hard_tie', 'mirror_one_sided', 'mirror_two_sided', 'lora_rank1', 'lora_rank2', 'hypernet_rank2', 'independent_full')


def rotate_pair(x, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    a, b = x[..., 0], x[..., 1]
    return torch.cat((torch.stack((c * a - s * b, s * a + c * b), -1), x[..., 2:]), dim=-1)


def world(seed, condition):
    g = torch.Generator().manual_seed(seed)
    base_w = torch.randn(D, O, generator=g) / math.sqrt(D)
    base_b = torch.randn(O, generator=g) * .05
    if condition == 'aligned':
        in_angle, out_angle = (torch.rand((), generator=g) * 1.0 - .5 for _ in range(2))
        task_w, task_b = base_w, base_b
    elif condition == 'independent':
        in_angle, out_angle = torch.tensor(0.), torch.tensor(0.)
        task_w = torch.randn(D, O, generator=g) / math.sqrt(D)
        task_b = torch.randn(O, generator=g) * .05
    else:
        raise ValueError(condition)
    return {'seed': seed, 'condition': condition, 'base_w': base_w, 'base_b': base_b,
            'in_angle': in_angle, 'out_angle': out_angle, 'task_w': task_w, 'task_b': task_b}


def teacher(x, w, task=1):
    if task == 0:
        return x @ w['base_w'] + w['base_b']
    if w['condition'] == 'aligned':
        y = rotate_pair(x, w['in_angle']) @ w['base_w'] + w['base_b']
        return rotate_pair(y, w['out_angle'])
    return x @ w['task_w'] + w['task_b']


class Hyper(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(2, 8)
        self.proj = nn.Linear(8, RANK * (D + O))
        nn.init.normal_(self.embedding.weight, std=.1)
        nn.init.normal_(self.proj.weight, std=.015)
        nn.init.zeros_(self.proj.bias)

    def factors(self):
        v = self.proj(self.embedding(torch.tensor(1)))
        return v[:D * RANK].reshape(D, RANK), v[D * RANK:].reshape(RANK, O)


class Learner:
    def __init__(self, method, base_w, base_b, seed, lr):
        torch.manual_seed(seed)
        self.method, self.seed = method, seed
        self.base_w, self.base_b = base_w.detach().clone(), base_b.detach().clone()
        self.params, self.hyper = [], None
        self.angle_in = self.angle_out = None
        self.private_w = self.private_b = None
        self.a = self.b = None
        if method == 'mirror_one_sided':
            self.angle_in = nn.Parameter(torch.zeros(())); self.params = [self.angle_in]
        elif method == 'mirror_two_sided':
            self.angle_in = nn.Parameter(torch.zeros(())); self.angle_out = nn.Parameter(torch.zeros(())); self.params = [self.angle_in, self.angle_out]
        elif method in ('lora_rank1', 'lora_rank2'):
            rank = 1 if method == 'lora_rank1' else RANK
            self.a = nn.Parameter(torch.randn(D, rank) / math.sqrt(D)); self.b = nn.Parameter(torch.zeros(rank, O)); self.params = [self.a, self.b]
        elif method == 'hypernet_rank2':
            self.hyper = Hyper(); self.params = list(self.hyper.parameters())
        elif method == 'independent_full':
            self.private_w = nn.Parameter(self.base_w.clone()); self.private_b = nn.Parameter(self.base_b.clone()); self.params = [self.private_w, self.private_b]
        elif method != 'hard_tie':
            raise ValueError(method)
        self.optimizer = torch.optim.AdamW(self.params, lr=lr, weight_decay=0.) if self.params else None
        self.updates, self.examples, self.wall = 0, 0, 0.

    def forward(self, x, task=1):
        if task == 0 or self.method == 'hard_tie':
            return x @ self.base_w + self.base_b
        if self.method == 'mirror_one_sided':
            return rotate_pair(x, self.angle_in) @ self.base_w + self.base_b
        if self.method == 'mirror_two_sided':
            return rotate_pair(rotate_pair(x, self.angle_in) @ self.base_w + self.base_b, self.angle_out)
        if self.method in ('lora_rank1', 'lora_rank2'):
            return x @ self.base_w + (x @ self.a) @ self.b + self.base_b
        if self.method == 'hypernet_rank2':
            a, b = self.hyper.factors()
            return x @ self.base_w + (x @ a) @ b + self.base_b
        if self.method == 'independent_full':
            return x @ self.private_w + self.private_b
        raise AssertionError(self.method)

    def fit(self, x, y, updates=200):
        start = time.perf_counter()
        if self.optimizer:
            for _ in range(updates):
                loss = (self.forward(x) - y).square().mean()
                self.optimizer.zero_grad(set_to_none=True); loss.backward(); self.optimizer.step()
            self.updates += updates; self.examples += updates * len(x)
        self.wall += time.perf_counter() - start

    def records(self, base_only=False):
        rec = [('base_w', self.base_w), ('base_b', self.base_b), ('task_id', torch.tensor([1], dtype=torch.uint8))]
        if not base_only:
            if self.angle_in is not None: rec.append(('mirror.angle_in', self.angle_in.detach()))
            if self.angle_out is not None: rec.append(('mirror.angle_out', self.angle_out.detach()))
            if self.a is not None: rec.extend([('lora.A', self.a.detach()), ('lora.B', self.b.detach())])
            if self.hyper is not None: rec.extend((f'hyper.{k}', v.detach()) for k, v in sorted(self.hyper.state_dict().items()))
            if self.private_w is not None: rec.extend([('private.W', self.private_w.detach()), ('private.b', self.private_b.detach())])
        return rec

    def serialize(self, base_only=False):
        return serialize_records(self.method, self.records(base_only))

    def inference_bytes(self): return len(self.serialize())
    def base_bytes(self): return len(self.serialize(base_only=True))

    def resume_bytes(self):
        f = io.BytesIO(); torch.save({'state': self.records(), 'optimizer': self.optimizer.state_dict() if self.optimizer else {}}, f)
        return len(f.getvalue())

    def resume_base_bytes(self):
        f = io.BytesIO(); torch.save({'state': self.records(base_only=True), 'optimizer': {}}, f)
        return len(f.getvalue())

    @classmethod
    def from_payload(cls, payload):
        method, records = deserialize_records(payload)
        state = dict(records)
        learner = cls(method, state['base_w'], state['base_b'], 0, .003)
        with torch.no_grad():
            if 'mirror.angle_in' in state: learner.angle_in.copy_(state['mirror.angle_in'])
            if 'mirror.angle_out' in state: learner.angle_out.copy_(state['mirror.angle_out'])
            if 'lora.A' in state:
                learner.a.copy_(state['lora.A']); learner.b.copy_(state['lora.B'])
            if 'private.W' in state:
                learner.private_w.copy_(state['private.W']); learner.private_b.copy_(state['private.b'])
        if learner.hyper is not None:
            learner.hyper.load_state_dict({k[len('hyper.'):]: v for k, v in state.items() if k.startswith('hyper.')})
        return learner


def serialize_records(method, records):
    method = method.encode('ascii'); out = bytearray(b'MA189I1\0')
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
    if bytes(view[:8]) != b'MA189I1\0': raise ValueError('bad inference magic')
    method_len, n, d, o, rank = struct.unpack_from('<BHHHH', view, 8)
    if (d, o, rank) != (D, O, RANK): raise ValueError('dimension mismatch')
    off = 17; method = bytes(view[off:off+method_len]).decode('ascii'); off += method_len; records = []
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


def evaluate(m, w, seed):
    g = torch.Generator().manual_seed(seed); x = torch.randn(4096, D, generator=g)
    with torch.no_grad():
        task0_before = float((x @ w['base_w'] + w['base_b'] - teacher(x, w, 0)).square().mean())
        task0_after = float((m.forward(x, 0) - teacher(x, w, 0)).square().mean())
        task1 = float((m.forward(x, 1) - teacher(x, w, 1)).square().mean())
    return task0_before, task0_after, task1


def mac_proxy(method, examples):
    per_example = D * O
    if method == 'mirror_one_sided': per_example += 4
    elif method == 'mirror_two_sided': per_example += 8
    elif method.startswith('lora'): per_example += (1 if method.endswith('1') else RANK) * (D + O)
    elif method == 'hypernet_rank2': per_example += RANK * (D + O) + 8 * RANK * (D + O)
    return examples * per_example


def throughput(m, seed):
    g = torch.Generator().manual_seed(seed); x = torch.randn(64, D, generator=g)
    with torch.no_grad():
        for _ in range(5): m.forward(x)
        t = time.perf_counter()
        for _ in range(30): m.forward(x)
    return 64 * 30 / (time.perf_counter() - t)
