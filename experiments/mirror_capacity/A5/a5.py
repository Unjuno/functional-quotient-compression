"""A5: fixed-budget task-indexed function-family distillation, CPU FP32.

All numerical quantities are dimensionless. Teacher mechanisms do NOT use Mirror.
See PROTOCOL.md for frozen comparisons, sampling, evidence boundaries.
"""
from __future__ import annotations

import hashlib
import math
import struct
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

D, O, TASKS, RANK = 6, 3, 16, 4
KINDS = ('mirror4', 'pre4', 'film4', 'basis4', 'lowrank4', 'dense4')
MAGIC = b'FQCA5\0\0\0'
HEADER = struct.Struct('<8s6I')  # 32 B total


def params_at_width(kind: str, width: int) -> int:
    if kind not in KINDS or width < 1:
        raise ValueError('invalid kind/width')
    base = (D + 1 + O) * width + O + TASKS * RANK
    return base + {
        'mirror4': 0, 'pre4': 0, 'film4': 2 * RANK * width,
        'basis4': RANK * O * width,
        'lowrank4': RANK * (D + width),
        'dense4': RANK * width,
    }[kind]


class Student(nn.Module):
    def __init__(self, kind: str, width: int):
        super().__init__()
        if kind not in KINDS or width < 2 or (kind in ('mirror4', 'pre4') and width % 2):
            raise ValueError('invalid architecture')
        self.kind, self.width = kind, width
        self.up = nn.Linear(D + (RANK if kind == 'dense4' else 0), width)
        self.down = nn.Linear(width, O)
        self.code = nn.Embedding(TASKS, RANK)
        nn.init.uniform_(self.code.weight, -0.5, 0.5)
        if kind == 'film4':
            self.gain = nn.Parameter(torch.randn(RANK, width) * 0.2)
            self.shift = nn.Parameter(torch.randn(RANK, width) * 0.2)
        if kind == 'basis4':
            self.basis = nn.Parameter(torch.randn(width, RANK * O) * (0.2 / math.sqrt(width)))
        if kind == 'lowrank4':
            self.lora_in = nn.Parameter(torch.randn(RANK, D) * 0.1)
            self.lora_out = nn.Parameter(torch.randn(RANK, width) * 0.1)

    def forward(self, x: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != D or task.shape != (len(x),) or task.dtype != torch.long:
            raise ValueError('input/ID shape/dtype')
        code = self.code(task)
        v = self.up(torch.cat((x, code), -1) if self.kind == 'dense4' else x)
        if self.kind in ('pre4', 'mirror4'):
            paired = v.reshape(-1, self.width // 2, 2)
            a = code[:, torch.arange(self.width // 2, device=x.device) % RANK]
            h0 = F.gelu(paired[..., 0] + a * paired[..., 1])
            h1 = F.gelu(paired[..., 1])
            if self.kind == 'mirror4':
                h0 = h0 - a * h1
            h = torch.stack((h0, h1), -1).reshape(-1, self.width)
        else:
            if self.kind == 'film4':
                v = v * (1 + code @ self.gain) + code @ self.shift
            if self.kind == 'lowrank4':
                v = v + ((x @ self.lora_in.T) * code) @ self.lora_out
            h = F.gelu(v)
        y = self.down(h)
        if self.kind == 'basis4':
            y = y + ((h @ self.basis).reshape(-1, RANK, O) * code[:, :, None]).sum(1)
        return y


def create(kind: str, cap: int) -> Student:
    if kind not in KINDS or cap < 128:
        raise ValueError('invalid specification')
    step = 2 if kind in ('mirror4', 'pre4') else 1
    legal = [h for h in range(2, (cap - 64) // 4 + 1) if h % step == 0 and (64 + 4 * params_at_width(kind, h) <= cap)]
    if not legal:
        raise ValueError('too little storage')
    model = Student(kind, max(legal))
    assert sum(p.numel() for p in model.parameters()) == params_at_width(kind, model.width)
    return model


def serialize(model: Student) -> bytes:
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 or not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('only finite CPU FP32 parameters')
    header = HEADER.pack(MAGIC, 1, KINDS.index(model.kind), TASKS, model.width, D, O)
    payload = b''.join(p.detach().contiguous().numpy().astype('<f4', copy=False).tobytes() for p in model.parameters())
    raw = header + payload
    assert len(raw) == 32 + 4 * params_at_width(model.kind, model.width)
    return raw + hashlib.sha256(raw).digest()


def deserialize(raw: bytes) -> Student:
    if len(raw) < 64 or hashlib.sha256(raw[:-32]).digest() != raw[-32:]:
        raise ValueError('bad length/digest')
    magic, version, type_id, tasks, width, d, o = HEADER.unpack(raw[:32])
    if magic != MAGIC or version != 1 or type_id >= len(KINDS) or tasks != TASKS or d != D or o != O:
        raise ValueError('bad model header')
    kind = KINDS[type_id]
    model = Student(kind, width)
    if len(raw) != 64 + 4 * params_at_width(kind, width):
        raise ValueError('bad payload size')
    values = np.frombuffer(raw[32:-32], dtype='<f4')
    if not np.isfinite(values).all():
        raise ValueError('non-finite payload')
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(torch.from_numpy(values[offset:offset + n].copy()).reshape(p.shape))
            offset += n
    assert offset == len(values)
    return model


def shear_explicit(model: Student, x: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
    """Independent arithmetic via generated expert matrices, for tests only."""
    if model.kind not in ('mirror4', 'pre4'):
        raise ValueError('only shear models')
    outs = []
    for i in range(len(x)):
        a = model.code(task[i:i+1])[0]
        pair = a[torch.arange(model.width // 2) % RANK]
        up = model.up.weight.reshape(-1, 2, D)
        bias = model.up.bias.reshape(-1, 2)
        generated_up = torch.stack((up[:, 0] + pair[:, None] * up[:, 1], up[:, 1]), 1).reshape(-1, D)
        generated_bias = torch.stack((bias[:, 0] + pair * bias[:, 1], bias[:, 1]), 1).reshape(-1)
        h = F.gelu(generated_up @ x[i] + generated_bias)
        down = model.down.weight
        if model.kind == 'mirror4':
            d_pair = down.reshape(O, -1, 2)
            down = torch.stack((d_pair[..., 0], d_pair[..., 1] - pair[None, :] * d_pair[..., 0]), -1).reshape_as(down)
        outs.append(down @ h + model.down.bias)
    return torch.stack(outs)


def make_world(world: int, family: str, sizes=(384, 128, 256, 256)) -> dict:
    if family not in ('compositional', 'independent') or world < 0 or len(sizes) != 4:
        raise ValueError('unsupported world/family')
    family_id = 0 if family == 'compositional' else 1
    gen = torch.Generator().manual_seed(202611150 + 1009 * world + 881 * family_id)
    count = RANK if family == 'compositional' else TASKS
    hidden = 24
    w1 = torch.randn(count, D, hidden, generator=gen) / math.sqrt(D)
    b1 = torch.randn(count, hidden, generator=gen) * 0.16
    w2 = torch.randn(count, hidden, O, generator=gen) / math.sqrt(hidden)
    b2 = torch.randn(count, O, generator=gen) * 0.06
    coeff = torch.randn(TASKS, RANK, generator=gen) * 0.6 if family == 'compositional' else None
    out = {}
    with torch.no_grad():
        for split_index, (split, n) in enumerate(zip(('train', 'val', 'iid', 'ood'), sizes)):
            rng = torch.Generator().manual_seed(301251000 + world * 3491 + family_id * 761 + split_index * 197)
            x = torch.randn(n, D, generator=rng) * (1.6 if split == 'ood' else 1.0)
            hidden_features = F.gelu(torch.einsum('nd,rdh->rnh', x, w1) + b1[:, None, :])
            base = torch.einsum('rnh,rho->rno', hidden_features, w2) + b2[:, None, :]
            y = torch.einsum('kr,rno->kno', coeff, base) if coeff is not None else base
            out[split] = (x.contiguous(), y.contiguous())
    return out


def eval_model(model: Student, data: tuple[torch.Tensor, torch.Tensor]) -> dict:
    x, y = data
    if x.ndim != 2 or y.ndim != 3 or y.shape != (TASKS, len(x), O):
        raise ValueError('invalid dataset')
    with torch.no_grad():
        xx = x.repeat(TASKS, 1)
        task = torch.arange(TASKS, dtype=torch.long).repeat_interleave(len(x))
        pred = model(xx, task).reshape_as(y)
        delta = (pred.double() - y.double()).square()
        denom = (y.double() - y.double().mean(0, keepdim=True)).square().mean()
        if denom <= 1e-12:
            raise ValueError('degenerate target family')
        mse = delta.mean()
        return {'nrmse': math.sqrt(float(mse / denom)), 'mse': float(mse),
                'worst_task_nrmse': math.sqrt(float(delta.mean((1, 2)).max() / denom)),
                'between': float(denom)}


def train_model(model: Student, data: dict, steps: int, seed: int, batch: int = 256) -> float:
    if steps < 1 or batch < 1:
        raise ValueError('steps/batch')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=0.0)
    x, y = data['train']
    model.train()
    for _ in range(steps):
        ii = torch.randint(len(x), (batch,), generator=rng)
        tasks = torch.randint(TASKS, (batch,), generator=rng)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x[ii], tasks), y[tasks, ii])
        if not torch.isfinite(loss):
            raise ValueError('nonfinite loss')
        loss.backward()
        gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(gradnorm):
            raise ValueError('nonfinite gradient')
        opt.step()
    model.eval()
    return float(loss.item())
