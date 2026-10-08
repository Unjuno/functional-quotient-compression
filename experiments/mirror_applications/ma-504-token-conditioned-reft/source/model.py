"""Small conditional hidden-state intervention models for MA-504."""
from __future__ import annotations
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

WIDTH, RANK, OUT = 16, 4, 8
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def givens(angles: torch.Tensor) -> torch.Tensor:
    if angles.shape[-1] != 6:
        raise ValueError("rank-4 Givens coordinates require six angles")
    eye = torch.eye(RANK, dtype=angles.dtype, device=angles.device)
    rot = eye
    for k, (i, j) in enumerate(PAIRS):
        c, s = torch.cos(angles[..., k]), torch.sin(angles[..., k])
        g = eye.expand(*angles.shape[:-1], RANK, RANK).clone()
        g[..., i, i], g[..., j, j] = c, c
        g[..., i, j], g[..., j, i] = -s, s
        rot = g @ rot
    return rot


def teacher_params(seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed + 700)
    a = torch.randn(RANK, WIDTH, generator=g) / math.sqrt(WIDTH)
    b = torch.randn(OUT, RANK, generator=g) / math.sqrt(RANK)
    # Two deterministic distinct rotations, fixed before any model training.
    angles = torch.tensor([[.25, -.15, .2, .1, -.2, .18],
                           [-.28, .17, -.13, .22, .11, -.19]])
    return a, b, angles


def make_batch(seed: int, split: str, count: int):
    offsets = {"train": 0, "dev": 1_000_000, "fresh": 2_000_000, "audit": 3_000_000}
    g = torch.Generator(device="cpu").manual_seed(seed + offsets[split])
    x = torch.randn(count, 8, WIDTH, generator=g)
    context = torch.randint(0, 2, (count, 8), generator=g).float()
    a, b, angles = teacher_params(seed)
    z = F.linear(x, a)
    r = givens(angles[context.long()])
    z = torch.einsum("btij,btj->bti", r, z)
    y = F.linear(z, b)
    return x, context, y


class Intervention(nn.Module):
    """mode in static, independent, mirror, film, or full."""
    def __init__(self, mode: str, seed: int, generator_hidden: int = 4):
        super().__init__()
        if mode not in {"static", "independent", "mirror", "film", "full"}:
            raise ValueError(mode)
        self.mode = mode
        if mode in {"static", "mirror", "film"}:
            self.a = nn.Parameter(torch.empty(RANK, WIDTH))
            self.b = nn.Parameter(torch.zeros(OUT, RANK))
        if mode == "independent":
            self.a_private = nn.Parameter(torch.empty(2, RANK, WIDTH))
            self.b_private = nn.Parameter(torch.zeros(2, OUT, RANK))
        if mode == "full":
            self.weight = nn.Parameter(torch.zeros(2, OUT, WIDTH))
        if mode in {"mirror", "film"}:
            out_dim = 6 if mode == "mirror" else RANK
            self.generator = nn.Sequential(nn.Linear(1, generator_hidden), nn.Tanh(),
                                           nn.Linear(generator_hidden, out_dim))
        self.reset_parameters(seed)

    def reset_parameters(self, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            if self.mode in {"static", "mirror", "film"}:
                self.a.copy_(torch.randn(self.a.shape, generator=g) / math.sqrt(WIDTH))
            if self.mode == "independent":
                self.a_private.copy_(torch.randn(self.a_private.shape, generator=g) / math.sqrt(WIDTH))

    def forward(self, x, context):
        if self.mode == "full":
            w = self.weight[context.long()]
            return torch.einsum("bti,btoi->bto", x, w)
        if self.mode == "independent":
            a = self.a_private[context.long()]
            b = self.b_private[context.long()]
            z = torch.einsum("btri,bti->btr", a, x)
            return torch.einsum("btor,btr->bto", b, z)
        z = F.linear(x, self.a)
        if self.mode == "static":
            return F.linear(z, self.b)
        code = self.generator(context.unsqueeze(-1))
        if self.mode == "mirror":
            r = givens(code)
            z = torch.einsum("btij,btj->bti", r, z)
        else:
            z = z * (1.0 + code)
        return F.linear(z, self.b)

    def active_macs_per_token(self):
        if self.mode == "full":
            return OUT * WIDTH
        if self.mode == "independent":
            return 2 * RANK * WIDTH
        base = 2 * RANK * WIDTH
        if self.mode == "mirror":
            return base + 6 * 4 * RANK + 4 * 1 + 4 * 6
        if self.mode == "film":
            return base + 4 * RANK + 4 * 1 + 4 * RANK
        return base


def save_payload(path: Path, model: Intervention, seed: int):
    payload = {"schema": "MA-504/inference-v1", "seed": seed, "mode": model.mode,
               "sequence_length": 8, "input_width": WIDTH, "rank": RANK,
               "output_width": OUT, "context_values": [0, 1],
               "state_dict": {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}}
    torch.save(payload, path)
    raw = path.read_bytes()
    import hashlib
    return {"path": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
