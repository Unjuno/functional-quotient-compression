"""Sub-300-scalar research models for MA-1176 scaling-axis screening.

No claim that these are arbitrary learned MLPs or full Transformer attention.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

D, RULES = 4, 3


def householder(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """H_u x = x - 2 (x.u)/(u.u) u, with zero-safe normalization."""
    denominator = (u * u).sum(dim=-1, keepdim=True).clamp_min(1e-10)
    return x - (2.0 * ((x * u).sum(dim=-1, keepdim=True) / denominator)) * u


class FeatureModel(nn.Module):
    """Shared / FiLM / Householder nonlinear Mirror / independent per-rule FFN."""
    def __init__(self, kind: str, width: int):
        super().__init__()
        if kind not in ('shared', 'film', 'mirror', 'independent'):
            raise ValueError(kind)
        self.kind, self.width = kind, width
        if kind == 'independent':
            self.up = nn.Parameter(torch.randn(RULES, width, D) * (0.55 / math.sqrt(D)))
            self.down = nn.Parameter(torch.randn(RULES, D, width) * (0.55 / math.sqrt(width)))
        else:
            self.up = nn.Parameter(torch.randn(width, D) * (0.55 / math.sqrt(D)))
            self.down = nn.Parameter(torch.randn(D, width) * (0.55 / math.sqrt(width)))
        if kind in ('film', 'mirror'):
            self.code = nn.Parameter(torch.randn(RULES, width) * 0.45)

    def forward(self, x: torch.Tensor, rule: torch.Tensor) -> torch.Tensor:
        if self.kind == 'independent':
            z = torch.einsum('bi,bhi->bh', x, self.up[rule])
            return torch.einsum('bh,boh->bo', torch.tanh(z), self.down[rule])
        z = F.linear(x, self.up)
        if self.kind == 'mirror':
            u = self.code[rule]
            z = householder(z, u)
            z = torch.tanh(z)
            z = householder(z, u)  # Reflection is self-inverse
        elif self.kind == 'film':
            z = torch.tanh(z) * (1.0 + 0.8 * torch.tanh(self.code[rule]))
        else:
            z = torch.tanh(z)
        return F.linear(z, self.down)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def active_linear_macs(self) -> int:
        return 8 * self.width

    def extra_vector_ops(self) -> int:
        # Arithmetic proxy only: two Householder applications, each O(H).
        return 16 * self.width if self.kind == 'mirror' else (3 * self.width if self.kind == 'film' else 0)


class LoopModel(nn.Module):
    """Shared recurrent weights; number of unrolled steps does not change parameter count."""
    def __init__(self, steps: int):
        super().__init__()
        assert steps in (1, 2, 3)
        self.steps = steps
        self.recur = nn.Parameter(torch.randn(D, D) * 0.19)
        self.readout = nn.Parameter(torch.randn(D, D) * 0.19)

    def forward(self, x, rule=None):
        h = x
        for _ in range(self.steps):
            h = h + 0.42 * torch.tanh(F.linear(h, self.recur))
        return F.linear(h - x, self.readout)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def active_linear_macs(self):
        return 16 * self.steps + 16

    def extra_vector_ops(self):
        return 8 * self.steps


class LoopOnePassControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Parameter(torch.randn(D, D) * 0.19)
        self.down = nn.Parameter(torch.randn(D, D) * 0.19)

    def forward(self, x, rule=None):
        return F.linear(torch.tanh(F.linear(x, self.up)), self.down)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def active_linear_macs(self):
        return 32

    def extra_vector_ops(self):
        return 4


def model_from_spec(spec):
    """All initialization seeds are owned by the runner, not by the model."""
    return LoopOnePassControl() if spec['model'] == 'onepass' else (LoopModel(spec['loop_steps']) if spec['model'] == 'loop' else FeatureModel(spec['model'], spec['width']))


def packet_bytes(model) -> int:
    """Exact self-delimiting inference tensor payload: header/schema + all stored FP32 scalars.

    Model implementation is an assumed shared deployment decoder; no runtime
    executable code or optimizer states are included.
    """
    import struct
    kind = {'shared': 0, 'film': 1, 'mirror': 2, 'independent': 3}.get(getattr(model, 'kind', ''), 4 if isinstance(model, LoopModel) else 5)
    width = getattr(model, 'width', D)
    steps = getattr(model, 'steps', 1)
    scalars = sum(p.numel() for p in model.parameters())
    raw = bytearray(b'MA1176\x00\x01')
    raw.extend(struct.pack('<BBBBI', kind, width, steps, RULES, scalars))
    for name, param in model.named_parameters():
        # Exact fixed schema for kind+width+steps: declaration order known to decoder.
        raw.extend(param.detach().contiguous().cpu().numpy().astype('<f4').tobytes())
    assert len(raw) == 16 + 4 * scalars
    return len(raw)
