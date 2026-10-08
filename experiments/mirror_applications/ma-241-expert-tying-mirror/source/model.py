"""Small MoE screens for MA-241. All expert routes are soft/dense for stable CPU training."""
from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    input_dim: int = 16
    output_dim: int = 16
    hidden_dim: int = 32
    experts: int = 4
    layers: int = 2
    residual_rank: int = 2
    view_pairs: int = 4


def rotate(x: torch.Tensor, angles: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Apply independent 2D Givens rotations to the first 2*len(angles) channels."""
    y = x.clone()
    sign = -1.0 if inverse else 1.0
    for pair, angle in enumerate(angles):
        i, j = 2 * pair, 2 * pair + 1
        c, s = torch.cos(angle), sign * torch.sin(angle)
        xi, xj = x[..., i], x[..., j]
        y[..., i] = c * xi - s * xj
        y[..., j] = s * xi + c * xj
    return y


class Expert(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.up = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.down = nn.Linear(cfg.hidden_dim, cfg.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class Bank(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([e(x) for e in self.experts], dim=1)


class MoEViews(nn.Module):
    """Methods: untied, tied, gate, lowrank, mirror."""
    def __init__(self, cfg: Config, method: str):
        super().__init__()
        if method not in {"untied", "tied", "gate", "lowrank", "mirror"}:
            raise ValueError(method)
        self.cfg, self.method = cfg, method
        banks = 2 if method == "untied" else 1
        self.banks = nn.ModuleList([Bank(cfg) for _ in range(banks)])
        # PA01 retains layer-specific routing when experts are tied.
        self.routers = nn.ModuleList([nn.Linear(cfg.input_dim, cfg.experts) for _ in range(cfg.layers)])
        if method == "gate":
            self.gates = nn.Parameter(torch.ones(cfg.layers, cfg.experts))
        if method == "lowrank":
            r = cfg.residual_rank
            self.resid_a = nn.Parameter(torch.randn(cfg.layers, cfg.experts, cfg.input_dim, r) * 0.02)
            self.resid_b = nn.Parameter(torch.zeros(cfg.layers, cfg.experts, r, cfg.output_dim))
        if method == "mirror":
            self.angles = nn.Parameter(torch.zeros(cfg.layers, cfg.view_pairs))

    def forward(self, x: torch.Tensor, layer: torch.Tensor) -> torch.Tensor:
        outputs = []
        for l in range(self.cfg.layers):
            mask = layer == l
            if not bool(mask.any()):
                continue
            xl = x[mask]
            if self.method == "mirror":
                xl_view = rotate(xl, self.angles[l])
                expert_x = xl_view
            else:
                expert_x = xl
            bank = self.banks[l] if self.method == "untied" else self.banks[0]
            expert_out = bank(expert_x)
            if self.method == "mirror":
                expert_out = rotate(expert_out, self.angles[l], inverse=True)
            if self.method == "gate":
                expert_out = expert_out * self.gates[l].view(1, -1, 1)
            if self.method == "lowrank":
                delta = torch.einsum("bd,edr,ero->beo", xl, self.resid_a[l], self.resid_b[l])
                expert_out = expert_out + delta
            weights = torch.softmax(self.routers[l](xl), dim=-1)
            outputs.append((mask, torch.einsum("be,beo->bo", weights, expert_out)))
        y = x.new_zeros((x.shape[0], self.cfg.output_dim))
        for mask, value in outputs:
            y[mask] = value
        return y

    def inference_payload_bytes(self) -> int:
        metadata = {**asdict(self.cfg), "method": self.method}
        cfg_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        weights = io.BytesIO()
        torch.save({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}, weights)
        return len(weights.getvalue()) + len(cfg_bytes)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Teacher(nn.Module):
    """Shared random expert teacher conjugated by independent layer rotations."""
    def __init__(self, cfg: Config, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.cfg = cfg
            self.bank = Bank(cfg)
            self.routers = nn.ModuleList([nn.Linear(cfg.input_dim, cfg.experts) for _ in range(cfg.layers)])
            self.register_buffer("angles", torch.empty(cfg.layers, cfg.view_pairs).uniform_(-0.9, 0.9))
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, layer: torch.Tensor) -> torch.Tensor:
        result = x.new_zeros((x.shape[0], self.cfg.output_dim))
        for l in range(self.cfg.layers):
            mask = layer == l
            if not bool(mask.any()):
                continue
            xl = x[mask]
            xv = rotate(xl, self.angles[l])
            expert_out = rotate(self.bank(xv), self.angles[l], inverse=True)
            w = torch.softmax(self.routers[l](xv), dim=-1)
            result[mask] = torch.einsum("be,beo->bo", w, expert_out)
        return result


def mac_proxy_per_example(cfg: Config, method: str) -> int:
    d, h, e = cfg.input_dim, cfg.hidden_dim, cfg.experts
    # One multiply-add is one MAC. Router and every expert are evaluated densely.
    router = cfg.layers * d * e
    expert = cfg.layers * e * (d * h + h * cfg.output_dim)
    extra = 0
    if method == "mirror":
        extra = cfg.layers * cfg.view_pairs * 12
    elif method == "lowrank":
        extra = cfg.layers * e * cfg.residual_rank * (d + cfg.output_dim)
    elif method == "gate":
        extra = cfg.layers * e * cfg.output_dim
    return router + expert + extra
