"""Nonlinear top-1 MoE experiment for MA-001 (kept outside nanoGPT)."""
from __future__ import annotations

import io
import json
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


METHODS = ("full_dense", "full_rank2", "tied", "film", "residual2", "mirror")
MODES = ("aligned", "independent")


@dataclass(frozen=True)
class Config:
    input_dim: int = 16
    hidden_dim: int = 32
    output_dim: int = 16
    experts: int = 4
    view_pairs: int = 4
    residual_rank: int = 2
    router_rank: int = 2


def rotate(x: torch.Tensor, angles: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Per-example 2D Givens rotations on paired channels; supports batch angles."""
    pairs = x.shape[-1] // 2
    if angles.shape[-1] > pairs:
        raise ValueError(f"at most {pairs} Givens angles fit this representation, got {angles.shape[-1]}")
    y = x.clone()
    sign = -1.0 if inverse else 1.0
    for k in range(angles.shape[-1]):
        a, b = x[..., 2 * k], x[..., 2 * k + 1]
        c, s = torch.cos(angles[..., k]), sign * torch.sin(angles[..., k])
        y[..., 2 * k] = c * a - s * b
        y[..., 2 * k + 1] = s * a + c * b
    return y


class FFN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.up = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.down = nn.Linear(cfg.hidden_dim, cfg.output_dim)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.up(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.hidden(x))


class Top1MoE(nn.Module):
    """All methods use learned top-1 routing and identical dense router targets."""
    def __init__(self, cfg: Config, method: str):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.cfg, self.method = cfg, method
        if method == "full_rank2":
            self.router_a = nn.Linear(cfg.input_dim, cfg.router_rank, bias=False)
            self.router_b = nn.Linear(cfg.router_rank, cfg.experts)
        else:
            self.router = nn.Linear(cfg.input_dim, cfg.experts)
        count = cfg.experts if method.startswith("full_") else 1
        self.experts = nn.ModuleList([FFN(cfg) for _ in range(count)])
        if method == "film":
            self.gamma = nn.Parameter(torch.zeros(cfg.experts, cfg.hidden_dim))
            self.beta = nn.Parameter(torch.zeros(cfg.experts, cfg.hidden_dim))
        elif method == "residual2":
            self.resid_a = nn.Parameter(torch.empty(cfg.experts, cfg.input_dim, cfg.residual_rank))
            self.resid_b = nn.Parameter(torch.zeros(cfg.experts, cfg.residual_rank, cfg.output_dim))
            nn.init.normal_(self.resid_a, std=0.02)
        elif method == "mirror":
            self.angles = nn.Parameter(torch.zeros(cfg.experts, cfg.view_pairs))

    def route_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "full_rank2":
            return self.router_b(self.router_a(x))
        return self.router(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.route_logits(x)
        role = logits.argmax(dim=-1)
        y = x.new_zeros((x.shape[0], self.cfg.output_dim))
        for r in range(self.cfg.experts):
            mask = role == r
            if not bool(mask.any()):
                continue
            xr = x[mask]
            if self.method.startswith("full_"):
                yr = self.experts[r](xr)
            elif self.method == "mirror":
                code = self.angles[r].expand(xr.shape[0], -1)
                yr = rotate(self.experts[0](rotate(xr, code)), code, inverse=True)
            elif self.method == "film":
                h = self.experts[0].hidden(xr)
                h = h * (1.0 + self.gamma[r]) + self.beta[r]
                yr = self.experts[0].down(h)
            else:
                yr = self.experts[0](xr)
                if self.method == "residual2":
                    yr = yr + (xr @ self.resid_a[r]) @ self.resid_b[r]
            y[mask] = yr
        return y, logits, role

    def serialize(self) -> bytes:
        config = {**asdict(self.cfg), "method": self.method, "top_k": 1, "activation": "GELU"}
        buffer = io.BytesIO()
        torch.save({"state_dict": {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}, "config": config}, buffer)
        return buffer.getvalue()

    def inference_payload_bytes(self) -> int:
        return len(self.serialize())


class Teacher:
    """Fixed four-role FFN teacher, aligned by Givens conjugacy or independent."""
    def __init__(self, cfg: Config, mode: str, seed: int):
        if mode not in MODES:
            raise ValueError(mode)
        self.cfg, self.mode = cfg, mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if mode == "aligned":
                self.base = FFN(cfg).eval()
                self.angles = torch.empty(cfg.experts, cfg.view_pairs).uniform_(-0.8, 0.8)
                self.private = None
            else:
                self.base = None
                self.angles = None
                self.private = nn.ModuleList([FFN(cfg).eval() for _ in range(cfg.experts)])
        modules = [self.base] if self.base is not None else list(self.private)
        for module in modules:
            for param in module.parameters():
                param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, role: torch.Tensor) -> torch.Tensor:
        result = x.new_zeros((x.shape[0], self.cfg.output_dim))
        for r in range(self.cfg.experts):
            mask = role == r
            if not bool(mask.any()):
                continue
            xr = x[mask]
            if self.mode == "aligned":
                code = self.angles[r].expand(xr.shape[0], -1)
                yr = rotate(self.base(rotate(xr, code)), code, inverse=True)
            else:
                yr = self.private[r](xr)
            result[mask] = yr
        return result


def route_labels(x: torch.Tensor) -> torch.Tensor:
    """Linear argmax router realizes the four quadrant labels."""
    a, b = x[:, 0], x[:, 1]
    return torch.stack((a + b, -a + b, -a - b, a - b), dim=-1).argmax(dim=-1)


def active_mac_proxy(cfg: Config, method: str) -> int:
    """One top-1 FFN is active; count router and active affine MACs."""
    ffn = cfg.input_dim * cfg.hidden_dim + cfg.hidden_dim * cfg.output_dim
    router = cfg.input_dim * cfg.experts
    if method == "full_rank2":
        router = cfg.input_dim * cfg.router_rank + cfg.router_rank * cfg.experts
    if method == "residual2":
        ffn += cfg.residual_rank * (cfg.input_dim + cfg.output_dim)
    return ffn + router


def coordinate_flop_proxy(cfg: Config, method: str) -> int:
    # One Givens pair uses four multiplies and two additions per example.
    return 2 * cfg.view_pairs * 6 if method == "mirror" else 0
