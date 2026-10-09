"""Nonlinear sparse top-2 expert experiment for MA-002."""
from __future__ import annotations

import io
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
    top_k: int = 4
    view_pairs: int = 4
    residual_rank: int = 2
    router_rank: int = 2


def rotate(x: torch.Tensor, angles: torch.Tensor, inverse: bool = False) -> torch.Tensor:
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


class Teacher:
    """Frozen top-2 router and either aligned or independent role FFNs."""
    def __init__(self, cfg: Config, mode: str, seed: int):
        if mode not in MODES:
            raise ValueError(mode)
        self.cfg, self.mode = cfg, mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.router = nn.Linear(cfg.input_dim, cfg.experts)
            if mode == "aligned":
                self.base = FFN(cfg)
                self.angles = torch.empty(cfg.experts, cfg.view_pairs).uniform_(-0.8, 0.8)
                self.private = None
            else:
                self.base = None
                self.angles = None
                self.private = nn.ModuleList([FFN(cfg) for _ in range(cfg.experts)])
        for p in self.router.parameters():
            p.requires_grad_(False)
        modules = [self.base] if self.base is not None else list(self.private)
        for module in modules:
            for p in module.parameters():
                p.requires_grad_(False)

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(x)
        probs = logits.softmax(dim=-1)
        values, indices = logits.topk(self.cfg.top_k, dim=-1)
        weights = values.softmax(dim=-1)
        return probs, indices, weights

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, indices, weights = self.route(x)
        result = x.new_zeros((x.shape[0], self.cfg.output_dim))
        for role in range(self.cfg.experts):
            rows, slots = torch.where(indices == role)
            if rows.numel() == 0:
                continue
            xr = x[rows]
            if self.mode == "aligned":
                code = self.angles[role].expand(xr.shape[0], -1)
                yr = rotate(self.base(rotate(xr, code)), code, inverse=True)
            else:
                yr = self.private[role](xr)
            result = result.index_add(0, rows, yr * weights[rows, slots, None])
        return result, probs, indices


class Top2MoE(nn.Module):
    """Sparse top-2 router; selected expert outputs use normalized top-2 weights."""
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
        n_physical = cfg.experts if method.startswith("full_") else 1
        self.experts = nn.ModuleList([FFN(cfg) for _ in range(n_physical)])
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.route_logits(x)
        values, indices = logits.topk(self.cfg.top_k, dim=-1)
        weights = values.softmax(dim=-1)
        output = x.new_zeros((x.shape[0], self.cfg.output_dim))
        if self.method == "tied":
            # Both selected roles are identical. Their normalized weights sum to one,
            # so hard tying requires only one active FFN call.
            output = self.experts[0](x)
        else:
            for role in range(self.cfg.experts):
                rows, slots = torch.where(indices == role)
                if rows.numel() == 0:
                    continue
                xr = x[rows]
                if self.method.startswith("full_"):
                    yr = self.experts[role](xr)
                elif self.method == "mirror":
                    code = self.angles[role].expand(xr.shape[0], -1)
                    yr = rotate(self.experts[0](rotate(xr, code)), code, inverse=True)
                elif self.method == "film":
                    h = self.experts[0].hidden(xr)
                    yr = self.experts[0].down(h * (1.0 + self.gamma[role]) + self.beta[role])
                else:
                    yr = self.experts[0](xr)
                    if self.method == "residual2":
                        yr = yr + (xr @ self.resid_a[role]) @ self.resid_b[role]
                output = output.index_add(0, rows, yr * weights[rows, slots, None])
        return output, logits, indices, weights

    def serialize(self) -> bytes:
        config = {**asdict(self.cfg), "method": self.method, "activation": "GELU"}
        buffer = io.BytesIO()
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        torch.save({"state_dict": state, "config": config}, buffer)
        return buffer.getvalue()


def active_mac_proxy(cfg: Config, method: str) -> int:
    ffn = cfg.input_dim * cfg.hidden_dim + cfg.hidden_dim * cfg.output_dim
    active_experts = 1 if method == "tied" else cfg.top_k
    router = cfg.input_dim * cfg.experts
    if method == "full_rank2":
        router = cfg.input_dim * cfg.router_rank + cfg.router_rank * cfg.experts
    total = active_experts * ffn + router
    if method == "residual2":
        total += cfg.top_k * cfg.residual_rank * (cfg.input_dim + cfg.output_dim)
    return total


def coordinate_flop_proxy(cfg: Config, method: str) -> int:
    return 2 * cfg.top_k * cfg.view_pairs * 6 if method == "mirror" else 0
