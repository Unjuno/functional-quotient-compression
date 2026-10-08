"""Capacity-balanced expert-choice nonlinear MoE for MA-006."""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


METHODS = ("full_ec", "full_token", "full_rank2_token", "tied_token", "film_token", "residual2_token", "mirror_token", "mirror_ec")
MODES = ("aligned", "independent")


@dataclass(frozen=True)
class Config:
    input_dim: int = 16
    hidden_dim: int = 32
    output_dim: int = 16
    experts: int = 4
    capacity_per_expert: int = 16
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

    def hidden(self, x):
        return F.gelu(self.up(x))

    def forward(self, x):
        return self.down(self.hidden(x))


def assignments(logits: torch.Tensor, policy: str, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a token-by-expert mask and per-token mixture weights."""
    batch, experts = logits.shape
    if policy == "token":
        idx = logits.argmax(dim=-1)
        mask = F.one_hot(idx, num_classes=experts).bool()
    elif policy == "expert_choice":
        if batch % experts:
            raise ValueError("expert-choice batch size must divide evenly by expert count")
        # Keep the registered capacity ratio of 1/experts at train, validation,
        # and inference batch sizes. Each expert therefore receives exactly
        # batch/experts dispatch slots.
        cap = batch // experts
        if cap == 0:
            raise ValueError("expert-choice batch must contain at least one slot per expert")
        mask = torch.zeros_like(logits, dtype=torch.bool)
        for e in range(experts):
            selected = logits[:, e].topk(cap).indices
            mask[selected, e] = True
    else:
        raise ValueError(policy)
    no_route = ~mask.any(dim=-1)
    active_logits = logits.masked_fill(~mask, -torch.inf)
    active_logits = torch.where(no_route[:, None], torch.zeros_like(logits), active_logits)
    weights = active_logits.softmax(dim=-1)
    weights = torch.where(no_route[:, None], torch.zeros_like(weights), weights)
    return mask, weights


class Teacher:
    def __init__(self, cfg: Config, mode: str, seed: int):
        if mode not in MODES:
            raise ValueError(mode)
        self.cfg, self.mode = cfg, mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if mode == "aligned":
                self.base = FFN(cfg)
                self.angles = torch.empty(cfg.experts, cfg.view_pairs).uniform_(-0.8, 0.8)
                self.private = None
            else:
                self.base = None
                self.angles = None
                self.private = nn.ModuleList([FFN(cfg) for _ in range(cfg.experts)])
        modules = [self.base] if self.base is not None else list(self.private)
        for module in modules:
            for p in module.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        out = x.new_zeros((x.shape[0], self.cfg.output_dim))
        for role in range(self.cfg.experts):
            rows = torch.where(labels == role)[0]
            xr = x[rows]
            if self.mode == "aligned":
                code = self.angles[role].expand(xr.shape[0], -1)
                yr = rotate(self.base(rotate(xr, code)), code, inverse=True)
            else:
                yr = self.private[role](xr)
            out[rows] = yr
        return out


class ExpertChoiceMoE(nn.Module):
    def __init__(self, cfg: Config, method: str):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.cfg, self.method = cfg, method
        self.policy = "token" if method.endswith("_token") else "expert_choice"
        if "rank2" in method:
            self.router_a = nn.Linear(cfg.input_dim, cfg.router_rank, bias=False)
            self.router_b = nn.Linear(cfg.router_rank, cfg.experts)
        else:
            self.router = nn.Linear(cfg.input_dim, cfg.experts)
        n_physical = cfg.experts if method.startswith("full_") else 1
        self.experts = nn.ModuleList([FFN(cfg) for _ in range(n_physical)])
        if method.startswith("film_"):
            self.gamma = nn.Parameter(torch.zeros(cfg.experts, cfg.hidden_dim))
            self.beta = nn.Parameter(torch.zeros(cfg.experts, cfg.hidden_dim))
        elif method.startswith("residual2_"):
            self.resid_a = nn.Parameter(torch.empty(cfg.experts, cfg.input_dim, cfg.residual_rank))
            self.resid_b = nn.Parameter(torch.zeros(cfg.experts, cfg.residual_rank, cfg.output_dim))
            nn.init.normal_(self.resid_a, std=0.02)
        elif method.startswith("mirror_"):
            self.angles = nn.Parameter(torch.zeros(cfg.experts, cfg.view_pairs))

    def route_logits(self, x):
        if "rank2" in self.method:
            return self.router_b(self.router_a(x))
        return self.router(x)

    def forward(self, x):
        logits = self.route_logits(x)
        mask, weights = assignments(logits, self.policy, self.cfg.capacity_per_expert)
        out = x.new_zeros((x.shape[0], self.cfg.output_dim))
        if self.method.startswith("tied_"):
            active = mask.any(dim=-1)
            if bool(active.any()):
                out[active] = self.experts[0](x[active])
        else:
            for role in range(self.cfg.experts):
                rows = torch.where(mask[:, role])[0]
                if rows.numel() == 0:
                    continue
                xr = x[rows]
                if self.method.startswith("full_"):
                    yr = self.experts[role](xr)
                elif self.method.startswith("mirror_"):
                    code = self.angles[role].expand(xr.shape[0], -1)
                    yr = rotate(self.experts[0](rotate(xr, code)), code, inverse=True)
                elif self.method.startswith("film_"):
                    h = self.experts[0].hidden(xr)
                    yr = self.experts[0].down(h * (1 + self.gamma[role]) + self.beta[role])
                else:
                    yr = self.experts[0](xr) + (xr @ self.resid_a[role]) @ self.resid_b[role]
                out = out.index_add(0, rows, yr * weights[rows, role, None])
        return out, logits, mask, weights

    def serialize(self) -> bytes:
        config = {**asdict(self.cfg), "method": self.method, "routing": self.policy, "activation": "GELU"}
        buff = io.BytesIO()
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        torch.save({"state_dict": state, "config": config}, buff)
        return buff.getvalue()


def active_mac_proxy(cfg: Config, method: str, mean_assignments: float = 1.0) -> int:
    ffn = cfg.input_dim * cfg.hidden_dim + cfg.hidden_dim * cfg.output_dim
    router = cfg.input_dim * cfg.experts
    if "rank2" in method:
        router = cfg.input_dim * cfg.router_rank + cfg.router_rank * cfg.experts
    total = round(mean_assignments * ffn) + router
    if method.startswith("residual2_"):
        total += round(mean_assignments * cfg.residual_rank * (cfg.input_dim + cfg.output_dim))
    return total


def coordinate_flop_proxy(cfg: Config, method: str, mean_assignments: float = 1.0) -> int:
    return round(mean_assignments * 2 * cfg.view_pairs * 6) if method.startswith("mirror_") else 0
