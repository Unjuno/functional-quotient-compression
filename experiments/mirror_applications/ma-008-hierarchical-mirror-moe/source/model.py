"""Two-level token-choice nonlinear MoE for MA-008."""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


METHODS = ("full_flat", "full_hier", "full_rank2_flat", "tied_hier", "film_hier", "residual2_hier", "mirror_hier")
MODES = ("aligned", "independent")


@dataclass(frozen=True)
class Config:
    input_dim: int = 16
    hidden_dim: int = 32
    output_dim: int = 16
    experts: int = 4
    groups: int = 2
    experts_per_group: int = 2
    view_pairs: int = 4
    residual_rank: int = 2
    router_rank: int = 2


def rotate(x: torch.Tensor, angles: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    y = x.clone()
    sign = -1.0 if inverse else 1.0
    for k in range(angles.shape[-1]):
        a, b = x[..., 2*k], x[..., 2*k+1]
        c, s = torch.cos(angles[..., k]), sign * torch.sin(angles[..., k])
        y[..., 2*k] = c*a - s*b
        y[..., 2*k+1] = s*a + c*b
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


class Teacher:
    def __init__(self, cfg: Config, mode: str, seed: int):
        self.cfg, self.mode = cfg, mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if mode == "aligned":
                self.base = FFN(cfg)
                self.angles = torch.empty(cfg.experts, cfg.view_pairs).uniform_(-0.8, 0.8)
                self.private = None
            elif mode == "independent":
                self.base = None
                self.angles = None
                self.private = nn.ModuleList([FFN(cfg) for _ in range(cfg.experts)])
            else:
                raise ValueError(mode)
        for mod in ([self.base] if self.base is not None else list(self.private)):
            for p in mod.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x, labels):
        y = x.new_zeros((x.shape[0], self.cfg.output_dim))
        for e in range(self.cfg.experts):
            rows = torch.where(labels == e)[0]
            xr = x[rows]
            if self.mode == "aligned":
                a = self.angles[e].expand(xr.shape[0], -1)
                yr = rotate(self.base(rotate(xr, a)), a, inverse=True)
            else:
                yr = self.private[e](xr)
            y[rows] = yr
        return y


class HierarchicalMoE(nn.Module):
    def __init__(self, cfg: Config, method: str):
        super().__init__()
        if method not in METHODS: raise ValueError(method)
        self.cfg, self.method = cfg, method
        self.hierarchical = method.endswith("_hier")
        if self.hierarchical:
            self.group_router = nn.Linear(cfg.input_dim, cfg.groups)
            self.child_routers = nn.ModuleList([nn.Linear(cfg.input_dim, cfg.experts_per_group) for _ in range(cfg.groups)])
        elif method == "full_rank2_flat":
            self.router_a = nn.Linear(cfg.input_dim, cfg.router_rank, bias=False)
            self.router_b = nn.Linear(cfg.router_rank, cfg.experts)
        else:
            self.router = nn.Linear(cfg.input_dim, cfg.experts)
        physical = cfg.experts if method.startswith("full_") else 1
        self.experts = nn.ModuleList([FFN(cfg) for _ in range(physical)])
        if method == "film_hier":
            self.gamma = nn.Parameter(torch.zeros(cfg.experts, cfg.hidden_dim))
            self.beta = nn.Parameter(torch.zeros(cfg.experts, cfg.hidden_dim))
        elif method == "residual2_hier":
            self.resid_a = nn.Parameter(torch.empty(cfg.experts, cfg.input_dim, cfg.residual_rank))
            self.resid_b = nn.Parameter(torch.zeros(cfg.experts, cfg.residual_rank, cfg.output_dim))
            nn.init.normal_(self.resid_a, std=0.02)
        elif method == "mirror_hier":
            self.angles = nn.Parameter(torch.zeros(cfg.experts, cfg.view_pairs))

    def route(self, x):
        if self.hierarchical:
            group_logits = self.group_router(x)
            group = group_logits.argmax(-1)
            child_logits = x.new_zeros((x.shape[0], self.cfg.experts_per_group))
            role = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            for g in range(self.cfg.groups):
                rows = torch.where(group == g)[0]
                if rows.numel():
                    child = self.child_routers[g](x[rows])
                    child_logits[rows] = child
                    role[rows] = g*self.cfg.experts_per_group + child.argmax(-1)
            return role, group_logits, child_logits
        if self.method == "full_rank2_flat":
            logits = self.router_b(self.router_a(x))
        else:
            logits = self.router(x)
        return logits.argmax(-1), logits, None

    def forward(self, x):
        role, primary_logits, child_logits = self.route(x)
        y = x.new_zeros((x.shape[0], self.cfg.output_dim))
        for e in range(self.cfg.experts):
            rows = torch.where(role == e)[0]
            if not rows.numel(): continue
            xr = x[rows]
            if self.method.startswith("full_"):
                yr = self.experts[e](xr)
            elif self.method == "mirror_hier":
                a = self.angles[e].expand(xr.shape[0], -1)
                yr = rotate(self.experts[0](rotate(xr, a)), a, inverse=True)
            elif self.method == "film_hier":
                h = self.experts[0].hidden(xr)
                yr = self.experts[0].down(h*(1+self.gamma[e]) + self.beta[e])
            elif self.method == "residual2_hier":
                yr = self.experts[0](xr) + (xr @ self.resid_a[e]) @ self.resid_b[e]
            else:
                yr = self.experts[0](xr)
            y[rows] = yr
        return y, role, primary_logits, child_logits

    def router_loss(self, primary_logits, child_logits, labels):
        if not self.hierarchical:
            return F.cross_entropy(primary_logits, labels)
        groups = labels // self.cfg.experts_per_group
        child = labels % self.cfg.experts_per_group
        group_loss = F.cross_entropy(primary_logits, groups)
        return group_loss + F.cross_entropy(child_logits, child)

    def serialize(self):
        config = {**asdict(self.cfg), "method": self.method, "hierarchical": self.hierarchical, "activation": "GELU"}
        buff = io.BytesIO()
        state = {k:v.detach().cpu().contiguous() for k,v in self.state_dict().items()}
        torch.save({"state_dict":state,"config":config},buff)
        return buff.getvalue()


def router_mac_proxy(cfg: Config, method: str) -> int:
    return (cfg.input_dim*cfg.groups + cfg.input_dim*cfg.experts_per_group
            if method.endswith("_hier") else
            (cfg.input_dim*cfg.router_rank + cfg.router_rank*cfg.experts if "rank2" in method else cfg.input_dim*cfg.experts))


def active_mac_proxy(cfg: Config, method: str) -> int:
    return cfg.input_dim*cfg.hidden_dim + cfg.hidden_dim*cfg.output_dim + router_mac_proxy(cfg, method) + (cfg.residual_rank*(cfg.input_dim+cfg.output_dim) if method.startswith("residual2_") else 0)


def coordinate_flop_proxy(cfg: Config, method: str) -> int:
    return 2*cfg.view_pairs*6 if method == "mirror_hier" else 0
