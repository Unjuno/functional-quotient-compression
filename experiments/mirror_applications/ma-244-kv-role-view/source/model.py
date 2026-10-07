"""Synthetic multihead K/V sharing, MQA, gate, and role-view models."""
from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    model_dim: int = 16
    heads: int = 4
    head_dim: int = 4
    memory_tokens: int = 8


def rotate(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """One Givens rotation on channels 0/1 of a projected K or V vector."""
    y = x.clone()
    c, s = torch.cos(angle), torch.sin(angle)
    a, b = x[..., 0], x[..., 1]
    y[..., 0] = c * a - s * b
    y[..., 1] = s * a + c * b
    return y


class AttentionVariant(nn.Module):
    METHODS = ("mha", "kv_equal_mha", "gqa2", "mqa", "mqa_equal", "mirror_mqa", "gate_mqa")

    def __init__(self, cfg: Config, method: str):
        super().__init__()
        if method not in self.METHODS: raise ValueError(method)
        self.cfg, self.method = cfg, method
        if method in ("mha", "kv_equal_mha"):
            groups = cfg.heads
        elif method == "gqa2":
            groups = 2
        else:
            groups = 1
        self.groups = groups
        if method == "kv_equal_mha":
            self.w = nn.Parameter(torch.empty(groups, cfg.model_dim, cfg.head_dim))
        elif method in ("mqa_equal",):
            self.w = nn.Parameter(torch.empty(1, cfg.model_dim, cfg.head_dim))
        elif method in ("mirror_mqa", "gate_mqa"):
            self.w = nn.Parameter(torch.empty(1, cfg.model_dim, cfg.head_dim))
            if method == "mirror_mqa": self.angles = nn.Parameter(torch.zeros(2, cfg.heads))
            else: self.gates = nn.Parameter(torch.ones(2, cfg.heads))
        else:
            self.wk = nn.Parameter(torch.empty(groups, cfg.model_dim, cfg.head_dim))
            self.wv = nn.Parameter(torch.empty(groups, cfg.model_dim, cfg.head_dim))
        for p in self.parameters():
            if p.ndim == 3: nn.init.xavier_uniform_(p)

    def _role_projection(self, mem: torch.Tensor, role: int) -> torch.Tensor:
        raw = mem @ self.w[0]
        if self.method == "mirror_mqa":
            return torch.stack([rotate(raw, self.angles[role, h]) for h in range(self.cfg.heads)], dim=1)
        if self.method == "gate_mqa":
            return torch.stack([raw * self.gates[role, h] for h in range(self.cfg.heads)], dim=1)
        return raw.unsqueeze(1).expand(-1, self.cfg.heads, -1, -1)

    def project_kv(self, mem: torch.Tensor):
        if self.method in ("mirror_mqa", "gate_mqa"):
            k, v = self._role_projection(mem, 0), self._role_projection(mem, 1)
        elif self.method in ("kv_equal_mha", "mqa_equal"):
            k = v = torch.einsum("btd,gdf->bgtf", mem, self.w)
            ids = torch.arange(self.cfg.heads, device=mem.device) % self.groups
            k, v = k[:, ids], v[:, ids]
        else:
            k = torch.einsum("btd,gdf->bgtf", mem, self.wk)
            v = torch.einsum("btd,gdf->bgtf", mem, self.wv)
            ids = torch.arange(self.cfg.heads, device=mem.device) % self.groups
            k, v = k[:, ids], v[:, ids]
        return k, v

    def forward(self, query: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        k, v = self.project_kv(mem)
        scores = (query.unsqueeze(2) * k).sum(dim=-1) / (self.cfg.head_dim ** 0.5)
        weights = scores.softmax(dim=-1)
        out = (weights.unsqueeze(-1) * v).sum(dim=2)
        return out.reshape(query.shape[0], self.cfg.heads * self.cfg.head_dim)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def inference_payload_bytes(self):
        meta = json.dumps({**asdict(self.cfg), "method": self.method}, sort_keys=True, separators=(",", ":")).encode()
        buf = io.BytesIO()
        torch.save({k:v.detach().cpu().contiguous() for k,v in self.state_dict().items()}, buf)
        return len(meta) + len(buf.getvalue())

    def cache_bytes(self, tokens: int | None = None):
        t = tokens or self.cfg.memory_tokens
        if self.method == "mha": groups, roles = self.cfg.heads, 2
        elif self.method == "kv_equal_mha": groups, roles = self.cfg.heads, 1
        elif self.method == "gqa2": groups, roles = 2, 2
        elif self.method == "mqa": groups, roles = 1, 2
        else: groups, roles = 1, 1
        return groups * roles * t * self.cfg.head_dim * 4

    def materialized_cache(self, mem: torch.Tensor):
        """Return the physical cache tensors implied by this parameterization."""
        if self.method in ("mirror_mqa", "gate_mqa", "mqa_equal"):
            return (mem @ self.w[0],)
        if self.method == "kv_equal_mha":
            k, _ = self.project_kv(mem)
            return (k,)
        if self.method == "mha":
            return self.project_kv(mem)
        if self.method == "gqa2":
            return (torch.einsum("btd,gdf->bgtf", mem, self.wk), torch.einsum("btd,gdf->bgtf", mem, self.wv))
        if self.method == "mqa":
            return (mem @ self.wk[0], mem @ self.wv[0])
        raise ValueError(self.method)

    def measured_cache_bytes(self, mem: torch.Tensor):
        return sum(t.numel() * t.element_size() for t in self.materialized_cache(mem))


class Teacher(nn.Module):
    """One shared projection with independent role/head Givens transforms."""
    def __init__(self, cfg: Config, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.cfg = cfg
        self.register_buffer("w", torch.empty(1, cfg.model_dim, cfg.head_dim).normal_(0, 0.25, generator=g))
        self.register_buffer("angles", torch.empty(2, cfg.heads).uniform_(-0.8, 0.8, generator=g))

    @torch.no_grad()
    def forward(self, query: torch.Tensor, mem: torch.Tensor):
        raw = mem @ self.w[0]
        k = torch.stack([rotate(raw, self.angles[0,h]) for h in range(self.cfg.heads)], dim=1)
        v = torch.stack([rotate(raw, self.angles[1,h]) for h in range(self.cfg.heads)], dim=1)
        scores = (query.unsqueeze(2) * k).sum(-1) / (self.cfg.head_dim ** 0.5)
        return ((scores.softmax(-1).unsqueeze(-1) * v).sum(2)).reshape(query.shape[0], -1)


def mac_proxy_per_example(cfg: Config, method: str):
    projection_groups = {"mha":8,"kv_equal_mha":4,"gqa2":4,"mqa":2,"mqa_equal":1,"mirror_mqa":1,"gate_mqa":1}[method]
    projection = cfg.memory_tokens * cfg.model_dim * cfg.head_dim * projection_groups
    attention = cfg.heads * cfg.memory_tokens * cfg.head_dim * 2
    if method == "mirror_mqa": projection += 2 * cfg.heads * cfg.memory_tokens * 8
    if method == "gate_mqa": projection += 2 * cfg.heads * cfg.memory_tokens * cfg.head_dim
    return projection + attention
