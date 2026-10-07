"""MA-253 synthetic domain-update models and a causal K/V placement probe."""
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
    domains: int = 4
    rank: int = 2
    private_rank: int = 1
    view_pairs: int = 4


def rotate(x: torch.Tensor, angles: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    y = x.clone()
    sign = -1.0 if inverse else 1.0
    for pair, angle in enumerate(angles):
        i, j = 2 * pair, 2 * pair + 1
        c, s = torch.cos(angle), sign * torch.sin(angle)
        xi, xj = x[..., i], x[..., j]
        y[..., i] = c * xi - s * xj
        y[..., j] = s * xi + c * xj
    return y


class DomainUpdate(nn.Module):
    METHODS = ("base", "dmoe_lora_r2", "mirror", "mirror_private_r1", "independent")

    def __init__(self, cfg: Config, method: str, base: torch.Tensor):
        super().__init__()
        if method not in self.METHODS:
            raise ValueError(method)
        self.cfg, self.method = cfg, method
        self.base = nn.Parameter(base.detach().clone())
        if method == "dmoe_lora_r2":
            self.a = nn.Parameter(torch.randn(cfg.domains, cfg.input_dim, cfg.rank) * 0.01)
            self.b = nn.Parameter(torch.zeros(cfg.domains, cfg.rank, cfg.output_dim))
        elif method == "mirror":
            self.angles = nn.Parameter(torch.zeros(cfg.domains, cfg.view_pairs))
        elif method == "mirror_private_r1":
            self.angles = nn.Parameter(torch.zeros(cfg.domains, cfg.view_pairs))
            self.private_a = nn.Parameter(torch.randn(cfg.domains, cfg.input_dim, cfg.private_rank) * 0.01)
            self.private_b = nn.Parameter(torch.zeros(cfg.domains, cfg.private_rank, cfg.output_dim))
        elif method == "independent":
            self.delta = nn.Parameter(torch.zeros(cfg.domains, cfg.input_dim, cfg.output_dim))

    def forward(self, x: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        y = x @ self.base
        if self.method in ("mirror", "mirror_private_r1"):
            for d in range(self.cfg.domains):
                mask = domain == d
                if bool(mask.any()):
                    xv = rotate(x[mask], self.angles[d])
                    y[mask] = rotate(xv @ self.base, self.angles[d], inverse=True)
        if self.method == "dmoe_lora_r2":
            y = y + torch.einsum("bd,bdr,bro->bo", x, self.a[domain], self.b[domain])
        elif self.method == "mirror_private_r1":
            y = y + torch.einsum("bd,bdr,bro->bo", x, self.private_a[domain], self.private_b[domain])
        elif self.method == "independent":
            y = y + torch.einsum("bd,bdo->bo", x, self.delta[domain])
        return y

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def inference_payload_bytes(self) -> int:
        meta = json.dumps({**asdict(self.cfg), "method": self.method}, sort_keys=True, separators=(",", ":")).encode()
        payload = io.BytesIO()
        torch.save({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}, payload)
        return len(meta) + len(payload.getvalue())


class CacheProbeBlock(nn.Module):
    def __init__(self, dim: int, domains: int, pairs: int, seed: int):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.up = nn.Linear(dim, 2 * dim)
        self.down = nn.Linear(2 * dim, dim)
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("view_angles", torch.empty(domains, pairs).uniform_(-0.8, 0.8, generator=g))

    def forward(self, x: torch.Tensor, domain: int, enable_view: bool):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = q @ k.transpose(-1, -2) / (d ** 0.5)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        weights = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        h = x + self.proj(weights @ v)
        if enable_view:
            hv = rotate(h, self.view_angles[domain])
            ff = self.down(F.gelu(self.up(hv)))
            ff = rotate(ff, self.view_angles[domain], inverse=True)
        else:
            ff = self.down(F.gelu(self.up(h)))
        return h + ff, (k.detach().clone(), v.detach().clone())


class TinyCacheDecoder(nn.Module):
    """Two-layer causal decoder; expert/view transformation is applied after each block's attention."""
    def __init__(self, dim: int = 16, domains: int = 2, pairs: int = 4, seed: int = 100):
        super().__init__()
        self.layers = nn.ModuleList([CacheProbeBlock(dim, domains, pairs, seed + i) for i in range(2)])

    @torch.no_grad()
    def forward(self, x: torch.Tensor, domain: int, placement: str):
        if placement not in ("final_only", "early_and_final"):
            raise ValueError(placement)
        caches = []
        h = x
        for i, block in enumerate(self.layers):
            active = (placement == "final_only" and i == 1) or placement == "early_and_final"
            h, kv = block(h, domain, active)
            caches.append(kv)
        return h, caches


def mac_proxy_per_example(cfg: Config, method: str) -> int:
    d = cfg.input_dim
    macs = d * d
    if method == "dmoe_lora_r2":
        macs += cfg.rank * (d + cfg.output_dim)
    elif method == "mirror":
        macs += cfg.view_pairs * 12
    elif method == "mirror_private_r1":
        macs += cfg.view_pairs * 12 + cfg.private_rank * (d + cfg.output_dim)
    elif method == "independent":
        macs += d * cfg.output_dim
    return macs
