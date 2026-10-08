from __future__ import annotations
import io
import json
import torch
from torch import nn

METHODS = ["full_moe", "tied_expert", "scalar_gate", "lowrank_expert", "mirror"]
N, D, O, RANK = 4, 16, 12, 1


def signed_coefficients(x: torch.Tensor) -> torch.Tensor:
    a = torch.where(x[:, 0] >= 0, 1.0, -1.0)
    b = torch.where(x[:, 1] >= 0, 1.0, -1.0)
    return torch.stack((torch.ones_like(a), a, b, a * b), dim=1) * 0.5


def givens(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply one learnable rotation to each of eight disjoint coordinate pairs."""
    y = x
    for k in range(D // 2):
        i, j = 2 * k, 2 * k + 1
        c, s = torch.cos(angles[..., k]), torch.sin(angles[..., k])
        vi, vj = y[..., i], y[..., j]
        ni, nj = c * vi - s * vj, s * vi + c * vj
        y = torch.cat((y[..., :i], ni.unsqueeze(-1), nj.unsqueeze(-1), y[..., j + 1:]), dim=-1)
    return y


class SignedExperts(nn.Module):
    def __init__(self, method: str, seed: int = 0):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        g = torch.Generator().manual_seed(seed)
        if method == "full_moe":
            self.weight = nn.Parameter(torch.randn(N, D, O, generator=g) * 0.08)
        else:
            self.weight = nn.Parameter(torch.randn(D, O, generator=g) * 0.08)
            if method == "scalar_gate":
                self.amplitude = nn.Parameter(torch.ones(N))
            elif method == "lowrank_expert":
                self.u = nn.Parameter(torch.randn(N, D, RANK, generator=g) * 0.05)
                self.v = nn.Parameter(torch.zeros(N, RANK, O))
            elif method == "mirror":
                self.angles = nn.Parameter(torch.zeros(N, D // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = signed_coefficients(x)
        if self.method == "full_moe":
            components = torch.einsum("bd,ndo->bno", x, self.weight)
        elif self.method == "mirror":
            views = torch.stack([givens(x, angle.expand(x.shape[0], -1)) for angle in self.angles], dim=1)
            components = torch.einsum("bnd,do->bno", views, self.weight)
        else:
            components = torch.einsum("bd,do->bo", x, self.weight)[:, None, :].expand(-1, N, -1)
            if self.method == "scalar_gate":
                components = components * self.amplitude[None, :, None]
            elif self.method == "lowrank_expert":
                residual = torch.einsum("bd,ndr,nro->bno", x, self.u, self.v)
                components = components + residual
        return torch.einsum("bn,bno->bo", c, components)

    def serialized_payload_bytes(self) -> int:
        buf = io.BytesIO()
        torch.save({"state_dict": self.state_dict(), "config": json.dumps({"method": self.method, "n": N, "d": D, "o": O, "rank": RANK}, sort_keys=True)}, buf)
        return len(buf.getvalue())


def compute_proxy(method: str, examples: int) -> int:
    # Dense MAC proxy; Givens rotations add D MAC-equivalents per view and example.
    if method == "full_moe":
        per = N * D * O + N * O
    elif method == "mirror":
        per = N * (D * O + D) + N * O
    elif method == "lowrank_expert":
        per = D * O + N * (D + O) * RANK + N * O
    else:
        per = D * O + N * O
    return examples * per
