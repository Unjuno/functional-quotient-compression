from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x), approximate="tanh"))


class MirrorSharedFFN(nn.Module):
    """Shared FFN plus sparse per-rule Mirror residual coordinates.

    A rule does not own an independent FFN. It stores only 2*rank coefficients:
    rank stretch coordinates and rank shear coordinates. Multiple selected rules
    are evaluated around the same shared W1/W2 and their residuals are composed.
    """

    def __init__(
        self,
        n_rules: int,
        d_model: int,
        d_ff: int,
        rank: int,
        code_max: float = 1.0,
    ):
        super().__init__()
        if 2 * rank > d_ff:
            raise ValueError("2*rank must be <= d_ff")
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.rank = rank
        self.code_max = float(code_max)

        # Only these coordinates are private to each rule.
        self.raw_code = nn.Parameter(torch.zeros(n_rules, 2 * rank))

        # Fixed structured basis. It is deterministically reconstructible.
        self.register_buffer("src", torch.arange(rank), persistent=False)
        self.register_buffer("dst", torch.arange(rank, 2 * rank), persistent=False)

    def forward(self, x: torch.Tensor, rule_ids: torch.Tensor) -> torch.Tensor:
        # x: [batch, token, d_model]
        # rule_ids: [batch, k], where k may be > 1.
        v = self.w1(x)
        base_act = F.gelu(v, approximate="tanh")
        base = self.w2(base_act)

        code = self.code_max * torch.tanh(self.raw_code[rule_ids])
        stretch, shear = code[..., : self.rank], code[..., self.rank :]

        q = v[:, None].expand(-1, rule_ids.shape[1], -1, -1).clone()
        e = torch.exp(stretch)[:, :, None, :]

        src = q[..., self.src].clone()
        dst = q[..., self.dst].clone()

        # Q = shear * stretch
        src_q = src * e
        dst_q = dst / e + shear[:, :, None, :] * src_q
        q[..., self.src] = src_q
        q[..., self.dst] = dst_q

        act_q = F.gelu(q, approximate="tanh")

        # Q^{-1}: inverse shear, then inverse stretch.
        src_a = act_q[..., self.src].clone()
        dst_a = act_q[..., self.dst].clone()
        dst_inv = (dst_a - shear[:, :, None, :] * src_a) * e
        src_inv = src_a / e

        inv = act_q.clone()
        inv[..., self.src] = src_inv
        inv[..., self.dst] = dst_inv

        # Compose multiple active shared rules sparsely.
        delta = (inv - base_act[:, None]).mean(dim=1)
        return base + self.w2(delta)


class LearnedSoftMoE(nn.Module):
    """Stronger full-expert control used in SRM001.

    The router may mix every available independent expert, so the standard-MoE
    control is not limited by the deterministic hash routing used in the first
    panel.
    """

    def __init__(self, n_experts: int, d_model: int, d_ff: int):
        super().__init__()
        self.experts = nn.ModuleList(
            [FFN(d_model, d_ff) for _ in range(n_experts)]
        )
        self.router = nn.Linear(d_model, n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.router(x[:, 0]), dim=-1)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return (outputs * weights[:, :, None, None]).sum(dim=1)
