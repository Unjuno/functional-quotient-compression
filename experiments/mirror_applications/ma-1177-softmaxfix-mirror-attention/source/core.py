"""MA-1177 preregistered tiny Mirror attention with hard CE and matched-teacher soft CE tests.

No arbitrary auxiliary loss, no off-label teacher information to the student.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

D, N, R = 8, 12, 3
SEEDS = (1801, 1811, 1823)
WORLDS = ("householder_aligned", "unrelated_orthogonal")
KINDS = ("mirror", "film", "independent")


def house(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    uu = (u * u).sum(-1, keepdim=True).clamp_min(1e-8)
    return x - (2.0 * ((x * u).sum(-1, keepdim=True) / uu)) * u


def orthogonal(gen: torch.Generator) -> torch.Tensor:
    a = torch.randn(D, D, generator=gen)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diag(r)).unsqueeze(0)


def teacher_matrices(world: str, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed * 53 + 7)
    if world == "householder_aligned":
        w = orthogonal(gen)
        v = torch.randn(R, D, generator=gen)
        eye = torch.eye(D)
        return torch.stack([w @ (eye - 2.0 * torch.outer(x, x) / x.dot(x)) for x in v])
    if world == "unrelated_orthogonal":
        return torch.stack([orthogonal(gen) for _ in range(R)])
    raise ValueError(world)


def dataset(world: str, seed: int, split: str, size: int) -> dict:
    seeds = {"train": 19, "dev": 37, "fresh": 71, "ood": 83}
    assert split in seeds
    gen = torch.Generator().manual_seed(seed * 1000 + seeds[split])
    q = torch.randn(size, D, generator=gen)
    k = torch.randn(size, N, D, generator=gen)
    if split == "ood":
        q, k = q * 1.35, k * 1.15
    rule = torch.randint(0, R, (size,), generator=gen)
    with torch.no_grad():
        t = teacher_matrices(world, seed)
        projected = torch.bmm(q.unsqueeze(1), t[rule]).squeeze(1)
        logits = torch.bmm(k, projected.unsqueeze(-1)).squeeze(-1) / math.sqrt(D)
        y = logits.argmax(-1)
    return {"q": q, "k": k, "rule": rule, "target": y}


class TinyAttention(nn.Module):
    def __init__(self, kind: str, seed: int, beta: float):
        super().__init__()
        assert kind in KINDS
        self.kind, self.beta = kind, float(beta)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 34)
            if kind == "independent":
                self.w = nn.Parameter(torch.eye(D)[None].repeat(R, 1, 1) + 0.09 * torch.randn(R, D, D))
            else:
                self.w = nn.Parameter(torch.eye(D) + 0.09 * torch.randn(D, D))
            if kind == "mirror":
                self.u = nn.Parameter(torch.randn(R, D))
            elif kind == "film":
                self.gain = nn.Parameter(torch.ones(R, D) + 0.09 * torch.randn(R, D))
        assert sum(p.numel() for p in self.parameters()) == (192 if kind == "independent" else 88)

    def forward(self, q: torch.Tensor, k: torch.Tensor, rule: torch.Tensor) -> torch.Tensor:
        if self.kind == "independent":
            projected = torch.bmm(q.unsqueeze(1), self.w[rule]).squeeze(1)
        else:
            projected = q @ self.w
            if self.kind == "mirror":
                projected = house(projected, self.u[rule])
            else:
                projected = projected * self.gain[rule]
        return self.beta * torch.bmm(k, projected.unsqueeze(-1)).squeeze(-1) / math.sqrt(D)


def fixed_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Barron et al 2610.31642: hard-target CE as log(1+sum_{j≠t}exp(z_j-z_t)).
    Teacher target logit omitted BEFORE logsumexp; softplus provides stable large and small margins.
    """
    z_t = logits.gather(-1, target.unsqueeze(-1))
    z_dif = logits - z_t
    others = z_dif.scatter(-1, target.unsqueeze(-1), -torch.inf)
    return F.softplus(torch.logsumexp(others, dim=-1)).mean()


def standard_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target)


def log_softmax_fixed(logits: torch.Tensor) -> torch.Tensor:
    """Argmax exclusion / softplus stabilization for gradient at high confidence.
    Only standalone tests use this; it is not patched into attention/SDPA kernels.
    """
    best = logits.argmax(-1, keepdim=True)
    best_logit = logits.gather(-1, best)
    shifted = logits - best_logit
    shifted = shifted.scatter(-1, best, 0.0)
    others = shifted.scatter(-1, best, -torch.inf)
    correction = F.softplus(torch.logsumexp(others, dim=-1, keepdim=True))
    return shifted - correction


def fixed_soft_ce(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Barron Eq. 9, differentiable forward equal to standard soft CE, exact corrected backward.

    Target and student probabilities kept in same fp32 dtype as logits; only the custom
    detached gradient path replaces the inaccurate (separately rounded) q-p subtraction.
    """
    with torch.no_grad():
        p = log_softmax_fixed(teacher.detach()).exp()
        q = log_softmax_fixed(student.detach()).exp()
        delta = student.detach() - teacher.detach()
        d = delta - (p * delta).sum(-1, keepdim=True)
        s = d - torch.log1p((p * torch.expm1(d)).sum(-1, keepdim=True))
        u = torch.where(s.abs() < 1.0, p * torch.expm1(s), q - p)
        g = u - q * u.sum(-1, keepdim=True)
    # Both arms return precisely the same original-forward CE quantity.
    forward_loss = -(teacher.detach().softmax(-1) * F.log_softmax(student, dim=-1)).sum(-1).mean()
    surrogate = (g * (student - student.detach())).sum(-1).mean()
    return forward_loss.detach() + surrogate


def standard_soft_ce(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    p = teacher.detach().softmax(-1)
    return -(p * F.log_softmax(student, dim=-1)).sum(-1).mean()


def hard_target_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    z_t = logits.gather(-1, targets.unsqueeze(-1))
    rest = (logits - z_t).scatter(-1, targets.unsqueeze(-1), -torch.inf)
    return -torch.logsumexp(rest, dim=-1)


def fp64_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(F.cross_entropy(logits.double(), target).detach())


def fp64_distill_kl(s: torch.Tensor, t: torch.Tensor) -> float:
    s, t = s.double(), t.double()
    p = F.softmax(t, dim=-1)
    return float((p * (F.log_softmax(t, -1) - F.log_softmax(s, -1))).sum(-1).mean().detach())
