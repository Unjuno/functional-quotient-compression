from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class Proposal:
    name: str
    surrogate_score: float
    payload: object = None


@dataclass(frozen=True)
class ValidatedProposal:
    proposal: Proposal
    exact_objective: float


def validate_top_k(
    proposals: Iterable[Proposal],
    *,
    k: int,
    exact_objective: Callable[[Proposal], float],
    incumbent_objective: float,
) -> ValidatedProposal | None:
    """Use a surrogate only to rank; commit only an exact improvement.

    Lower surrogate score and lower exact objective are assumed better.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    ranked = sorted(proposals, key=lambda p: (p.surrogate_score, p.name))[:k]
    validated = [ValidatedProposal(p, exact_objective(p)) for p in ranked]
    improving = [v for v in validated if v.exact_objective < incumbent_objective]
    return min(improving, key=lambda v: (v.exact_objective, v.proposal.name)) if improving else None
