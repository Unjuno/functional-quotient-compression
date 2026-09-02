from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable


@dataclass(frozen=True)
class QuotientKey:
    """Safe state-equivalence key for optimizer quotienting.

    Equal decoded outputs are not sufficient when candidates differ in future
    capabilities or decoder prerequisites. Both components therefore belong in
    the key.
    """

    decoded: Hashable
    prerequisites: Hashable = ()


@dataclass(frozen=True)
class QuotientCandidate:
    key: QuotientKey
    bits: int
    distortion: float
    payload: object = None


def quotient_candidates(candidates: Iterable[QuotientCandidate]) -> list[QuotientCandidate]:
    """Merge safely equivalent optimizer states.

    Within one quotient class, keep the candidate with the smallest bit cost;
    distortion must be equal because the decoded representation is assumed
    identical. A mismatch is rejected instead of silently merging states whose
    equivalence contract is violated.
    """
    best: dict[QuotientKey, QuotientCandidate] = {}
    for candidate in candidates:
        incumbent = best.get(candidate.key)
        if incumbent is None:
            best[candidate.key] = candidate
            continue
        if candidate.distortion != incumbent.distortion:
            raise ValueError(
                "quotient class contains unequal distortions; decoded equivalence contract violated"
            )
        if candidate.bits < incumbent.bits:
            best[candidate.key] = candidate
    return sorted(best.values(), key=lambda c: (c.bits, c.distortion, repr(c.key)))
