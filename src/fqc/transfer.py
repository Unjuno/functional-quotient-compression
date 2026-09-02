from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable


@dataclass(frozen=True)
class MaskState:
    mask: int
    bits: int
    distortion: float


def direct_rotated_optimum(
    states: Iterable[MaskState],
    *,
    budget_bits: int,
    delta_bits: Callable[[int], int],
) -> MaskState | None:
    feasible = [s for s in states if s.bits + delta_bits(s.mask) <= budget_bits]
    return min(feasible, key=lambda s: (s.distortion, s.bits + delta_bits(s.mask), s.mask)) if feasible else None


def conditional_frontier_transfer(
    states: Iterable[MaskState],
    *,
    budget_bits: int,
    local_signature: Callable[[int], Hashable],
    delta_for_signature: Callable[[Hashable], int],
) -> MaskState | None:
    """Exact description-only transfer under tree-independent distortion.

    For each local signature s, compute the best base-tree state available under
    the shifted budget B-delta(s), then minimize distortion over signatures.
    The theorem guarantees the optimum distortion value; tied argmin masks need
    not match a direct enumerator's tie-breaking rule.
    """
    groups: dict[Hashable, list[MaskState]] = defaultdict(list)
    for state in states:
        groups[local_signature(state.mask)].append(state)

    candidates: list[MaskState] = []
    for signature, group in groups.items():
        shifted_budget = budget_bits - delta_for_signature(signature)
        feasible = [state for state in group if state.bits <= shifted_budget]
        if feasible:
            candidates.append(min(feasible, key=lambda s: (s.distortion, s.bits, s.mask)))
    return min(candidates, key=lambda s: (s.distortion, s.bits, s.mask)) if candidates else None
