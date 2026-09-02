from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class BundleAssignment:
    states: tuple[int, ...]
    bits: int
    distortion: float
    layout: str


def enumerate_bundle(
    per_leaf_states: Sequence[Sequence[int]],
    *,
    layouts: Iterable[str],
    bit_cost: Callable[[str, tuple[int, ...]], int],
    distortion: Callable[[tuple[int, ...]], float],
) -> list[BundleAssignment]:
    """Exhaustively enumerate a small local codec bundle.

    Distortion is intentionally layout-independent here: this models Type-D
    description-layout moves. Representation-changing layouts must use a
    different distortion callback or invalidate the cached distortion values.
    """
    assignments: list[BundleAssignment] = []
    for state_tuple in product(*per_leaf_states):
        d = distortion(state_tuple)
        for layout in layouts:
            assignments.append(
                BundleAssignment(
                    states=tuple(state_tuple),
                    bits=bit_cost(layout, tuple(state_tuple)),
                    distortion=d,
                    layout=layout,
                )
            )
    return assignments


def best_bundle_under_budget(
    assignments: Iterable[BundleAssignment], budget_bits: int
) -> BundleAssignment | None:
    feasible = [a for a in assignments if a.bits <= budget_bits]
    return min(feasible, key=lambda a: (a.distortion, a.bits, a.layout, a.states)) if feasible else None
