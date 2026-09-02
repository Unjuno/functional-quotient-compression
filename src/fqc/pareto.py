from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class ParetoState:
    """A codec candidate for hard-budget optimization."""

    logical_bits: int
    distortion: float
    payload: Any = field(default=None, compare=False)


def pareto_frontier(
    states: Iterable[ParetoState],
    *,
    bit_cost: Callable[[ParetoState], int] | None = None,
) -> list[ParetoState]:
    """Return exact non-dominated states under bit-cost/distortion ordering.

    State a dominates state b when it is no worse in both bit cost and
    distortion and strictly better in at least one. Equal-cost states keep only
    the lowest-distortion representative.
    """
    bit_cost = bit_cost or (lambda s: s.logical_bits)
    ordered = sorted(states, key=lambda s: (bit_cost(s), s.distortion))
    out: list[ParetoState] = []
    best_distortion = float("inf")
    seen_cost: int | None = None

    for state in ordered:
        cost = bit_cost(state)
        if seen_cost == cost:
            continue
        seen_cost = cost
        if state.distortion < best_distortion:
            out.append(state)
            best_distortion = state.distortion
    return out


def best_under_budget(
    states: Iterable[ParetoState],
    budget_bits: int,
    *,
    bit_cost: Callable[[ParetoState], int] | None = None,
) -> ParetoState | None:
    """Return minimum-distortion feasible state under an exact bit budget."""
    if budget_bits < 0:
        raise ValueError("budget_bits must be non-negative")
    bit_cost = bit_cost or (lambda s: s.logical_bits)
    feasible = [s for s in states if bit_cost(s) <= budget_bits]
    return min(feasible, key=lambda s: (s.distortion, bit_cost(s))) if feasible else None
