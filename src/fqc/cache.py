from __future__ import annotations

from collections import defaultdict
from typing import Callable, Hashable, Iterable

from .pareto import ParetoState, pareto_frontier


def conditional_pareto_cache(
    states: Iterable[ParetoState],
    *,
    signature: Callable[[ParetoState], Hashable],
    bit_cost: Callable[[ParetoState], int] | None = None,
) -> dict[Hashable, list[ParetoState]]:
    """Build exact bit/distortion Pareto frontiers per local signature."""
    groups: dict[Hashable, list[ParetoState]] = defaultdict(list)
    for state in states:
        groups[signature(state)].append(state)
    return {
        key: pareto_frontier(group, bit_cost=bit_cost)
        for key, group in groups.items()
    }
