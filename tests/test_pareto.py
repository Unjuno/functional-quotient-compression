from fqc.pareto import ParetoState, best_under_budget, pareto_frontier
from fqc.serializer import byte_aligned_bits


def test_serialized_pareto_can_differ_from_logical_pareto() -> None:
    states = [
        ParetoState(70, 10.0, "a"),
        ParetoState(76, 8.0, "b"),
        ParetoState(79, 7.9, "c"),
        ParetoState(83, 7.0, "d"),
    ]
    raw = pareto_frontier(states)
    serialized = pareto_frontier(states, bit_cost=lambda s: byte_aligned_bits(s.logical_bits))
    assert raw != serialized


def test_hard_budget_uses_serialized_bits() -> None:
    states = [ParetoState(72, 9.0), ParetoState(76, 8.0)]
    best = best_under_budget(states, 80, bit_cost=lambda s: byte_aligned_bits(s.logical_bits))
    assert best is not None
    assert best.logical_bits == 72
