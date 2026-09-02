from fqc.cache import conditional_pareto_cache
from fqc.pareto import ParetoState, best_under_budget


def test_conditional_pareto_cache_preserves_exact_budget_optimum_per_signature() -> None:
    states = [
        ParetoState(bits, distortion=float(30 - value), payload=(sig, value))
        for sig in range(4)
        for value, bits in ((1, 5 + sig), (2, 7 + sig), (3, 10 + sig), (4, 14 + sig))
    ]
    cache = conditional_pareto_cache(states, signature=lambda s: s.payload[0])
    for sig in range(4):
        original = [s for s in states if s.payload[0] == sig]
        for budget in range(4, 20):
            assert best_under_budget(cache[sig], budget) == best_under_budget(original, budget)
