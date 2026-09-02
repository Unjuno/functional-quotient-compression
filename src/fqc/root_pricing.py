"""Finite shared-root pricing and completeness checks.

Canonicalized from D64-D69. Exact solution of the assignment master over a
known root pool does not certify that the candidate family is complete.
"""
from __future__ import annotations
from itertools import combinations
from typing import Callable, Iterable, Mapping, Sequence, Any


def k_complete_root_sets(root_ids: Sequence[str], max_roots: int):
    yield ()
    for k in range(1,max_roots+1):
        yield from combinations(root_ids,k)


def exact_finite_family_optimum(root_ids: Sequence[str], max_roots: int, solve_master: Callable[[tuple[str,...]], Mapping[str,Any]|None]):
    best=None
    for roots in k_complete_root_sets(root_ids,max_roots):
        result=solve_master(tuple(roots))
        if result is None: continue
        key=(result['error'],result['bits'],tuple(roots))
        if best is None or key < best[0]: best=(key,result)
    return None if best is None else best[1]


def one_column_lookahead(current_pool: Sequence[str], omitted: Iterable[str], solve_master: Callable[[tuple[str,...]], Mapping[str,Any]|None]):
    trials=[]
    current=tuple(current_pool)
    for root in omitted:
        result=solve_master(current+(root,))
        if result is not None:
            trials.append((result['error'],result['bits'],root,result))
    return min(trials) if trials else None


def coalition_break_even(fixed_bits: int, private_bits: Sequence[int], shared_assignment_bits: Sequence[int], coalition: Iterable[int]) -> int:
    return sum(private_bits[i]-shared_assignment_bits[i] for i in coalition)-fixed_bits


def pair_overlap_lower_bound(delta_a: float, delta_b: float, bit_price: float, shared_overlap_bits: int) -> float:
    """D66 lower bound: two single-root reduced costs minus reusable overlap refund."""
    return delta_a + delta_b - bit_price*shared_overlap_bits


def exact_replacement_score(bit_price: float, current_closure_bits: int, new_closure_bits: int, current_block_q: Sequence[float], new_block_q: Sequence[float]) -> float:
    """D67 exact reduced-objective delta for a concrete replacement coalition."""
    if len(current_block_q) != len(new_block_q):
        raise ValueError("block vectors must have equal length")
    return bit_price * (new_closure_bits-current_closure_bits) + sum(n-c for n,c in zip(new_block_q,current_block_q))


def optimistic_refund_lower_bound(bit_price: float, current_closure_bits: int, new_closure_bits: int, current_block_q: Sequence[float], best_new_block_q: Sequence[float]) -> float:
    """D67 safe optimistic replacement lower bound.

    The relaxation illegally allows each old block to remain at zero delta after
    dropping the incumbent root, so it is optimistic and therefore suitable
    only as a lower bound for pruning.
    """
    if len(current_block_q) != len(best_new_block_q):
        raise ValueError("block vectors must have equal length")
    return bit_price*(new_closure_bits-current_closure_bits) + sum(min(0.0,n-c) for n,c in zip(best_new_block_q,current_block_q))


def family_lower_bound(bit_price: float, closure_delta_lower: int, current_q: Sequence[float], optimistic_new_q: Sequence[float]) -> float:
    """D68 separable optimistic lower bound for an entire replacement family."""
    if len(current_q) != len(optimistic_new_q):
        raise ValueError("block vectors must have equal length")
    return bit_price*closure_delta_lower + sum(n-c for n,c in zip(optimistic_new_q,current_q))


def safe_prune(lower_bound: float, tol: float=1e-12) -> bool:
    """Prune a family only when a certified lower bound proves no strict improvement."""
    return lower_bound >= -tol


def grouped_lower_bound(bit_price: float, closure_lower: int, current_q: Sequence[float], legal_coalitions: Sequence[Mapping[str,Any]], partition: Sequence[Sequence[int]]) -> float:
    """D69 compatibility-aware grouped bound.

    Each group's minimum must be attained by a legal concrete coalition; this
    prevents mutually incompatible per-block minima from being combined inside
    the group. Different groups may still use different coalitions, so this is
    a lower bound rather than an exact family value in general.
    """
    if not legal_coalitions:
        raise ValueError("legal_coalitions must be non-empty")
    total=bit_price*closure_lower
    for group in partition:
        total += min(sum(c['block_q'][i]-current_q[i] for i in group) for c in legal_coalitions)
    return total


def closure_coupled_group_bound(bit_price: float, current_q: Sequence[float], legal_coalitions: Sequence[Mapping[str,Any]], first_group: Sequence[int], remaining_groups: Sequence[Sequence[int]]) -> float:
    """D69 bound coupling closure cost to one compatible block group."""
    if not legal_coalitions:
        raise ValueError("legal_coalitions must be non-empty")
    first=min(bit_price*c['closure_delta'] + sum(c['block_q'][i]-current_q[i] for i in first_group) for c in legal_coalitions)
    rest=0.0
    for group in remaining_groups:
        rest += min(sum(c['block_q'][i]-current_q[i] for i in group) for c in legal_coalitions)
    return first+rest
