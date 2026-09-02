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
    trials=[]; current=tuple(current_pool)
    for root in omitted:
        result=solve_master(current+(root,))
        if result is not None: trials.append((result['error'],result['bits'],root,result))
    return min(trials) if trials else None

def coalition_break_even(fixed_bits: int, private_bits: Sequence[int], shared_assignment_bits: Sequence[int], coalition: Iterable[int]) -> int:
    return sum(private_bits[i]-shared_assignment_bits[i] for i in coalition)-fixed_bits

def pair_overlap_lower_bound(delta_a: float, delta_b: float, bit_price: float, shared_overlap_bits: int) -> float:
    return delta_a + delta_b - bit_price*shared_overlap_bits
