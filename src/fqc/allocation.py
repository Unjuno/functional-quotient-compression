"""Sensitivity-weighted hard-budget allocation (D62).

This module is the separable special case of D63. Use it only when all shared
mandatory state has already been charged as fixed overhead and block options do
not conditionally open shared paid atoms.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

@dataclass(frozen=True)
class AllocationOption:
    option_id: str
    bits: int
    certified_error: float

@dataclass(frozen=True)
class AllocationResult:
    option_ids: tuple[str,...]
    bits: int
    error: float


def variable_budget(total_budget_bits: int, mandatory_fixed_bits: int) -> int:
    if total_budget_bits<0 or mandatory_fixed_bits<0:
        raise ValueError("bit budgets must be non-negative")
    return total_budget_bits-mandatory_fixed_bits


def prune_dominated(options: Iterable[AllocationOption]) -> tuple[AllocationOption,...]:
    opts=tuple(options); keep=[]
    for i,a in enumerate(opts):
        if a.bits<0: raise ValueError("option bits must be non-negative")
        dominated=False
        for j,b in enumerate(opts):
            if i==j: continue
            if b.bits<=a.bits and b.certified_error<=a.certified_error and (b.bits<a.bits or b.certified_error<a.certified_error):
                dominated=True; break
        if not dominated: keep.append(a)
    return tuple(sorted(keep,key=lambda x:(x.bits,x.certified_error,x.option_id)))


def exact_multiple_choice_allocate(blocks: Sequence[Sequence[AllocationOption]], budget_bits: int) -> AllocationResult|None:
    if budget_bits<0: return None
    best=None
    for choice in product(*(range(len(b)) for b in blocks)):
        selected=[blocks[i][j] for i,j in enumerate(choice)]
        bits=sum(o.bits for o in selected)
        if bits>budget_bits: continue
        err=sum(o.certified_error for o in selected)
        rec=AllocationResult(tuple(o.option_id for o in selected),bits,err)
        key=(err,bits,rec.option_ids)
        if best is None or key<best[0]: best=(key,rec)
    return None if best is None else best[1]


def lagrangian_dual(blocks: Sequence[Sequence[AllocationOption]], budget_bits: int, bit_price: float):
    if bit_price<0: raise ValueError("bit_price must be non-negative")
    total=-bit_price*budget_bits; ids=[]
    for block in blocks:
        if not block: raise ValueError("each block must have an option")
        o=min(block,key=lambda x:(x.certified_error+bit_price*x.bits,x.bits,x.option_id))
        total += o.certified_error+bit_price*o.bits; ids.append(o.option_id)
    return float(total),tuple(ids)


def minimum_bits_for_error_target(blocks: Sequence[Sequence[AllocationOption]], error_target: float) -> AllocationResult|None:
    best=None
    for choice in product(*(range(len(b)) for b in blocks)):
        selected=[blocks[i][j] for i,j in enumerate(choice)]
        err=sum(o.certified_error for o in selected)
        if err>error_target: continue
        bits=sum(o.bits for o in selected)
        rec=AllocationResult(tuple(o.option_id for o in selected),bits,err)
        key=(bits,err,rec.option_ids)
        if best is None or key<best[0]: best=(key,rec)
    return None if best is None else best[1]
