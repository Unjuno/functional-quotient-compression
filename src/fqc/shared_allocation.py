"""Exact small-instance allocation with shared paid atoms.

Canonicalized from D63. This is a fixed-charge/dependency-aware
multiple-choice problem: private payloads are separable, shared atom costs are
paid once over the union of prerequisite closures.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence
from .decoder_dag import Atom, compile_decoder_dag

@dataclass(frozen=True)
class BlockOption:
    option_id: str
    private_bits: int
    error: float
    requires: tuple[str, ...] = ()

@dataclass(frozen=True)
class Allocation:
    option_ids: tuple[str, ...]
    private_bits: int
    shared_paid_bits: int
    total_bits: int
    error: float
    active_atoms: frozenset[str]

def evaluate_selection(blocks: Sequence[Sequence[BlockOption]], atoms: Iterable[Atom], selection: Sequence[int]) -> Allocation:
    if len(selection)!=len(blocks): raise ValueError("selection length must match block count")
    chosen=[blocks[i][j] for i,j in enumerate(selection)]
    if any(o.private_bits<0 for o in chosen): raise ValueError("private_bits must be non-negative")
    private=sum(o.private_bits for o in chosen)
    required=sorted({a for o in chosen for a in o.requires})
    compiled=compile_decoder_dag(atoms,required)
    if not compiled.valid: raise ValueError("invalid shared atom DAG: "+"; ".join(compiled.errors))
    return Allocation(tuple(o.option_id for o in chosen),private,compiled.total_paid_bits,private+compiled.total_paid_bits,sum(o.error for o in chosen),compiled.reachable_atoms)

def exact_shared_allocate(blocks: Sequence[Sequence[BlockOption]], atoms: Iterable[Atom], budget_bits: int) -> Allocation|None:
    atoms=tuple(atoms); best=None
    for selection in product(*(range(len(b)) for b in blocks)):
        c=evaluate_selection(blocks,atoms,selection)
        if c.total_bits>budget_bits: continue
        if best is None or (c.error,c.total_bits,c.option_ids)<(best.error,best.total_bits,best.option_ids): best=c
    return best

def coalition_rate_gain(fixed_closure_bits: int, private_savings: Sequence[int]) -> int:
    return sum(private_savings)-fixed_closure_bits
