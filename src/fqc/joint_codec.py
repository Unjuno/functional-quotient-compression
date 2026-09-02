"""Integrated exact joint-codec state for small certification problems.

This is a canonical integration layer, not a recovered historical generator.
It combines:
- E7 global layout / local precision complementarity;
- D63 conditional shared paid atoms and private/shared choices;
- E2-style nonseparable task error through an exact joint-error callback.

Use only for small exact problems; larger instances require BnB/DP relaxations.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Callable, Mapping, Sequence, Any, Iterable
from .decoder_dag import Atom, compile_decoder_dag

@dataclass(frozen=True)
class JointOption:
    option_id: str
    private_bits: int
    requires: tuple[str,...]=()
    mode: str='private'
    precision: str|int|None=None
    metadata: tuple[tuple[str,Any],...]=()

@dataclass(frozen=True)
class LayoutSpec:
    layout_id: str
    block_options: tuple[tuple[JointOption,...],...]
    private_bits: int=0
    requires: tuple[str,...]=()

@dataclass(frozen=True)
class JointCodecResult:
    layout_id: str
    option_ids: tuple[str,...]
    private_bits: int
    shared_paid_bits: int
    total_bits: int
    error: float
    active_atoms: frozenset[str]


def evaluate_joint_candidate(layout: LayoutSpec, atoms: Iterable[Atom], selection: Sequence[int], joint_error: Callable[[str,tuple[JointOption,...]],float]) -> JointCodecResult:
    if len(selection)!=len(layout.block_options):
        raise ValueError('selection length must match block count')
    chosen=tuple(layout.block_options[i][j] for i,j in enumerate(selection))
    if layout.private_bits<0 or any(o.private_bits<0 for o in chosen):
        raise ValueError('private bits must be non-negative')
    required=set(layout.requires)
    for o in chosen: required.update(o.requires)
    compiled=compile_decoder_dag(tuple(atoms),sorted(required))
    if not compiled.valid:
        raise ValueError('invalid decoder DAG: '+'; '.join(compiled.errors))
    private=layout.private_bits+sum(o.private_bits for o in chosen)
    error=float(joint_error(layout.layout_id,chosen))
    return JointCodecResult(layout.layout_id,tuple(o.option_id for o in chosen),private,compiled.total_paid_bits,private+compiled.total_paid_bits,error,compiled.reachable_atoms)


def enumerate_joint_codec(layouts: Sequence[LayoutSpec], atoms: Iterable[Atom], joint_error: Callable[[str,tuple[JointOption,...]],float]) -> tuple[JointCodecResult,...]:
    atoms=tuple(atoms); out=[]
    for layout in layouts:
        for selection in product(*(range(len(opts)) for opts in layout.block_options)):
            out.append(evaluate_joint_candidate(layout,atoms,selection,joint_error))
    return tuple(out)


def exact_joint_codec_allocate(layouts: Sequence[LayoutSpec], atoms: Iterable[Atom], budget_bits: int, joint_error: Callable[[str,tuple[JointOption,...]],float]) -> JointCodecResult|None:
    best=None
    for c in enumerate_joint_codec(layouts,atoms,joint_error):
        if c.total_bits>budget_bits: continue
        if best is None or (c.error,c.total_bits,c.layout_id,c.option_ids)<(best.error,best.total_bits,best.layout_id,best.option_ids):
            best=c
    return best


def coordinate_best_from_incumbent(layouts: Sequence[LayoutSpec], atoms: Iterable[Atom], budget_bits: int, joint_error: Callable[[str,tuple[JointOption,...]],float], incumbent_layout: str, incumbent_option_ids: Sequence[str]) -> Mapping[str,JointCodecResult|None]:
    """Best one-axis layout-only and local-option-only moves around an incumbent.

    This is a diagnostic witness for complementarity, not a general optimizer.
    """
    by_layout={l.layout_id:l for l in layouts}
    base_layout=by_layout[incumbent_layout]
    if len(incumbent_option_ids)!=len(base_layout.block_options): raise ValueError('incumbent length mismatch')
    def selection_for(layout, ids):
        sel=[]
        for opts,oid in zip(layout.block_options,ids):
            matches=[i for i,o in enumerate(opts) if o.option_id==oid]
            if not matches: return None
            sel.append(matches[0])
        return tuple(sel)
    layout_candidates=[]
    for l in layouts:
        s=selection_for(l,incumbent_option_ids)
        if s is None: continue
        c=evaluate_joint_candidate(l,atoms,s,joint_error)
        if c.total_bits<=budget_bits: layout_candidates.append(c)
    local_candidates=[]
    for selection in product(*(range(len(opts)) for opts in base_layout.block_options)):
        c=evaluate_joint_candidate(base_layout,atoms,selection,joint_error)
        if c.total_bits<=budget_bits: local_candidates.append(c)
    key=lambda c:(c.error,c.total_bits,c.layout_id,c.option_ids)
    return {'layout_only':min(layout_candidates,key=key) if layout_candidates else None,
            'local_only':min(local_candidates,key=key) if local_candidates else None}
