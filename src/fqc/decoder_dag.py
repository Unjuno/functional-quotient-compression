"""Paid-atom decoder DAG accounting.

Canonicalized from the D56 real-pilot contract and the D59 diagnostic-to-codec
compiler. The key invariant is union accounting over the dependency closure:
a shared paid atom is charged exactly once.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Mapping

VALID_CLASSES = frozenset({"PAID", "EXTERNAL_FIXED", "DERIVED"})

@dataclass(frozen=True)
class Atom:
    atom_id: str
    atom_class: str
    bits: int
    dependencies: tuple[str, ...] = ()

@dataclass(frozen=True)
class CompileResult:
    errors: tuple[str, ...]
    reachable_atoms: frozenset[str]
    total_paid_bits: int
    @property
    def valid(self) -> bool:
        return not self.errors

def _cycle(graph: Mapping[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    visiting=set(); done=set(); stack=[]
    def dfs(node):
        if node in done: return None
        if node in visiting:
            i=stack.index(node); return tuple(stack[i:]+[node])
        visiting.add(node); stack.append(node)
        for dep in graph.get(node,()):
            found=dfs(dep)
            if found is not None: return found
        stack.pop(); visiting.remove(node); done.add(node)
        return None
    for node in graph:
        found=dfs(node)
        if found is not None: return found
    return None

def compile_decoder_dag(atoms: Iterable[Atom], required_outputs: Iterable[str]) -> CompileResult:
    errors=[]; by_id={}
    for atom in atoms:
        if not atom.atom_id:
            errors.append("empty atom_id"); continue
        if atom.atom_id in by_id:
            errors.append(f"duplicate atom_id: {atom.atom_id}"); continue
        if atom.atom_class not in VALID_CLASSES:
            errors.append(f"{atom.atom_id}: invalid atom class {atom.atom_class}")
        if not isinstance(atom.bits,int) or isinstance(atom.bits,bool) or atom.bits<0:
            errors.append(f"{atom.atom_id}: bits must be a non-negative integer")
        if atom.atom_class!="PAID" and atom.bits!=0:
            errors.append(f"{atom.atom_id}: non-PAID atom must have zero bits")
        if atom.atom_class=="EXTERNAL_FIXED" and atom.dependencies:
            errors.append(f"{atom.atom_id}: EXTERNAL_FIXED atom must not depend on candidate atoms")
        by_id[atom.atom_id]=atom
    graph={}
    for atom_id,atom in by_id.items():
        deps=[]
        for dep in atom.dependencies:
            if dep not in by_id: errors.append(f"{atom_id}: missing dependency {dep}")
            else: deps.append(dep)
        graph[atom_id]=tuple(deps)
    found=_cycle(graph)
    if found is not None: errors.append("dependency cycle: "+" -> ".join(found))
    outputs=tuple(required_outputs)
    for output in outputs:
        if output not in by_id: errors.append(f"missing required output: {output}")
    reachable=set(); work=[o for o in outputs if o in by_id]
    while work:
        node=work.pop()
        if node in reachable: continue
        reachable.add(node); work.extend(graph.get(node,()))
    bits=sum(by_id[n].bits for n in reachable if by_id[n].atom_class=="PAID")
    return CompileResult(tuple(errors),frozenset(reachable),bits)

def union_paid_bits(atoms: Iterable[Atom], required_sets: Iterable[Iterable[str]]) -> CompileResult:
    roots=set()
    for required in required_sets: roots.update(required)
    return compile_decoder_dag(atoms,sorted(roots))
