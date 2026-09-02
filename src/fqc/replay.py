"""Deterministic numeric replay witnesses for real-model extraction.

Replay compares original-module outputs with outputs reconstructed from extracted
primitives under a predeclared tolerance contract. It is an extraction-correctness
witness, not a compression-quality metric.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .manifest_builder import sha256_json


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    inputs: Mapping[str, Any]
    reference_output: Any
    extracted_output: Any


@dataclass(frozen=True)
class ReplayCaseResult:
    case_id: str
    passed: bool
    reason: str | None
    input_hashes: Mapping[str, str]
    reference_hash: str
    extracted_hash: str
    max_abs_error: float
    max_tolerance_ratio: float


@dataclass(frozen=True)
class ReplayWitness:
    contract_hash: str
    cases: tuple[ReplayCaseResult, ...]
    passed: bool


def _array_hash(value: Any) -> str:
    a=np.asarray(value)
    c=np.ascontiguousarray(a)
    h=hashlib.sha256()
    h.update(str(c.dtype).encode('utf-8'))
    h.update(repr(tuple(int(d) for d in c.shape)).encode('ascii'))
    h.update(c.tobytes(order='C'))
    return 'sha256:'+h.hexdigest()


def _tolerances(contract: Mapping[str, Any]) -> tuple[float,float]:
    atol=contract.get('atol'); rtol=contract.get('rtol')
    if not isinstance(atol,(int,float)) or not isinstance(rtol,(int,float)):
        raise ValueError('replay contract requires numeric atol and rtol')
    atol=float(atol); rtol=float(rtol)
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol<0 or rtol<0:
        raise ValueError('atol and rtol must be finite and nonnegative')
    return atol,rtol


def compare_replay_case(case: ReplayCase, contract: Mapping[str, Any]) -> ReplayCaseResult:
    atol,rtol=_tolerances(contract)
    ref=np.asarray(case.reference_output)
    got=np.asarray(case.extracted_output)
    input_hashes={k:_array_hash(v) for k,v in sorted(case.inputs.items())}
    ref_hash=_array_hash(ref); got_hash=_array_hash(got)
    if ref.shape!=got.shape:
        return ReplayCaseResult(case.case_id,False,'shape mismatch',input_hashes,ref_hash,got_hash,float('inf'),float('inf'))
    try:
        r=ref.astype(np.float64,copy=False); g=got.astype(np.float64,copy=False)
    except (TypeError,ValueError):
        return ReplayCaseResult(case.case_id,False,'non-numeric output',input_hashes,ref_hash,got_hash,float('inf'),float('inf'))
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(g)):
        return ReplayCaseResult(case.case_id,False,'non-finite output',input_hashes,ref_hash,got_hash,float('inf'),float('inf'))
    diff=np.abs(g-r)
    allowed=atol+rtol*np.abs(r)
    passed=bool(np.all(diff<=allowed))
    max_abs=float(diff.max()) if diff.size else 0.0
    if diff.size:
        ratio=np.divide(diff,allowed,out=np.where(diff==0,0.0,np.inf),where=allowed>0)
        max_ratio=float(np.max(ratio))
    else:
        max_ratio=0.0
    return ReplayCaseResult(case.case_id,passed,None if passed else 'tolerance exceeded',input_hashes,ref_hash,got_hash,max_abs,max_ratio)


def build_replay_witness(cases: Sequence[ReplayCase], contract: Mapping[str, Any]) -> ReplayWitness:
    _tolerances(contract)
    ids=[c.case_id for c in cases]
    if any(not x for x in ids) or len(set(ids))!=len(ids):
        raise ValueError('replay case IDs must be unique and nonempty')
    results=tuple(compare_replay_case(c,contract) for c in sorted(cases,key=lambda x:x.case_id))
    return ReplayWitness(sha256_json(contract),results,all(r.passed for r in results))
