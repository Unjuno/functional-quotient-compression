"""Certificate-aware structural diagnostics from the historical D58 line.

These functions measure candidate structure. They do not produce codec-bit
claims. Exact algebraic eligibility checks are kept separate from descriptive
statistics so a numerical diagnostic cannot silently inherit a theorem whose
hypotheses are false.
"""
from __future__ import annotations
from collections import Counter
from typing import Iterable, Sequence
import numpy as np


def _same_shape(mats: Sequence[np.ndarray]) -> tuple[int, ...]:
    if not mats:
        raise ValueError("at least one matrix is required")
    shapes={np.asarray(x).shape for x in mats}
    if len(shapes)!=1:
        raise ValueError("all matrices must have identical shape")
    return next(iter(shapes))


def operator_span_spectrum(mats: Sequence[np.ndarray], normalize: bool=False):
    """Exact best shared-linear-span residual curve by Eckart-Young."""
    _same_shape(mats)
    rows=[]
    for X in mats:
        Y=np.asarray(X,dtype=float)
        if normalize:
            n=np.linalg.norm(Y,'fro')
            if n>0: Y=Y/n
        rows.append(Y.reshape(-1))
    M=np.stack(rows)
    s=np.linalg.svd(M,compute_uv=False)
    total=float(np.sum(s*s))
    residual=np.array([float(np.sum(s[r:]**2)/total) if total>0 else 0.0 for r in range(len(s)+1)])
    return s,residual


def normalized_commutator(X, Y) -> float:
    X=np.asarray(X,dtype=float); Y=np.asarray(Y,dtype=float)
    if X.ndim!=2 or Y.ndim!=2 or X.shape!=Y.shape or X.shape[0]!=X.shape[1]:
        raise ValueError("commutator requires same-shape square matrices")
    den=max(float(np.linalg.norm(X,'fro')*np.linalg.norm(Y,'fro')),1e-30)
    return float(np.linalg.norm(X@Y-Y@X,'fro')/den)


def symmetric_orthogonal_jd_eligible(mats: Sequence[np.ndarray], atol: float=1e-10) -> bool:
    """Whether the real-symmetric simultaneous-orthogonal-JD theorem applies."""
    try: shape=_same_shape(mats)
    except ValueError: return False
    if len(shape)!=2 or shape[0]!=shape[1]: return False
    return all(np.allclose(np.asarray(X,float),np.asarray(X,float).T,atol=atol,rtol=0.0) for X in mats)


def fixed_support_residual(X, Q, support: Iterable[int], *, require_orthogonal: bool=True, atol: float=1e-10):
    """Residual outside a declared coordinate block after basis transform."""
    X=np.asarray(X,dtype=float); Q=np.asarray(Q,dtype=float)
    if X.ndim!=2 or X.shape[0]!=X.shape[1] or Q.shape!=X.shape:
        raise ValueError("X and Q must be same-shape square matrices")
    if require_orthogonal and not np.allclose(Q.T@Q,np.eye(Q.shape[0]),atol=atol,rtol=0.0):
        raise ValueError("Q must be orthogonal for coordinate-support interpretation")
    S=tuple(sorted(set(int(i) for i in support)))
    if any(i<0 or i>=Q.shape[0] for i in S): raise ValueError("support index out of range")
    mask=np.zeros(Q.shape[0]); mask[list(S)]=1.0
    P=np.diag(mask); Y=Q.T@X@Q; R=Y-P@Y@P
    sq=float(np.linalg.norm(R,'fro')**2); den=float(np.linalg.norm(Y,'fro')**2)
    return sq, (sq/den if den>0 else 0.0)


def gqa_consumer_multiplicity(q_to_kv_map: Sequence[int], kv_heads: int) -> tuple[int,...]:
    if kv_heads<=0: raise ValueError("kv_heads must be positive")
    if any((not isinstance(x,int)) or isinstance(x,bool) or x<0 or x>=kv_heads for x in q_to_kv_map):
        raise ValueError("invalid q_to_kv_map")
    c=Counter(q_to_kv_map)
    return tuple(c.get(i,0) for i in range(kv_heads))


def channel_outer_descriptor(up_affine, down_row):
    """Exact descriptor invariant under z->c z, d->d/c for nonzero c."""
    z=np.asarray(up_affine,dtype=float).reshape(-1)
    d=np.asarray(down_row,dtype=float).reshape(-1)
    return np.outer(z,d)
