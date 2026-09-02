"""Intervention builders for task-conditioned real-model audits.

These routines construct explicit weight perturbations from structural candidates.
They do not decide whether an intervention is codec-positive or task-safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class FamilyRankApproximation:
    reconstructed: tuple[np.ndarray, ...]
    rank: int
    member_count: int
    normalized_family_residual: float
    per_member_relative_frobenius_change: tuple[float, ...]


def normalized_family_rank_approximation(mats: Sequence[np.ndarray], rank: int) -> FamilyRankApproximation:
    """Best rank-r approximation of Frobenius-normalized family members.

    Each member is normalized before the family SVD and rescaled to its original
    Frobenius norm after reconstruction. This matches the T002 structural-family
    metric and makes the intervention definition explicit.
    """
    xs=[np.asarray(x,dtype=np.float64) for x in mats]
    if not xs:
        raise ValueError('at least one matrix is required')
    shape=xs[0].shape
    if len(shape)!=2 or any(x.shape!=shape for x in xs):
        raise ValueError('all family members must be same-shape matrices')
    m=len(xs)
    if not isinstance(rank,int) or isinstance(rank,bool) or rank<=0 or rank>m:
        raise ValueError('rank must be an integer in [1, member_count]')
    norms=np.array([np.linalg.norm(x,'fro') for x in xs],dtype=np.float64)
    if np.any(norms<=0) or not np.all(np.isfinite(norms)):
        raise ValueError('family members must have finite nonzero Frobenius norm')
    M=np.stack([(x/n).reshape(-1) for x,n in zip(xs,norms)])
    U,s,Vh=np.linalg.svd(M,full_matrices=False)
    Mr=(U[:,:rank]*s[:rank])@Vh[:rank,:]
    recon=tuple((Mr[i].reshape(shape)*norms[i]) for i in range(m))
    total=float(np.sum(s*s))
    residual=float(np.sum(s[rank:]**2)/total) if total>0 else 0.0
    changes=[]
    for x,y,n in zip(xs,recon,norms):
        changes.append(float(np.linalg.norm(y-x,'fro')/n))
    return FamilyRankApproximation(recon,rank,m,residual,tuple(changes))


def bottom_fraction_indices(scores, fraction: float) -> np.ndarray:
    """Indices of the lowest-scoring fraction with deterministic stable ties."""
    x=np.asarray(scores,dtype=np.float64).reshape(-1)
    if x.size==0 or not np.all(np.isfinite(x)) or np.any(x<0):
        raise ValueError('scores must be nonempty, finite, and nonnegative')
    if not isinstance(fraction,(int,float)) or not (0<float(fraction)<1):
        raise ValueError('fraction must lie strictly between 0 and 1')
    k=max(1,int(np.floor(float(fraction)*x.size)))
    order=np.argsort(x,kind='stable')
    return order[:k]


def swiglu_descriptor_scores(up_weight, down_weight) -> np.ndarray:
    """Scale-invariant linear-branch channel descriptor norm used by T002."""
    U=np.asarray(up_weight,dtype=np.float64)
    D=np.asarray(down_weight,dtype=np.float64)
    if U.ndim!=2 or D.ndim!=2 or D.shape!=(U.shape[1],U.shape[0]):
        raise ValueError('expected up [intermediate,d_model], down [d_model,intermediate]')
    return np.linalg.norm(U,axis=1)*np.linalg.norm(D,axis=0)


def removed_descriptor_energy_fraction(scores, indices) -> float:
    x=np.asarray(scores,dtype=np.float64).reshape(-1)
    idx=np.asarray(indices,dtype=int).reshape(-1)
    if np.any(idx<0) or np.any(idx>=x.size):
        raise ValueError('index out of range')
    e=x*x; total=float(e.sum())
    return float(e[idx].sum()/total) if total>0 else 0.0
