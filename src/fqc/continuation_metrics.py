"""Dimensionless metrics for separating local perturbation from downstream amplification."""
from __future__ import annotations

import numpy as np


def relative_rms_delta(reference, perturbed, mask=None) -> float:
    r=np.asarray(reference,dtype=np.float64)
    p=np.asarray(perturbed,dtype=np.float64)
    if r.shape!=p.shape:
        raise ValueError('reference and perturbed must have the same shape')
    if mask is not None:
        m=np.asarray(mask,dtype=bool)
        if m.shape!=r.shape[:-1]:
            raise ValueError('mask must match all non-feature dimensions')
        r=r[m]; p=p[m]
    if r.size==0:
        raise ValueError('no values selected')
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(p)):
        raise ValueError('inputs must be finite')
    den=float(np.sqrt(np.mean(r*r)))
    num=float(np.sqrt(np.mean((p-r)*(p-r))))
    return num/max(den,1e-30)


def safe_ratio(numerator: float, denominator: float) -> float:
    n=float(numerator); d=float(denominator)
    if not np.isfinite(n) or not np.isfinite(d) or n<0 or d<0:
        raise ValueError('ratio inputs must be finite and nonnegative')
    return n/max(d,1e-30)


def pearson_correlation(x, y) -> float:
    a=np.asarray(x,dtype=np.float64).reshape(-1)
    b=np.asarray(y,dtype=np.float64).reshape(-1)
    if a.shape!=b.shape or a.size<2:
        raise ValueError('same-length vectors with at least two values required')
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('values must be finite')
    aa=a-a.mean(); bb=b-b.mean()
    den=float(np.linalg.norm(aa)*np.linalg.norm(bb))
    return float(aa@bb/den) if den>0 else 0.0
