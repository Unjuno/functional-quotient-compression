"""Functional perturbation certificates from the historical D61 line.

Bounds are deterministic on a declared input X. A loose upper bound is not
failure evidence; only a valid quality contract can turn a sufficient bound
into an admission/rejection certificate.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np


def softmax_rows(L):
    L=np.asarray(L,dtype=float)
    Z=L-np.max(L,axis=1,keepdims=True)
    E=np.exp(Z)
    return E/np.sum(E,axis=1,keepdims=True)


def bilinear_logit_bound(X, dP, alpha: float) -> float:
    X=np.asarray(X,dtype=float); dP=np.asarray(dP,dtype=float)
    return float(abs(alpha)*(np.linalg.norm(X,2)**2)*np.linalg.norm(dP,'fro'))


def softmax_half_lipschitz_bound(dL) -> float:
    return float(0.5*np.linalg.norm(np.asarray(dL,float),'fro'))


def attention_output_bound(X, R, dR, L, Lp) -> float:
    X=np.asarray(X,float); R=np.asarray(R,float); dR=np.asarray(dR,float)
    L=np.asarray(L,float); Lp=np.asarray(Lp,float)
    Ap=softmax_rows(Lp); dL=Lp-L
    return float(0.5*np.linalg.norm(dL,'fro')*np.linalg.norm(X@R,2) + np.linalg.norm(Ap,2)*np.linalg.norm(X,2)*np.linalg.norm(dR,'fro'))


def attention_output_bound_coarse(X, R, dR, L, Lp) -> float:
    X=np.asarray(X,float); R=np.asarray(R,float); dR=np.asarray(dR,float)
    L=np.asarray(L,float); Lp=np.asarray(Lp,float)
    T=Lp.shape[0]; dL=Lp-L
    return float(0.5*np.linalg.norm(dL,'fro')*np.linalg.norm(X@R,2) + np.sqrt(T)*np.linalg.norm(X,2)*np.linalg.norm(dR,'fro'))


def telescoping_composition_bound(local_deltas: Sequence[float], continuation_lipschitz: Sequence[float]) -> float:
    """Sum delta_l prod_{j>l} K_j with one continuation K per stage."""
    if len(local_deltas)!=len(continuation_lipschitz):
        raise ValueError("local_deltas and continuation_lipschitz must have equal length")
    total=0.0
    for i,d in enumerate(local_deltas):
        prod=1.0
        for K in continuation_lipschitz[i+1:]: prod*=K
        total += d*prod
    return float(total)


def top1_margin(logits) -> float:
    z=np.asarray(logits,float).reshape(-1)
    if len(z)<2: raise ValueError("at least two logits are required")
    order=np.sort(z)
    return float(order[-1]-order[-2])


def top1_preservation_certificate(reference_logits, eps_inf: float) -> bool:
    if eps_inf<0: raise ValueError("eps_inf must be non-negative")
    return top1_margin(reference_logits) > 2*eps_inf


def cross_entropy(logits, target: int) -> float:
    z=np.asarray(logits,float).reshape(-1)
    if target<0 or target>=len(z): raise ValueError("target out of range")
    m=float(np.max(z))
    return float(-(z[target]-m)+np.log(np.sum(np.exp(z-m))))


def cross_entropy_change_bounds(delta_logits) -> tuple[float,float]:
    dz=np.asarray(delta_logits,float).reshape(-1)
    return float(2*np.linalg.norm(dz,np.inf)), float(np.sqrt(2)*np.linalg.norm(dz,2))
