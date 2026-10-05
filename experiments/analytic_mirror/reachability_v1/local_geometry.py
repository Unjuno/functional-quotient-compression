# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Small exact-linear-model diagnostics, NOT a Transformer optimizer.

All inputs are finite float64 arrays. Metrics must be explicitly specified and
positive definite. Singular directions are checked for feasibility before costs
are computed; a pseudoinverse must not turn an unreachable target into a free one.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cholesky, eigh, expm, solve_triangular
from scipy.special import ndtr


def _array(x, ndim: int) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64)
    if out.ndim != ndim or not np.isfinite(out).all() or min(out.shape, default=0) < 1:
        raise ValueError('Expected a nonempty finite array with the specified dimension')
    return out


def _spd(x, size: int) -> np.ndarray:
    out = _array(x, 2)
    if out.shape != (size, size) or not np.allclose(out, out.T, atol=1e-12, rtol=1e-12):
        raise ValueError('Metric must be symmetric and have the required shape')
    try:
        cholesky(out, lower=True)
    except np.linalg.LinAlgError as error:
        raise ValueError('Metric must be positive definite') from error
    return out


def _gelu(v: np.ndarray) -> np.ndarray:
    return v * ndtr(v)


def _gelu_prime(v: np.ndarray) -> np.ndarray:
    return ndtr(v) + v * np.exp(-0.5*v*v)/np.sqrt(2*np.pi)


def mirror(v, a, eps: float = 1.0) -> np.ndarray:
    v, a = _array(v, 1), _array(a, 2)
    if a.shape != (v.size, v.size) or not np.isfinite(eps):
        raise ValueError('Incompatible generator or nonfinite amplitude')
    return expm(-eps*a) @ _gelu(expm(eps*a) @ v)


def mirror_tangent(v, a) -> np.ndarray:
    v, a = _array(v, 1), _array(a, 2)
    if a.shape != (v.size, v.size):
        raise ValueError('Incompatible generator shape')
    return _gelu_prime(v)*(a@v) - a@_gelu(v)


def generator_sensitivity(v, downstream) -> np.ndarray:
    v, downstream = _array(v, 1), _array(downstream, 1)
    if downstream.shape != v.shape:
        raise ValueError('Incompatible downstream gradient')
    return np.outer(downstream*_gelu_prime(v),v) - np.outer(downstream,_gelu(v))


def weighted_reachability(j, metric, target, rtol: float = 1e-10) -> dict:
    """Minimize 0.5 step.T metric step subject to j step = target.

    The numerical column space uses a relative SVD cutoff rtol. Feasibility is
    tested separately. An unreachable target gets infinite cost and no step.
    """
    j, target = _array(j, 2), _array(target, 1)
    if target.size != j.shape[0] or not 0 < rtol < 1:
        raise ValueError('Incompatible target or invalid rank tolerance')
    metric = _spd(metric, j.shape[1])
    lower = cholesky(metric, lower=True)
    whitened = solve_triangular(lower,j.T,lower=True).T
    u, singular, vt = np.linalg.svd(whitened,full_matrices=False)
    rank = int(np.count_nonzero(singular > rtol*singular[0])) if singular[0] > 0 else 0
    basis = u[:,:rank]
    residual = target-basis@(basis.T@target)
    residual_norm = float(np.linalg.norm(residual))
    reachable = residual_norm <= rtol*max(1.0,float(np.linalg.norm(target)))
    result = {'reachable':reachable,'rank':rank,'residual_norm':residual_norm,
              'cost':float('inf'),'step':None,'gram':whitened@whitened.T}
    if reachable:
        z = vt[:rank].T @ ((basis.T@target)/singular[:rank])
        step = solve_triangular(lower.T,z,lower=False)
        result.update(cost=float(z@z/2),step=step)
    return result


def efficiency_directions(j, metric, b, mirror_metric):
    """Return eigenpairs of base_cost_matrix v = value mirror_metric v.

    High values mean expensive baseline emulation per unit Mirror metric, NOT
    high task utility. Caller must check signed task benefit and retention.
    All columns of b must be in the baseline's numerical reachable subspace.
    """
    j, b = _array(j,2), _array(b,2)
    if b.shape[0] != j.shape[0]:
        raise ValueError('Observable dimensions do not match')
    metric = _spd(metric,j.shape[1]); mirror_metric = _spd(mirror_metric,b.shape[1])
    solutions = [weighted_reachability(j,metric,b[:,i]) for i in range(b.shape[1])]
    if not all(s['reachable'] for s in solutions):
        raise ValueError('Restrict to reachable targets; analyze residuals separately')
    steps = np.column_stack([s['step'] for s in solutions])
    base_cost_matrix = steps.T@metric@steps
    base_cost_matrix = (base_cost_matrix+base_cost_matrix.T)/2
    values, vectors = eigh(base_cost_matrix,mirror_metric)
    return values[::-1], vectors[:,::-1], base_cost_matrix


def regular_simplex(rank: int) -> np.ndarray:
    """Unit-radius, zero-mean r+1-point simplex with covariance I/r."""
    if isinstance(rank,bool) or not isinstance(rank,(int,np.integer)) or rank < 1:
        raise ValueError('Rank must be a positive integer')
    count = rank+1
    projector = np.eye(count)-np.ones((count,count))/count
    basis,_ = np.linalg.qr(projector[:,:rank],mode='reduced')
    return np.sqrt(count/rank)*basis


def frozen_gradient_residual(j, metric, target, lr: float, steps: int) -> np.ndarray:
    """Exact residual for fixed-J, fixed-metric preconditioned least-squares GD.

    This is not an exact prediction for AdamW or a nonlinear neural network.
    """
    j, target = _array(j,2), _array(target,1)
    metric = _spd(metric,j.shape[1])
    if target.size != j.shape[0] or isinstance(steps,bool) or not isinstance(steps,(int,np.integer)) or steps < 0:
        raise ValueError('Incompatible target or invalid step count')
    gram = j@np.linalg.solve(metric,j.T)
    largest = float(np.linalg.eigvalsh(gram)[-1])
    if not np.isfinite(lr) or lr <= 0 or (largest > 0 and lr >= 2/largest):
        raise ValueError('Learning rate must lie inside the stated stable interval')
    return np.linalg.matrix_power(np.eye(j.shape[0])-lr*gram,steps)@target
