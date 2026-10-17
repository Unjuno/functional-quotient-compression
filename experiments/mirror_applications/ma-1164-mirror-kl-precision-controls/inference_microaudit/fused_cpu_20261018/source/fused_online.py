"""Fused, matrix-free K-way Mirror recurrence for CPU FP64.

Implements the exact linear-branch semantics of the prior online_operator.py:
    y_{r,k} = D H diag(u) H D z_r
    z_{r+1} = z_r + gamma * tanh(mean_k(y_{r,k}))
Also supports the genuinely nonlinear Mirror sandwich:
    y_{r,k} = D H tanh(diag(u) H D z_r)
    z_{r+1} = z_r + gamma * mean_k(y_{r,k})

The signed Walsh-Hadamard transforms are applied by in-place butterflies.
No dense d-by-d virtual weight matrix or Hadamard matrix is instantiated.

Numba kernels are inference-only, without autograd integration. A separate
exact backward kernel would be needed for end-to-end training.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, inline='always')
def _fwht_inplace(v):
    n = len(v)
    step = 1
    while step < n:
        period = 2 * step
        for base in range(0, n, period):
            for j in range(step):
                a = v[base + j]
                b = v[base + j + step]
                v[base + j] = a + b
                v[base + j + step] = a - b
        step = period


@njit(cache=True, inline='always')
def _one_sequence(row, u, codes, gamma, nonlinear_inside):
    rounds, kcount, n = codes.shape
    z = row.copy()
    v = np.empty(n, dtype=np.float64)
    acc = np.empty(n, dtype=np.float64)
    invroot = 1.0 / np.sqrt(float(n))
    for r in range(rounds):
        for j in range(n):
            acc[j] = 0.0
        for k in range(kcount):
            for j in range(n):
                v[j] = z[j] * codes[r, k, j]
            _fwht_inplace(v)
            if nonlinear_inside:
                for j in range(n):
                    v[j] = np.tanh(u[j] * v[j] * invroot)
            else:
                for j in range(n):
                    v[j] *= u[j] * invroot
            _fwht_inplace(v)
            for j in range(n):
                acc[j] += codes[r, k, j] * v[j] * invroot
        if nonlinear_inside:
            for j in range(n):
                z[j] += gamma * acc[j] / kcount
        else:
            for j in range(n):
                z[j] += gamma * np.tanh(acc[j] / kcount)
    return z


@njit(cache=True)
def _forward_serial(x, u, codes, gamma, nonlinear_inside):
    result = np.empty_like(x)
    for i in range(x.shape[0]):
        result[i] = _one_sequence(x[i], u, codes, gamma, nonlinear_inside)
    return result


@njit(parallel=True, cache=True)
def _forward_parallel(x, u, codes, gamma, nonlinear_inside):
    result = np.empty_like(x)
    for i in prange(x.shape[0]):
        result[i] = _one_sequence(x[i], u, codes, gamma, nonlinear_inside)
    return result


def forward(x, u, sign_codes, gamma=0.2, nonlinear_inside=False, parallel=False):
    """Numpy float64 arrays, shapes x[B,d], u[d], codes[R,K,d] with +/-1."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    u = np.ascontiguousarray(u, dtype=np.float64)
    codes = np.ascontiguousarray(sign_codes, dtype=np.int8)
    if x.ndim != 2 or u.ndim != 1 or codes.ndim != 3:
        raise ValueError('expected x[B,d], u[d], sign_codes[R,K,d]')
    d = x.shape[-1]
    if d < 2 or d & (d - 1):
        raise ValueError('d must be a power of two >= 2')
    if u.shape != (d,) or codes.shape[2] != d or codes.shape[0] < 1 or codes.shape[1] < 1:
        raise ValueError('incompatible shapes or empty K/R')
    if not np.all(np.abs(codes) == 1):
        raise ValueError('sign codes must be +/-1')
    if not np.isfinite(gamma):
        raise ValueError('gamma must be finite')
    fn = _forward_parallel if parallel else _forward_serial
    return fn(x, u, codes, float(gamma), bool(nonlinear_inside))
