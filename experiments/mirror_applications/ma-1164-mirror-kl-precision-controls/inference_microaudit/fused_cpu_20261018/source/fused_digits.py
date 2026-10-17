"""CPU inference optimizer for the existing learned MA-1164 digit classifier.

Unlike a separate Hadamard operator, this module reproduces the existing
1210-coordinate Householder+Givens+permutation+tanh Mirror weight compiler.

Only the 1210-wide temporary working state z is allocated for the generated
virtual weight vector, and consumed immediately in the subsequent two dense
MLP stages. There is no persistent virtual NN matrix and no trained new weights.
The generator executes the same K parallel branches (fused sequential CPU here)
followed by R dependent iterations. CPU FP64. Uses compile once per process.
"""
import math
import numpy as np
from numba import njit,prange

N = 1210
GAMMA = 0.16


def make_program(K, R):
    if K < 1 or R < 1:
        raise ValueError('K and R must be >=1')
    ix = np.arange(N,dtype=np.float64)
    house = np.empty((R,K,N),dtype=np.float64)
    biases = np.empty_like(house)
    angles = np.empty((R,K),dtype=np.float64)
    shifts = np.empty((R,K),dtype=np.int64)
    for r in range(R):
        for k in range(K):
            a = np.sin((.031+.003*r)*ix+.19*k) + np.cos((.047+.002*k)*ix+.14*r)
            house[r,k] = a/np.linalg.norm(a)
            biases[r,k]=.17*np.sin(.013*ix+1.07*r+.37*k)
            angles[r,k]=.31+.19*r+.29*k
            shifts[r,k]=17+11*r+29*k
    seed_house=np.empty((4,N),dtype=np.float64)
    seed_angles=np.empty(4,dtype=np.float64)
    seed_shifts=np.empty(4,dtype=np.int64)
    for t in range(4):
        r=11+t; k=(t+1)%3
        a=np.sin((.031+.003*r)*ix+.19*k) + np.cos((.047+.002*k)*ix+.14*r)
        seed_house[t]=a/np.linalg.norm(a)
        seed_angles[t]=.31+.19*r+.29*k
        seed_shifts[t]=17+11*r+29*k
    return house,biases,angles,shifts,seed_house,seed_angles,seed_shifts


@njit(cache=True)
def _mirror(z, u, angle, shift, inverse):
    n=z.size
    c,s=np.cos(angle),np.sin(angle)
    v=np.empty(n,dtype=np.float64)
    out=np.empty(n,dtype=np.float64)
    if not inverse:
        for j in range(0,n,2):
            a=z[(j-shift)%n];b=z[(j+1-shift)%n]
            v[j] = c*a-s*b
            v[j+1] = s*a+c*b
        dot=0.0
        for j in range(n):dot+=u[j]*v[j]
        for j in range(n):out[j]=v[j]-2.0*u[j]*dot
    else:
        dot=0.0
        for j in range(n):dot+=u[j]*z[j]
        for j in range(n):v[j]=z[j]-2.0*u[j]*dot
        for j in range(0,n,2):
            a=v[j];b=v[j+1]
            out[(j-shift)%n]=c*a+s*b
            out[(j+1-shift)%n]=-s*a+c*b
    return out


@njit(cache=True)
def _decode(coords,positions,house,biases,angles,shifts,seed_house,seed_angles,seed_shifts):
    z=np.zeros(N,dtype=np.float64)
    for i in range(coords.size):z[positions[i]]=coords[i]
    for t in range(4):
        z=_mirror(z,seed_house[t],seed_angles[t],seed_shifts[t],False)
    R,K=angles.shape
    for r in range(R):
        out=np.zeros(N,dtype=np.float64)
        for k in range(K):
            q=_mirror(z,house[r,k],angles[r,k],shifts[r,k],False)
            for j in range(N):q[j]=np.tanh(q[j]+biases[r,k,j])
            view=_mirror(q,house[r,k],angles[r,k],shifts[r,k],True)
            for j in range(N):out[j]+=view[j]
        for j in range(N):z[j]+=GAMMA*out[j]/K
    return z


@njit(cache=True,parallel=True)
def _logits_parallel(weights,x):
    B=x.shape[0]
    out=np.empty((B,10),dtype=np.float64)
    for b in prange(B):
        h=np.empty(16,dtype=np.float64)
        for j in range(16):
            s=weights[1024+j]
            for i in range(64): s+=weights[j*64+i]*x[b,i]
            h[j]=np.tanh(s)
        for j in range(10):
            s=weights[1200+j]
            for i in range(16):s+=weights[1040+j*16+i]*h[i]
            out[b,j]=s
    return out


@njit(cache=True)
def _logits_serial(weights,x):
    B=x.shape[0]
    out=np.empty((B,10),dtype=np.float64)
    for b in range(B):
        h=np.empty(16,dtype=np.float64)
        for j in range(16):
            s=weights[1024+j]
            for i in range(64): s+=weights[j*64+i]*x[b,i]
            h[j]=np.tanh(s)
        for j in range(10):
            s=weights[1200+j]
            for i in range(16):s+=weights[1040+j*16+i]*h[i]
            out[b,j]=s
    return out


def decode(coords,indices,program):
    coords=np.ascontiguousarray(coords,dtype=np.float64)
    indices=np.ascontiguousarray(indices,dtype=np.int64)
    if indices.ndim!=1 or coords.ndim!=1 or indices.size!=coords.size or indices.size<1:
        raise ValueError('shape mismatch')
    if np.any(indices<0) or np.any(indices>=N) or len(np.unique(indices))!=len(indices):
        raise ValueError('invalid indices')
    return _decode(coords,indices,*program)


def infer(coords,indices,program,x,parallel=False):
    # Generated virtual weights are transient O(1210) workspace, not persistent.
    w=decode(coords,indices,program)
    inputs=np.ascontiguousarray(x,dtype=np.float64)
    if inputs.ndim!=2 or inputs.shape[1]!=64:raise ValueError('expected x[B,64]')
    fn=_logits_parallel if parallel else _logits_serial
    return fn(w,inputs)
