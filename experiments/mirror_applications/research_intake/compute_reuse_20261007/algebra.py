"""Reference algebra for shared Mirror computation; not an optimized kernel.

Row-vector layout: v[B,H], down[H,O], signs[E,H], coefficients[B,E].
Dropout must be disabled (or explicitly coupled) for fusion equivalence.
"""
from typing import Callable
import math
import torch
from torch.nn import functional as F

Tensor = torch.Tensor
Activation = Callable[[Tensor], Tensor]

def _validate(v: Tensor, down: Tensor, signs: Tensor, coefficients: Tensor) -> None:
    if any(t.ndim != 2 for t in (v,down,signs,coefficients)):
        raise ValueError('Expected rank-2 tensors')
    if down.shape[0] != v.shape[1] or signs.shape[1] != v.shape[1]:
        raise ValueError('Hidden dimensions differ')
    if coefficients.shape != (v.shape[0],signs.shape[0]):
        raise ValueError('Coefficient shape must be [batch,views]')
    if not bool(torch.all((signs==1)|(signs==-1))):
        raise ValueError('Sign codes must be +/-1; do not use continuous codes here')

def sign_explicit(v: Tensor, down: Tensor, signs: Tensor, coefficients: Tensor,
                  activation: Activation = F.gelu) -> Tensor:
    _validate(v,down,signs,coefficients)
    branches=signs[None,:,:]*activation(v[:,None,:]*signs[None,:,:])
    outputs=branches@down
    return (outputs*coefficients[:,:,None]).sum(1)

def sign_fused(v: Tensor, down: Tensor, signs: Tensor, coefficients: Tensor,
               activation: Activation = F.gelu) -> Tensor:
    """Valid when activation(x)-activation(-x)=x. Router weights may be signed.

The function checks codes, not the activation identity. Callers must verify it.
No bias here: v may already include shared input bias; add shared output bias
multiplied by coefficients.sum(-1,keepdim=True) when using unnormalized weights.
"""
    _validate(v,down,signs,coefficients)
    gain=coefficients@signs
    mass=coefficients.sum(-1,keepdim=True)
    hidden=gain*activation(v)+.5*(mass-gain)*v
    return hidden@down

def mirror(v: Tensor, transform: Tensor, activation: Activation = F.gelu) -> Tensor:
    """Column-space Q^{-1} activation(Q v), evaluated with row-stored v."""
    return torch.linalg.solve(transform,activation(v@transform.T).T).T

def attention(q: Tensor,k: Tensor,v: Tensor) -> Tensor:
    if q.shape[-1]!=k.shape[-1] or k.shape[-2]!=v.shape[-2]:
        raise ValueError('Incompatible attention shapes')
    return torch.softmax(q@k.T/math.sqrt(q.shape[-1]),dim=-1)@v

def lazy_attention(q: Tensor,k: Tensor,v: Tensor,a: Tensor,b: Tensor) -> Tensor:
    """Exact for K_m=K A, V_m=V B with A and B constant over source tokens.

Assumes A square, so scaling dimension is unchanged. All cached hidden-state
provenance must already satisfy the right-transform relation.
"""
    if a.shape!=(k.shape[-1],k.shape[-1]):
        raise ValueError('This reference requires square A')
    return attention(q@a.T,k,v)@b

def partition_attention(q: Tensor,k1: Tensor,v1: Tensor,k2: Tensor,v2: Tensor) -> Tensor:
    """Exact prefix/suffix combination via log-partition weights, not averaging."""
    s1=q@k1.T/math.sqrt(q.shape[-1]);s2=q@k2.T/math.sqrt(q.shape[-1])
    l1=s1.logsumexp(-1,keepdim=True);l2=s2.logsumexp(-1,keepdim=True)
    normalizer=torch.logaddexp(l1,l2)
    return torch.exp(l1-normalizer)*(s1.softmax(-1)@v1)+torch.exp(l2-normalizer)*(s2.softmax(-1)@v2)

def mirror_mixture_fused(v: Tensor, down: Tensor, transforms: Tensor,
                         coefficients: Tensor, activation: Activation = F.gelu) -> Tensor:
    """Shared output projection once, for arbitrary invertible hidden Views.

Each nonlinear View still executes: this is not averaging transform codes.
The supplied v is a precomputed shared input projection for the same tokens.
"""
    if transforms.ndim!=3 or coefficients.shape!=(v.shape[0],transforms.shape[0]):
        raise ValueError('Invalid View/coefficients shape')
    hidden=torch.stack([mirror(v,t,activation) for t in transforms],dim=1)
    return (hidden*coefficients[:,:,None]).sum(1)@down
