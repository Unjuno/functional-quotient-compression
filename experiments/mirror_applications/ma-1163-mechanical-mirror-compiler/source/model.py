"""No-label mechanical compilation of a trained 64->16->10 digit MLP into a nonlinear Mirror orbit.

One physical *vector* of m learned scalar coefficients (m variable), not one
learned scalar. Fixed orthogonal mirrored transforms with K parallel directions
are run for R rounds; no hidden large trainable seed/decoder matrices.
"""
from __future__ import annotations
import io
import json
import math
import torch
from torch import nn
import torch.nn.functional as F

INPUT, HIDDEN, OUTPUT = 64, 16, 10
N = INPUT*HIDDEN+HIDDEN+HIDDEN*OUTPUT+OUTPUT
K, R, GAMMA = 3, 3, 0.16
assert N == 1210 and N % 2 == 0


def _unit_householder(round_idx: int, mirror_idx: int, n: int, *, device=None, dtype=torch.float64):
    i = torch.arange(n, dtype=dtype, device=device)
    u = torch.sin((0.031 + 0.003*round_idx)*i+0.19*mirror_idx) + torch.cos((0.047 + 0.002*mirror_idx)*i+.14*round_idx)
    return u / torch.linalg.vector_norm(u)


def givens(v, theta):
    v2=v.reshape(-1,2)
    c=math.cos(theta);s=math.sin(theta)
    return torch.stack((c*v2[:,0]-s*v2[:,1], s*v2[:,0]+c*v2[:,1]),dim=-1).reshape(-1)


def mirror_q(v, r, k, inverse=False):
    """Q=Householder*Givens*permutation, exactly orthogonal."""
    ang=.31 + .19*r + .29*k
    shift=17+11*r+29*k
    u=_unit_householder(r,k,v.numel(),device=v.device,dtype=v.dtype)
    if not inverse:
        q=givens(torch.roll(v,shift),ang)
        return q-2*u*torch.dot(u,q)
    q=v-2*u*torch.dot(u,v)
    return torch.roll(givens(q,-ang),-shift)


def seed_transform(z, inverse=False):
    """Deterministic invertible mixing; contains no stored learned weights."""
    if not inverse:
        for r in range(4):z=mirror_q(z,11+r,(r+1)%K)
    else:
        for r in range(3,-1,-1):z=mirror_q(z,11+r,(r+1)%K,True)
    return z


def mirror_increment(z,r, K=K):
    """1-Lipschitz average in Euclidean norm (orthogonal Q + tanh)."""
    n=z.numel()
    idx=torch.arange(n,device=z.device,dtype=z.dtype)
    acc=torch.zeros_like(z)
    for k in range(K):
        q=mirror_q(z,r,k)
        bias=.17*torch.sin(.013*idx+1.07*r+.37*k)
        acc=acc+mirror_q(torch.tanh(q+bias),r,k,True)
    return acc/K


def nonlinear_orbit(z, rounds=R, gamma=GAMMA, K=K):
    for r in range(rounds):
        z=z+gamma*mirror_increment(z,r,K)
    return z


def invert_nonlinear_orbit(w, rounds=R, gamma=GAMMA, K=K, iterations=65):
    """Exact up to floating error: z + gamma F(z)=w, contraction for gamma<1.

    Every F is 1-Lipschitz, so Picard inverse converges globally. Reverses R
    nonlinear residual Mirror layers, then undoes seed mixing.
    """
    z=w.clone()
    for r in range(rounds-1,-1,-1):
        y=z
        x=y.clone()
        for _ in range(iterations):x=y-gamma*mirror_increment(x,r,K)
        z=x
    return seed_transform(z,inverse=True)


def weight_to_flat(model):
    return torch.cat((model.l1.weight.detach().reshape(-1),model.l1.bias.detach(),
                      model.l2.weight.detach().reshape(-1),model.l2.bias.detach()))


def eval_flat(flat,x):
    a=flat[:1024].reshape(16,64)
    b=flat[1024:1040]
    c=flat[1040:1200].reshape(10,16)
    d=flat[1200:1210]
    return F.linear(torch.tanh(F.linear(x,a,b)),c,d)


class NativeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=nn.Linear(INPUT,HIDDEN,dtype=torch.float64)
        self.l2=nn.Linear(HIDDEN,OUTPUT,dtype=torch.float64)
    def forward(self,x):return self.l2(torch.tanh(self.l1(x)))


class MirrorMLP(nn.Module):
    def __init__(self,m=N,nonlinear=True):
        super().__init__()
        if not (1<=m<=N):raise ValueError(m)
        self.m=int(m);self.nonlinear=bool(nonlinear)
        self.coords=nn.Parameter(torch.zeros(m,dtype=torch.float64))
    def weights(self):
        # For m<N retain only first m coordinates in canonical *mixed* basis.
        seed=F.pad(self.coords,(0,N-self.m))
        z=seed_transform(seed)
        return nonlinear_orbit(z) if self.nonlinear else z
    def forward(self,x):return eval_flat(self.weights(),x)


def payload_bytes(kind,model):
    buf=io.BytesIO()
    config={'kind':kind,'dims':[INPUT,HIDDEN,OUTPUT]}
    if isinstance(model,MirrorMLP):config.update({'m':model.m,'nonlinear':model.nonlinear,'K':K,'R':R,'gamma':GAMMA})
    torch.save({'config':config,'weights':model.state_dict()},buf)
    return len(buf.getvalue())


def compile_exact(native,compiled):
    if compiled.m!=N or not compiled.nonlinear:raise ValueError('full nonlinear case only')
    target=weight_to_flat(native)
    with torch.no_grad():compiled.coords.copy_(invert_nonlinear_orbit(target))
    return compiled
