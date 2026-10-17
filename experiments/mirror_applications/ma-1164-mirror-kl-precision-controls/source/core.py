"""MA-1164: one physical coefficient vector, parameterized K/R nonlinear Mirror views.

No learned decoder, no future labels used during mechanical compilation.
"""
from __future__ import annotations
from functools import lru_cache
import io, math
import torch
from torch import nn
from torch.nn import functional as F

INPUT, HIDDEN, OUTPUT, N = 64, 16, 10, 1210
GAMMA = 0.16

@lru_cache(maxsize=128)
def unit_householder(r:int, k:int):
    i = torch.arange(N, dtype=torch.float64)
    u = torch.sin((.031+.003*r)*i+.19*k) + torch.cos((.047+.002*k)*i+.14*r)
    return u/torch.linalg.vector_norm(u)


def givens(v, angle):
    assert v.shape==(N,), (v.shape,N)
    ab=v.reshape(-1,2)
    c,s=math.cos(angle),math.sin(angle)
    return torch.stack((c*ab[:,0]-s*ab[:,1],s*ab[:,0]+c*ab[:,1]),dim=-1).reshape(-1)


def mirror_q(z,r,k,inverse=False):
    angle=.31+.19*r+.29*k
    shift=17+11*r+29*k
    u=unit_householder(r,k)
    if not inverse:
        v=givens(torch.roll(z,shift),angle)
        return v-2*u*torch.dot(u,v)
    v=z-2*u*torch.dot(u,z)
    return torch.roll(givens(v,-angle),-shift)


def seed_transform(z,inverse=False):
    # Identical for all K/R; ensures K/R comparisons change ONLY nonlinear Mirror.
    if not inverse:
        for r in range(4):z=mirror_q(z,11+r,(r+1)%3)
    else:
        for r in reversed(range(4)):z=mirror_q(z,11+r,(r+1)%3,True)
    return z


def step(z,r,k_count):
    idx=torch.arange(N,dtype=torch.float64)
    out=torch.zeros_like(z)
    for k in range(k_count):
        bias=.17*torch.sin(.013*idx+1.07*r+.37*k)
        out=out+mirror_q(torch.tanh(mirror_q(z,r,k)+bias),r,k,True)
    return z+GAMMA*out/k_count


def orbit(z, k_count, rounds):
    for r in range(rounds):z=step(z,r,k_count)
    return z


def invert_orbit(target,k_count,rounds,iterations=24):
    # ||d F_r / dz|| <= 1 for orthogonal Q, tanh and convex average;
    # Picard inverse contracts by GAMMA < 1.
    z=target.clone()
    for r in reversed(range(rounds)):
        y=z
        x=y.clone()
        for _ in range(iterations):
            x=y-(step(x,r,k_count)-x)
        z=x
    return seed_transform(z,True)


def native_flat(net):
    return torch.cat((net.l1.weight.detach().reshape(-1),net.l1.bias.detach(),
                      net.l2.weight.detach().reshape(-1),net.l2.bias.detach()))


def logits(flat,x):
    return F.linear(torch.tanh(F.linear(x,flat[:1024].reshape(16,64),flat[1024:1040])),
                    flat[1040:1200].reshape(10,16),flat[1200:1210])


class NativeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=nn.Linear(64,16,dtype=torch.float64)
        self.l2=nn.Linear(16,10,dtype=torch.float64)
    def forward(self,x):return self.l2(torch.tanh(self.l1(x)))


class CoeffDecoder(nn.Module):
    def __init__(self, indices, kind='mirror', k=3, rounds=3):
        super().__init__()
        if kind not in ('mirror','plain','raw'):raise ValueError(kind)
        self.kind,self.k,self.rounds=kind,k,rounds
        self.register_buffer('indices',torch.as_tensor(indices,dtype=torch.int16).clone())
        assert self.indices.numel()>0 and self.indices.min()>=0 and self.indices.max()<N
        assert self.indices.unique().numel()==self.indices.numel()
        self.coords=nn.Parameter(torch.zeros(len(indices),dtype=torch.float64))
    def weights(self):
        z=torch.zeros(N,dtype=torch.float64)
        z=z.scatter(0,self.indices.long(),self.coords)
        if self.kind=='raw':return z
        z=seed_transform(z)
        return orbit(z,self.k,self.rounds) if self.kind=='mirror' else z
    def forward(self,x):return logits(self.weights(),x)
    def payload(self):
        return { 'kind':self.kind,'k':self.k if self.kind=='mirror' else 0,
                 'rounds':self.rounds if self.kind=='mirror' else 0,
                 'gamma':GAMMA if self.kind=='mirror' else 0.,
                 'state':{k:v.detach().cpu().clone() for k,v in self.state_dict().items()} }


def state_bytes(payload):
    s=io.BytesIO();torch.save(payload,s);return len(s.getvalue())


def native_payload(flat,dtype=torch.float64):
    return {'kind':'native', 'stored_dtype':str(dtype), 'state':{'l1_w':flat[:1024].reshape(16,64).to(dtype).clone(),
            'l1_b':flat[1024:1040].to(dtype).clone(),'l2_w':flat[1040:1200].reshape(10,16).to(dtype).clone(),
            'l2_b':flat[1200:1210].to(dtype).clone()}}


def svd_payload(flat,rank):
    w=flat[:1024].reshape(16,64)
    u,s,vh=torch.linalg.svd(w,full_matrices=False)
    return {'kind':'svd','rank':rank, 'U':u[:,:rank]*s[:rank], 'Vh':vh[:rank].contiguous().clone(),
            'b1':flat[1024:1040].clone(), 'w2':flat[1040:1200].reshape(10,16).clone(), 'b2':flat[1200:1210].clone()}


def from_svd(p):
    return torch.cat(((p['U']@p['Vh']).flatten(),p['b1'],p['w2'].flatten(),p['b2']))


def best_svd_under_byte_cap(flat,cap):
    best=None
    for rank in range(1,17):
        p=svd_payload(flat,rank)
        if state_bytes(p)<=cap:best=p
        else:break
    return best
