"""Exact fixed-address Mirror identities. Column-vector convention for matrices."""
from __future__ import annotations
import torch
from torch.nn import functional as F

def conjugate(v:torch.Tensor,q:torch.Tensor)->torch.Tensor:
    return F.gelu(v@q.T,approximate='tanh')@torch.linalg.inv(q).T

def shear_matrix(shear,view:int,dtype=torch.float64)->torch.Tensor:
    if view not in range(4):raise ValueError('view must be 0..3')
    width=shear.src.shape[1]*2
    q=torch.eye(width,dtype=dtype,device=shear.src.device)
    q[shear.dst[view],shear.src[view]]=shear.rho*shear.sign[view].to(dtype)
    return q

def mirror_features(m,x:torch.Tensor,view:int)->torch.Tensor:
    ids=torch.full((len(x),),view,dtype=torch.long,device=x.device)
    h=m.shears[0](F.linear(x,m.w1,m.b1),ids)
    h=m.shears[1](F.linear(h,m.w2,m.b2),ids)
    return F.linear(h,m.w3,m.b3)

def fold_parameters(m,view:int)->dict[str,torch.Tensor]:
    q1=shear_matrix(m.shears[0],view,m.w1.dtype)
    q2=shear_matrix(m.shears[1],view,m.w1.dtype)
    # For these nilpotent shears the inverse is exact I-A; avoid a solver.
    i=torch.eye(m.config.width,dtype=m.w1.dtype,device=m.w1.device)
    r1=2*i-q1;r2=2*i-q2
    return {'w1':q1@m.w1,'b1':q1@m.b1,'w2':q2@m.w2@r1,'b2':q2@m.b2,
            'w3':m.w3@r2,'b3':m.b3}

def folded_forward(x:torch.Tensor,p:dict[str,torch.Tensor])->torch.Tensor:
    h=F.gelu(F.linear(x,p['w1'],p['b1']),approximate='tanh')
    h=F.gelu(F.linear(h,p['w2'],p['b2']),approximate='tanh')
    return F.linear(h,p['w3'],p['b3'])
