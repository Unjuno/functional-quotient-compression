"""Matrix-free learned plane rotations; input [..., roles, width]."""
import torch
from torch import Tensor

def orthonormal_planes(a:Tensor,b:Tensor,eps:float=1e-7)->tuple[Tensor,Tensor]:
    if a.shape!=b.shape or a.ndim!=2: raise ValueError('axes must be matching [planes,width] tensors')
    an=a.norm(dim=-1,keepdim=True)
    if bool((an.detach()<eps).any()): raise ValueError('degenerate first axis')
    u=a/an
    v=b-(b*u).sum(-1,keepdim=True)*u
    # Reorthogonalize before exact normalization; no additive epsilon bias.
    v=v-(v*u).sum(-1,keepdim=True)*u
    vn=v.norm(dim=-1,keepdim=True)
    if bool((vn.detach()<eps).any()): raise ValueError('degenerate second axis')
    return u,v/vn

def apply_planes(x:Tensor,a:Tensor,b:Tensor,angles:Tensor)->Tensor:
    u,v=orthonormal_planes(a,b)
    for j in range(a.shape[0]):
        alpha=(x*u[j]).sum(-1,keepdim=True); beta=(x*v[j]).sum(-1,keepdim=True)
        phi=angles[...,j,None]; c=torch.cos(phi); s=torch.sin(phi)
        x=x+((c-1)*alpha-s*beta)*u[j]+(s*alpha+(c-1)*beta)*v[j]
    return x

def householder_alias(x:Tensor,a:Tensor,b:Tensor,angles:Tensor)->Tensor:
    """Two reflections per plane give the same rotation, independent implementation."""
    u,v=orthonormal_planes(a,b)
    for j in range(a.shape[0]):
        phi=angles[...,j,None]*.5
        n=torch.cos(phi)*u[j]+torch.sin(phi)*v[j]
        x=x-2*(x*u[j]).sum(-1,keepdim=True)*u[j]
        x=x-2*(x*n).sum(-1,keepdim=True)*n
    return x
