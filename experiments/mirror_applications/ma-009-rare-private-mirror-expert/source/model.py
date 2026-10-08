from __future__ import annotations
import io, json
import torch
from torch import nn

METHODS=["full_moe","mirror_all","tied_private_rare","lowrank_all","mirror_private_rare"]
N,D,O,RANK=4,16,12,1


def role_of(x):
    return (x[:,0]>=0).long()*2+(x[:,1]>=0).long()


def givens(x,angles):
    y=x
    for k in range(D//2):
        i,j=2*k,2*k+1;c,s=torch.cos(angles[...,k]),torch.sin(angles[...,k]);vi,vj=y[...,i],y[...,j]
        ni,nj=c*vi-s*vj,s*vi+c*vj
        y=torch.cat((y[...,:i],ni.unsqueeze(-1),nj.unsqueeze(-1),y[...,j+1:]),dim=-1)
    return y


class RoutedExperts(nn.Module):
    def __init__(self,method,seed=0):
        super().__init__()
        if method not in METHODS: raise ValueError(method)
        self.method=method;g=torch.Generator().manual_seed(seed)
        if method=="full_moe": self.weight=nn.Parameter(torch.randn(N,D,O,generator=g)*.08)
        elif method=="mirror_all":
            self.weight=nn.Parameter(torch.randn(D,O,generator=g)*.08);self.angles=nn.Parameter(torch.zeros(N,D//2))
        elif method=="tied_private_rare":
            self.shared=nn.Parameter(torch.randn(D,O,generator=g)*.08);self.rare=nn.Parameter(torch.randn(D,O,generator=g)*.08)
        elif method=="lowrank_all":
            self.shared=nn.Parameter(torch.randn(D,O,generator=g)*.08);self.u=nn.Parameter(torch.randn(N,D,RANK,generator=g)*.05);self.v=nn.Parameter(torch.zeros(N,RANK,O))
        elif method=="mirror_private_rare":
            self.shared=nn.Parameter(torch.randn(D,O,generator=g)*.08);self.rare=nn.Parameter(torch.randn(D,O,generator=g)*.08);self.angles=nn.Parameter(torch.zeros(3,D//2))

    def forward(self,x):
        role=role_of(x)
        if self.method=="full_moe": return torch.einsum("bd,bdo->bo",x,self.weight[role])
        if self.method=="mirror_all":
            views=torch.stack([givens(x,a.expand(x.shape[0],-1)) for a in self.angles],dim=1)
            all_y=torch.einsum("bnd,do->bno",views,self.weight)
            return all_y[torch.arange(x.shape[0],device=x.device),role]
        if self.method=="tied_private_rare":
            y=torch.einsum("bd,do->bo",x,self.shared)
            private=torch.einsum("bd,do->bo",x,self.rare)
            return torch.where((role==0)[:,None],private,y)
        if self.method=="lowrank_all":
            base=torch.einsum("bd,do->bo",x,self.shared)
            residual=torch.einsum("bd,bdr,bro->bo",x,self.u[role],self.v[role])
            return base+residual
        # Role zero receives a private matrix; roles 1..3 use shared Mirror views.
        result=torch.einsum("bd,do->bo",x,self.rare)
        for r in range(1,N):
            mask=role==r
            if mask.any():
                z=givens(x[mask],self.angles[r-1].expand(int(mask.sum()),-1))
                result[mask]=z@self.shared
        return result

    def serialized_payload_bytes(self):
        b=io.BytesIO();torch.save({"state_dict":self.state_dict(),"config":json.dumps({"method":self.method,"n":N,"d":D,"o":O,"rank":RANK,"private_role":0},sort_keys=True)},b);return len(b.getvalue())


def compute_proxy(method,examples):
    # One hard-routed expert is active per example. Givens cost is an approximate dense MAC-equivalent.
    if method=="mirror_all": per=D*O+D
    elif method=="mirror_private_rare": per=D*O+D
    elif method=="lowrank_all": per=D*O+(D+O)*RANK
    else: per=D*O
    return examples*per
