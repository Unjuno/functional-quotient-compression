"""Four supervised modular functions in a single dual-input SwiGLU.

A compact model of the Mirror code insertion: F(x;theta,m_g[i],m_u[j]).
Not a Transformer and not a pretrained LM.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import io
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class Config:
    p:int=31
    h:int=64
    q:int=16


def mod_dataset(p:int):
    a=torch.arange(p).repeat_interleave(p)
    b=torch.arange(p).repeat(p)
    x=torch.cat([F.one_hot(a,p),F.one_hot(b,p)],dim=1).float()
    t=torch.stack([(ia*a+ib*b)%p for ia in (1,2) for ib in (1,2)],dim=1).long()
    return x,t


def rotate_plane(z,angle,q):
    """z [N,H], angles [V,q]. Return [N,V,H], Givens disjoint planes."""
    n,h=z.shape
    pair=z[:,:2*q].reshape(n,1,q,2)
    an=angle[None,:,:]
    c,s=an.cos(),an.sin()
    a=c*pair[...,0]-s*pair[...,1]
    b=s*pair[...,0]+c*pair[...,1]
    packed=torch.stack([a,b],-1).reshape(n,len(angle),2*q)
    return torch.cat([packed,z[:,None,2*q:].expand(n,len(angle),h-2*q)],dim=-1)


def gain_plane(z,delta,q):
    n,h=z.shape
    pair=z[:,:2*q].reshape(n,1,q,2)
    data=pair*(1.+delta[None,:,:,None])
    return torch.cat([data.reshape(n,len(delta),2*q),z[:,None,2*q:].expand(n,len(delta),h-2*q)],dim=-1)

class MirrorSwiGLU(nn.Module):
    METHODS=('tied','gain','mirror','pair_mirror','independent')
    def __init__(self,cfg:Config,method:str):
        super().__init__()
        assert method in self.METHODS
        assert cfg.h%2==0 and cfg.q>0 and 2*cfg.q<=cfg.h and cfg.q%2==0
        self.cfg=cfg;self.method=method
        in_dim=2*cfg.p
        n=4 if method=='independent' else 1
        self.wg=nn.Parameter(torch.empty(n,in_dim,cfg.h))
        self.wu=nn.Parameter(torch.empty(n,in_dim,cfg.h))
        self.wd=nn.Parameter(torch.empty(n,cfg.h,cfg.p))
        self.bg=nn.Parameter(torch.zeros(n,cfg.h))
        self.bu=nn.Parameter(torch.zeros(n,cfg.h))
        self.bo=nn.Parameter(torch.zeros(n,cfg.p))
        for w in (self.wg,self.wu,self.wd):
            nn.init.normal_(w,std=0.15)
        if method in ('gain','mirror'):
            self.ang_g=nn.Parameter(torch.zeros(2,cfg.q))
            self.ang_u=nn.Parameter(torch.zeros(2,cfg.q))
        elif method=='pair_mirror':
            self.ang_g=nn.Parameter(torch.zeros(4,cfg.q//2))
            self.ang_u=nn.Parameter(torch.zeros(4,cfg.q//2))

    def forward(self,x):
        if self.method=='independent':
            g=torch.einsum('nd,kdh->nkh',x,self.wg)+self.bg
            u=torch.einsum('nd,kdh->nkh',x,self.wu)+self.bu
            return torch.einsum('nkh,khp->nkp',F.silu(g)*u,self.wd)+self.bo[None]
        g=(x@self.wg[0]+self.bg[0]);u=(x@self.wu[0]+self.bu[0])
        if self.method=='tied':
            logits=(F.silu(g)*u)@self.wd[0]+self.bo[0]
            return logits[:,None,:].expand(-1,4,-1)
        if self.method=='pair_mirror':
            gv=rotate_plane(g,self.ang_g,self.cfg.q//2)
            uv=rotate_plane(u,self.ang_u,self.cfg.q//2)
            return (F.silu(gv)*uv)@self.wd[0]+self.bo[0]
        fn=rotate_plane if self.method=='mirror' else gain_plane
        gv=fn(g,self.ang_g,self.cfg.q)
        uv=fn(u,self.ang_u,self.cfg.q)
        features=F.silu(gv[:,:,None,:])*uv[:,None,:,:]
        return (features@self.wd[0]).reshape(len(x),4,self.cfg.p)+self.bo[0]

    def param_count(self):return sum(x.numel() for x in self.parameters())

def checkpoint_bytes(m:MirrorSwiGLU):
    b=io.BytesIO()
    torch.save({'config':asdict(m.cfg),'method':m.method,'state':{k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()}},b)
    return len(b.getvalue())
