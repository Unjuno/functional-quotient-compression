"""Natural-task fully independent SwiGLU teachers and shared Cartesian-view students."""
from __future__ import annotations
import io
import json
import torch
from torch import nn
from torch.nn import functional as F

K=3
H=32
D=64
A_SETS=({0,2,4,6,8},{2,3,5,7},{4,5,6,7,8,9})
B_SETS=({0,1,2,3,5,8},{1,3,4,6,7,9},{0,2,4,5,7,9})
HOLD=(2,3,7)
SEEN=tuple(i for i in range(9) if i not in HOLD)
METHODS=("tied","mirror","gain","pair_mirror","pair_readout","independent")

def labels_digits(labels):
    """N integer labels -> N x 9 natural compositional binary labels."""
    return torch.stack([((torch.isin(labels,torch.tensor(tuple(A_SETS[i]),dtype=labels.dtype))) &
                          torch.isin(labels,torch.tensor(tuple(B_SETS[j]),dtype=labels.dtype))).float()
                        for i in range(3) for j in range(3)],dim=1)


def givens(z,angle):
    """z [B,N,H], angles [N,q] applied to first q disjoint 2D planes."""
    q=angle.shape[-1]
    p=z[...,:2*q].reshape(z.shape[0],z.shape[1],q,2)
    c=torch.cos(angle)[None,...]
    s=torch.sin(angle)[None,...]
    a=c*p[...,0]-s*p[...,1]
    b=s*p[...,0]+c*p[...,1]
    return torch.cat([torch.stack((a,b),dim=-1).flatten(-2),z[...,2*q:]],dim=-1)


def gain(z,delta):
    q=delta.shape[-1]
    p=z[...,:2*q].reshape(z.shape[0],z.shape[1],q,2)
    v=p*(1+delta[None,:,:,None])
    return torch.cat([v.flatten(-2),z[...,2*q:]],dim=-1)

class Teachers(nn.Module):
    """Independent weights, no shared intermediate parameters between 9 tasks."""
    def __init__(self,d=D,h=H):
        super().__init__()
        self.wg=nn.Parameter(torch.randn(9,d,h)*0.10)
        self.wu=nn.Parameter(torch.randn(9,d,h)*0.10)
        self.wd=nn.Parameter(torch.randn(9,h)*0.11)
        self.bg=nn.Parameter(torch.zeros(9,h))
        self.bu=nn.Parameter(torch.zeros(9,h))
        self.bo=nn.Parameter(torch.zeros(9))
    def forward(self,x):
        g=torch.einsum('bd,ndh->bnh',x,self.wg)+self.bg
        u=torch.einsum('bd,ndh->bnh',x,self.wu)+self.bu
        return torch.einsum('bnh,nh->bn',F.silu(g)*u,self.wd)+self.bo

class Student(nn.Module):
    def __init__(self,method,d=D,h=H):
        super().__init__()
        assert method in METHODS
        self.method=method
        if method=='independent':
            self.wg=nn.Parameter(torch.randn(9,d,h)*0.10)
            self.wu=nn.Parameter(torch.randn(9,d,h)*0.10)
            self.wd=nn.Parameter(torch.randn(9,h)*0.11)
            self.bg=nn.Parameter(torch.zeros(9,h))
            self.bu=nn.Parameter(torch.zeros(9,h))
        else:
            self.wg=nn.Parameter(torch.randn(d,h)*0.10)
            self.wu=nn.Parameter(torch.randn(d,h)*0.10)
            if method!='pair_readout':
                self.wd=nn.Parameter(torch.randn(h)*0.11)
            self.bg=nn.Parameter(torch.zeros(h))
            self.bu=nn.Parameter(torch.zeros(h))
        if method=='independent':
            self.bo=nn.Parameter(torch.zeros(K*K))
        else:
            self.gbo=nn.Parameter(torch.zeros(K))
            self.ubo=nn.Parameter(torch.zeros(K))
        if method in ('mirror','gain'):
            self.gangle=nn.Parameter(torch.zeros(K,12))
            self.uangle=nn.Parameter(torch.zeros(K,12))
        if method=='pair_mirror':
            self.gangle=nn.Parameter(torch.zeros(K*K,4))
            self.uangle=nn.Parameter(torch.zeros(K*K,4))
        if method=='pair_readout':
            self.wd_pair=nn.Parameter(torch.randn(9,h)*0.11)
    def forward(self,x):
        bias=(self.bo[None,:] if self.method=='independent' else (self.gbo[:,None]+self.ubo[None,:]).reshape(1,9))
        if self.method=='independent':
            g=torch.einsum('bd,ndh->bnh',x,self.wg)+self.bg
            u=torch.einsum('bd,ndh->bnh',x,self.wu)+self.bu
            return torch.einsum('bnh,nh->bn',F.silu(g)*u,self.wd)+bias
        g=x@self.wg+self.bg
        u=x@self.wu+self.bu
        if self.method=='tied':return ((F.silu(g)*u)@self.wd)[:,None].expand(-1,9)+bias
        if self.method=='pair_readout':
            return (F.silu(g)*u)@self.wd_pair.T +bias
        if self.method=='pair_mirror':
            gv=givens(g[:,None,:].expand(-1,9,-1),self.gangle)
            uv=givens(u[:,None,:].expand(-1,9,-1),self.uangle)
            return (F.silu(gv)*uv)@self.wd +bias
        g=g[:,None,:].expand(-1,K,-1)
        u=u[:,None,:].expand(-1,K,-1)
        fn=givens if self.method=='mirror' else gain
        g=fn(g,self.gangle)
        u=fn(u,self.uangle)
        return (F.silu(g)[:,:,None,:]*u[:,None,:,:]@self.wd).reshape(-1,9)+bias


def payload(model,extra=None):
    buf=io.BytesIO()
    cfg={'class':model.__class__.__name__,'method':getattr(model,'method','teacher'),'d':D,'h':H,'K':K}
    if extra:cfg.update(extra)
    torch.save({'config':cfg,'state':{k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}},buf)
    return buf.getvalue()


def reload_payload(bytestring):
    obj=torch.load(io.BytesIO(bytestring),weights_only=True,map_location='cpu')
    assert obj['config']['class']=='Student'
    model=Student(obj['config']['method'])
    model.load_state_dict(obj['state'])
    return model
