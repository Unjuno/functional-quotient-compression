"""Dual-input Mirror SwiGLU: shared 3 matrices + composable gate/up view codes.

A small CPU mechanism study; all tensors are real tensors, not theoretical bitcounts.
"""
from __future__ import annotations
import io
import json
import math
import torch
from torch import nn
from torch.nn import functional as F

METHODS = ('tied','mirror','gain','mirror_gate','mirror_up','pair_mirror','pair_gain','factor_lora1','independent')


def transform(x: torch.Tensor, angles: torch.Tensor, q: int, kind: str):
    """x: [batch, nview, h], angles: [nview, qmax], q active disjoint 2-plane coords."""
    h = x.shape[-1]
    assert 2*q <= h
    p = x[..., :2*q].reshape(x.shape[0], x.shape[1], q, 2)
    a = angles[:, :q].unsqueeze(0)
    if kind=='rotation':
        c,s = a.cos(),a.sin()
        x0=c*p[...,0]-s*p[...,1]
        x1=s*p[...,0]+c*p[...,1]
    elif kind=='gain':
        sc=1.0+a  # exactly identity at a=0; unconstrained native scale
        x0=p[...,0]*sc
        x1=p[...,1]*sc
    else:raise ValueError(kind)
    return torch.cat([torch.stack([x0,x1],dim=-1).reshape(x.shape[0],x.shape[1],2*q),x[...,2*q:]],dim=-1)

class DualSwiGLU(nn.Module):
    def __init__(self,method: str,d=8,h=24,K=3,q=12):
        super().__init__()
        assert method in METHODS and 1<=q<=h//2
        self.method,self.d,self.h,self.K,self.max_q=method,d,h,K,q
        self.active_q=q
        if method=='independent':
            self.wg=nn.Parameter(torch.empty(K*K,d,h))
            self.wu=nn.Parameter(torch.empty(K*K,d,h))
            self.wd=nn.Parameter(torch.empty(K*K,h,d))
        else:
            self.wg=nn.Parameter(torch.empty(d,h))
            self.wu=nn.Parameter(torch.empty(d,h))
            self.wd=nn.Parameter(torch.empty(h,d))
        for p in (self.wg,self.wu): nn.init.normal_(p,mean=0,std=0.32)
        nn.init.normal_(self.wd,mean=0,std=0.23)
        if method in ('mirror','gain','mirror_gate','mirror_up'):
            self.gangle=nn.Parameter(torch.zeros(K,q))
            self.uangle=nn.Parameter(torch.zeros(K,q))
        if method in ('pair_mirror','pair_gain'):
            self.gangle=nn.Parameter(torch.zeros(K*K,q))
            self.uangle=nn.Parameter(torch.zeros(K*K,q))
        if method=='factor_lora1':
            self.ag=nn.Parameter(torch.randn(K,d)*0.04)
            self.bg=nn.Parameter(torch.zeros(K,h))
            self.au=nn.Parameter(torch.randn(K,d)*0.04)
            self.bu=nn.Parameter(torch.zeros(K,h))

    def set_active(self,q:int):
        assert 1<=q<=self.max_q
        self.active_q=q

    def active_trainable_params(self):
        common=3*self.d*self.h
        if self.method=='independent':return self.K*self.K*common
        if self.method=='tied':return common
        if self.method=='factor_lora1':return common + 2*self.K*(self.d+self.h)
        n = self.K if self.method in ('mirror','gain','mirror_gate','mirror_up') else self.K*self.K
        return common + 2*n*self.active_q

    def forward(self,x):
        b=x.shape[0];K=self.K; h=self.h
        if self.method=='independent':
            g=torch.einsum('bd,kdh->bkh',x,self.wg)
            u=torch.einsum('bd,kdh->bkh',x,self.wu)
            return torch.einsum('bkh,khd->bkd',F.silu(g)*u,self.wd)
        g=(x@self.wg)
        u=(x@self.wu)
        if self.method=='tied':
            y=(F.silu(g)*u)@self.wd
            return y[:,None,:].expand(-1,K*K,-1)
        if self.method=='factor_lora1':
            gv=g[:,None,:] + torch.einsum('bd,kd,kh->bkh',x,self.ag,self.bg)
            uv=u[:,None,:] + torch.einsum('bd,kd,kh->bkh',x,self.au,self.bu)
            return (F.silu(gv)[:,:,None,:]*uv[:,None,:,:]@self.wd).reshape(b,K*K,self.d)
        mode='rotation' if 'mirror' in self.method else 'gain'
        if self.method.startswith('pair_'):
            g=transform(g[:,None,:].expand(-1,K*K,-1),self.gangle,self.active_q,mode)
            u=transform(u[:,None,:].expand(-1,K*K,-1),self.uangle,self.active_q,mode)
            return (F.silu(g)*u)@self.wd
        g=g[:,None,:].expand(-1,K,-1)
        u=u[:,None,:].expand(-1,K,-1)
        if self.method!='mirror_up':g=transform(g,self.gangle,self.active_q,mode)
        if self.method!='mirror_gate':u=transform(u,self.uangle,self.active_q,mode)
        z=F.silu(g)[:,:,None,:]*u[:,None,:,:]
        return (z@self.wd).reshape(b,K*K,self.d)


def run_model(m,x):
    return m(x)


def serialize_inference(m):
    """Only active learned coordinates are retained; metadata and tensor keys are paid."""
    cfg={'method':m.method,'d':m.d,'h':m.h,'K':m.K,'q':m.active_q}
    state={}
    for k,v in m.state_dict().items():
        if k.endswith('angle'):
            state[k]=v[:,:m.active_q].detach().cpu().contiguous()
        else:state[k]=v.detach().cpu().contiguous()
    buf=io.BytesIO()
    torch.save({'config':cfg,'state':state},buf)
    return buf.getvalue()


def infer_serialized(data,x):
    buf=io.BytesIO(data)
    obj=torch.load(buf,map_location='cpu',weights_only=True)
    cfg=obj['config']
    m=DualSwiGLU(cfg['method'],d=cfg['d'],h=cfg['h'],K=cfg['K'],q=cfg['q'])
    m.load_state_dict(obj['state'])
    m.eval()
    with torch.no_grad():return m(x)


def generate_world(seed:int,family:str,d=8,h=24,K=3):
    assert family in ('aligned','partial','unrelated')
    # Independent RNG streams ensure data and model generation do not leak across splits.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if family in ('aligned','partial'):
            teacher=DualSwiGLU('mirror',d,h,K,q=h//2)
            with torch.no_grad():
                teacher.gangle.uniform_(-0.80,0.80)
                teacher.uangle.uniform_(-0.80,0.80)
        else:
            teacher=DualSwiGLU('independent',d,h,K,q=h//2)
        teacher.eval()
        interactions=None
        if family=='partial':
            # Pair-specific output residuals break factorization without changing teacher's shared backbone.
            interactions=torch.randn(K*K,d,d)*0.048
        pieces={}
        for k,n in [('train',512),('val',256),('test',512)]:
            x=torch.randn(n,d)
            with torch.no_grad():
                y=teacher(x)
                if interactions is not None:
                    y=y+torch.einsum('bd,kde->bke',torch.tanh(x),interactions)
            pieces[k]=(x,y)
        train_x,train_y=pieces['train']
        train_std=float(train_y.square().mean().sqrt().clamp_min(0.1))
        train_noise=torch.randn_like(train_y)*train_std*0.035
        pieces['train']=(train_x,train_y+train_noise)
        pieces['heldout_pairs']=[2,3,7]
        pieces['family']=family
        pieces['world']=seed
        pieces['scale']=train_std
        return pieces
