"""Small conditional nonlinear shared-kernel models. No training/eval sets built at module import."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DIN=8; H=12; DOUT=3; K=3

def reflect(z,u):
    u=F.normalize(u,dim=-1,eps=1e-12)
    return z-2.0*(z*u).sum(dim=-1,keepdim=True)*u

def rankone(z,u,beta):
    u=F.normalize(u,dim=-1,eps=1e-12)
    return z+beta*(z*u).sum(dim=-1,keepdim=True)*u

def activate(x,kind):
    if kind=='relu': return F.relu(x)
    if kind=='silu': return F.silu(x)
    raise ValueError(kind)

class ConditionalMLP(nn.Module):
    def __init__(self, arm, act='relu', initseed=0):
        super().__init__()
        self.arm=arm;self.act=act
        width=5 if arm=='ind_budget' else H
        self.width=width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initseed)
            if arm.startswith('ind_'):
                self.up=nn.Parameter(torch.empty(K,width,DIN));self.ub=nn.Parameter(torch.zeros(K,width))
                self.down=nn.Parameter(torch.empty(K,DOUT,width));self.db=nn.Parameter(torch.zeros(K,DOUT))
                nn.init.xavier_uniform_(self.up);nn.init.xavier_uniform_(self.down)
            else:
                self.up=nn.Parameter(torch.empty(H,DIN));self.ub=nn.Parameter(torch.zeros(H))
                self.down=nn.Parameter(torch.empty(DOUT,H));self.db=nn.Parameter(torch.zeros(DOUT))
                nn.init.xavier_uniform_(self.up);nn.init.xavier_uniform_(self.down)
                if arm in ('mirror','coupled','fixed_mirror'):
                    code=torch.randn(K,H)
                    if arm=='fixed_mirror': self.register_buffer('code',code)
                    else: self.code=nn.Parameter(code)
                    if arm=='coupled': self.beta=nn.Parameter(torch.full((K,1),-2.0))
                elif arm=='film':self.code=nn.Parameter(torch.randn(K,H)*0.27)
    def forward(self,x,task):
        if self.arm.startswith('ind_'):
            a=torch.einsum('bhi,bi->bh',self.up[task],x)+self.ub[task]
            return torch.einsum('boh,bh->bo',self.down[task],activate(a,self.act))+self.db[task]
        a=F.linear(x,self.up,self.ub)
        if self.arm in ('mirror','fixed_mirror'):
            u=self.code[task]; a=reflect(a,u);z=reflect(activate(a,self.act),u)
        elif self.arm=='coupled':
            u=self.code[task]; b=self.beta[task];a=rankone(a,u,b);z=rankone(activate(a,self.act),u,b)
        elif self.arm=='film':
            gain=1.+self.code[task];z=gain*activate(gain*a,self.act)
        elif self.arm=='shared':z=activate(a,self.act)
        else:raise ValueError(self.arm)
        return F.linear(z,self.down,self.db)
    def physical_params(self):
        return sum(p.numel() for p in self.parameters())

class Teacher:
    def __init__(self,seed,act,gamma):
        g=torch.Generator(device='cpu').manual_seed(int(seed)+772219)
        rand=lambda *shape: torch.randn(*shape,generator=g)
        self.act=act;self.gamma=gamma
        self.a=rand(H,DIN)*0.54;self.ab=rand(H)*0.14
        self.b=rand(DOUT,H)*0.37;self.bb=rand(DOUT)*0.1
        self.codes=rand(K,H)
        self.ia=rand(K,H,DIN)*0.54;self.iab=rand(K,H)*0.14
        self.ib=rand(K,DOUT,H)*0.37;self.ibb=rand(K,DOUT)*0.1
        # Global normalization over unobserved synthetic calibration inputs.
        x=make_inputs(3097+int(seed),4096,'id')
        t=torch.arange(K).repeat_interleave(len(x))
        outputs=self.raw(x.repeat(K,1),t)
        self.mean=outputs.mean(0)
        self.scale=outputs.std(dim=0).clamp_min(0.05)
    def raw(self,x,t):
        a=F.linear(x,self.a,self.ab)
        u=self.codes[t]
        y=F.linear(reflect(activate(reflect(a,u),self.act),u),self.b,self.bb)
        ai=torch.einsum('bhi,bi->bh',self.ia[t],x)+self.iab[t]
        yi=torch.einsum('boh,bh->bo',self.ib[t],activate(ai,self.act))+self.ibb[t]
        return (1-self.gamma)*y+self.gamma*yi
    @torch.no_grad()
    def output(self,x,t):return (self.raw(x,t)-self.mean)/self.scale

def make_inputs(seed,n,world):
    g=torch.Generator(device='cpu').manual_seed(int(seed))
    x=torch.randn(n,DIN,generator=g)
    # Each dimension has unequal significance; also includes weak and strong examples.
    x=x*(0.60+0.40*torch.rand(n,1,generator=g))
    if world=='ood':x=x*1.5
    return x

def dataset(teacher,seed,split,n):
    # task identities are provided, no oracle hidden-feature identifiers or teacher weights.
    offset={'train':1111,'dev':2222,'fresh':3333,'ood':4444}[split]
    xs=make_inputs(100000+seed*101+offset,n,'ood' if split=='ood' else 'id')
    x=xs.repeat(K,1)
    t=torch.arange(K,dtype=torch.long).repeat_interleave(n)
    with torch.no_grad():y=teacher.output(x,t)
    return x,t,y
