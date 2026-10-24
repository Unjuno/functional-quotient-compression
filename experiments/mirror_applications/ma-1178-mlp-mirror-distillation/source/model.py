"""MA-1178 functional teacher->Mirror distillation: model definitions."""
from __future__ import annotations
import math
from typing import Literal
import torch
from torch import nn
from torch.nn import functional as F
D=4
TEACHER_WIDTH=16
RULES=3
WIDTHS=(8,16,24)
INDEPENDENT_WIDTHS={8:3,16:7,24:10}


def reflect(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Fused Householder reflection: O(H) math per input instead of forming HxH matrices."""
    denom=v.square().sum(dim=-1,keepdim=True).clamp_min(1e-12)
    return u - 2*v*((u*v).sum(dim=-1,keepdim=True)/denom)


def householder(v: torch.Tensor) -> torch.Tensor:
    """Apply I - 2 uu^T / (u^T u), with learnable nonzero direction vectors."""
    denominator=v.square().sum(dim=-1,keepdim=True).clamp_min(1e-12)
    return torch.eye(v.shape[-1], device=v.device, dtype=v.dtype) - 2 * (v.unsqueeze(-1) @ v.unsqueeze(-2)) / denominator.unsqueeze(-1)


class DenseMLP(nn.Module):
    def __init__(self,h:int=16):
        super().__init__()
        self.up=nn.Linear(D,h)
        self.down=nn.Linear(h,D)
    def forward(self,x):return self.down(torch.tanh(self.up(x)))


class SharedStudent(nn.Module):
    def __init__(self,h:int,kind:Literal['mirror','film','shared']):
        super().__init__()
        assert kind in ('mirror','film','shared')
        self.h=h;self.kind=kind
        self.up=nn.Linear(D,h)
        self.down=nn.Linear(h,D)
        if kind!='shared':
            self.codes=nn.Parameter(0.45*torch.randn(RULES,h))
    def forward(self,x,rule):
        u=self.up(x)
        if self.kind=='mirror':
            v=self.codes[rule]
            z=reflect(u,v)
            z=reflect(torch.tanh(z),v)
        elif self.kind=='film':
            gamma=1+0.85*torch.tanh(self.codes[rule])
            z=torch.tanh(gamma*u)
        else:
            z=torch.tanh(u)
        return self.down(z)


class IndependentStudent(nn.Module):
    def __init__(self,h:int):
        super().__init__()
        self.h=h
        self.up=nn.Parameter(torch.randn(RULES,h,D)*math.sqrt(2/(D+h)))
        self.up_bias=nn.Parameter(torch.zeros(RULES,h))
        self.down=nn.Parameter(torch.randn(RULES,D,h)*math.sqrt(2/(D+h)))
        self.down_bias=nn.Parameter(torch.zeros(RULES,D))
    def forward(self,x,rule):
        a=torch.einsum('bhd,bd->bh',self.up[rule],x)+self.up_bias[rule]
        y=torch.einsum('bdh,bh->bd',self.down[rule],torch.tanh(a))+self.down_bias[rule]
        return y


def new_student(kind:str,h:int,seed:int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return IndependentStudent(INDEPENDENT_WIDTHS[h]) if kind=='independent' else SharedStudent(h,kind)


def param_count(x):return sum(v.numel() for v in x.parameters())


def teacher_mirror_family(base:DenseMLP,directions:torch.Tensor)->list[DenseMLP]:
    """Construct ordinary DenseMLP teachers that exactly implement distinct Mirror views of a trained base teacher."""
    h=base.up.weight.shape[0]
    hmat=householder(directions)
    out=[]
    for m in range(RULES):
        t=DenseMLP(h)
        with torch.no_grad():
            t.up.weight.copy_(hmat[m]@base.up.weight)
            t.up.bias.copy_(hmat[m]@base.up.bias)
            t.down.weight.copy_(base.down.weight@hmat[m])
            t.down.bias.copy_(base.down.bias)
        t.requires_grad_(False);t.eval();out.append(t)
    return out


def teacher_film_family(base:DenseMLP,codes:torch.Tensor)->list[DenseMLP]:
    h=base.up.weight.shape[0];gamma=1+0.85*torch.tanh(codes)
    out=[]
    for m in range(RULES):
        t=DenseMLP(h)
        with torch.no_grad():
            t.up.weight.copy_(gamma[m,:,None]*base.up.weight)
            t.up.bias.copy_(gamma[m]*base.up.bias)
            t.down.weight.copy_(base.down.weight)
            t.down.bias.copy_(base.down.bias)
        t.requires_grad_(False);t.eval();out.append(t)
    return out


def teacher_outputs(teachers:list[DenseMLP],x:torch.Tensor,rule:torch.Tensor):
    with torch.no_grad():
        # Inference for selected rule; all 3 teachers are ordinary learned-weight MLPs.
        ys=torch.stack([t(x) for t in teachers],dim=1)
        return ys[torch.arange(x.shape[0]),rule]


def packet_bytes(model):
    import io
    d={'architecture':type(model).__name__, 'kind':getattr(model,'kind','independent'), 'state':{k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}}
    b=io.BytesIO();torch.save(d,b);return len(b.getvalue())
