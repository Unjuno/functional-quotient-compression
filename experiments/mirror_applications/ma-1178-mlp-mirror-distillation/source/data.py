"""Independent teacher training inputs and fully separated student train/dev/fresh/OOD splits."""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F
from model import D,RULES,teacher_outputs


def draw_split(seed:int,n:int,scale:float=1.):
    gen=torch.Generator().manual_seed(seed)
    x=(torch.rand(n,D,generator=gen)*2-1)*scale
    rule=torch.randint(RULES,(n,),generator=gen)
    return x,rule


def analytic_target(x:torch.Tensor, teacher_id:int, seed:int):
    # Distinct nonlinear functions per independent teacher, not simply target-label copies.
    g=torch.Generator().manual_seed(seed*451+teacher_id*997+123)
    w=torch.randn(12,D,generator=g)*1.20
    b=torch.randn(12,generator=g)*0.35
    a=torch.randn(D,12,generator=g)*0.25
    return torch.tanh(x@w.T+b)@a.T + 0.15*torch.sin((x*x)@w.square().T)@a.T


def student_sets(world:str,seed:int,teachers,n_train=1024,n_dev=384):
    w=['mirror_related','film_related','independent'].index(world)
    x,r=draw_split(100_000+seed*100+w*1000,n_train)
    dx,dr=draw_split(200_000+seed*100+w*1000,n_dev)
    y=teacher_outputs(teachers,x,r)
    dy=teacher_outputs(teachers,dx,dr)
    norm=y.square().mean().clamp_min(1e-5).item()
    return (x,r,y),(dx,dr,dy),norm


def fresh_sets(world:str,seed:int,teachers,n_fresh=1024,n_ood=1024):
    w=['mirror_related','film_related','independent'].index(world)
    x,r=draw_split(300_000+seed*100+w*1000,n_fresh)
    ox,orr=draw_split(400_000+seed*100+w*1000,n_ood,scale=1.6)
    return (x,r,teacher_outputs(teachers,x,r)),(ox,orr,teacher_outputs(teachers,ox,orr))
