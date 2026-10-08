"""Shared and independent linear stitchers for MA-932."""
import hashlib, json, math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

DIM, RANK = 8, 2
MODES=("tied","mirror","film","independent")
REPS=("full","task0_only","unrelated_noise")


def make_split(seed, split, n):
    off={"train":0,"dev":1_000_000,"fresh":2_000_000,"audit":3_000_000}[split]
    g=torch.Generator().manual_seed(seed+off)
    u=torch.randn(n,generator=g);v=torch.randn(n,generator=g)
    nuisance=torch.randn(n,8,generator=g)
    full=nuisance.clone();full[:,0]=u;full[:,1]=v
    partial=nuisance.clone();partial[:,0]=u;partial[:,1]=torch.sin(2.7*nuisance[:,1])+0.25*nuisance[:,1]
    unrelated=torch.randn(n,DIM,generator=g)
    return {"full":full,"task0_only":partial,"unrelated_noise":unrelated},torch.stack((u,v),dim=1)


def givens(angle):
    c,s=torch.cos(angle),torch.sin(angle)
    return torch.stack((torch.stack((c,-s)),torch.stack((s,c))))


class Stitcher(nn.Module):
    def __init__(self,mode,seed):
        super().__init__();self.mode=mode
        g=torch.Generator().manual_seed(seed)
        if mode=="independent":
            self.weights=nn.Parameter(torch.randn(2,DIM,generator=g)*.02)
        else:
            self.a=nn.Parameter(torch.randn(RANK,DIM,generator=g)/math.sqrt(DIM))
            self.b=nn.Parameter(torch.zeros(1,RANK))
            if mode=="mirror":self.codes=nn.Parameter(torch.zeros(2))
            if mode=="film":self.codes=nn.Parameter(torch.ones(2,RANK))

    def forward(self,x,task):
        if self.mode=="independent":return (x*self.weights[task]).sum(-1)
        z=F.linear(x,self.a)
        if self.mode=="mirror":
            r=torch.stack([givens(a) for a in self.codes])
            z=torch.einsum('bij,bj->bi',r[task],z)
        elif self.mode=="film":z=z*self.codes[task]
        return F.linear(z,self.b).squeeze(-1)

    def active_macs(self):
        if self.mode=="independent":return DIM
        return RANK*DIM + RANK + (4 if self.mode=="mirror" else RANK if self.mode=="film" else 0)


def save_model(path,model,rep,seed):
    payload={"schema":"MA-932/linear-stitch-v1","seed":seed,"representation":rep,
      "mode":model.mode,"input_dim":DIM,"rank":RANK,"tasks":["u","v"],
      "state_dict":{k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}}
    torch.save(payload,path);raw=path.read_bytes()
    return {"path":path.name,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}


def ridge_probe(x,targets):
    # Closed-form ridge probe fit on the declared train split and evaluated on holdout.
    xtx=x.T@x + 1e-4*torch.eye(x.shape[1]);coef=torch.linalg.solve(xtx,x.T@targets)
    return ((x@coef-targets)**2).mean(0)
