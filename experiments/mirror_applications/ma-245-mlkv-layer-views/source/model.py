"""Cross-layer K/V sharing, cache views, and attention-output regression models."""
from __future__ import annotations

import io,json
from dataclasses import asdict,dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    layers:int=4
    model_dim:int=16
    head_dim:int=4
    memory_tokens:int=8


def rotate(x:torch.Tensor,angle:torch.Tensor):
    y=x.clone();c,s=torch.cos(angle),torch.sin(angle);a,b=x[...,0],x[...,1]
    y[...,0]=c*a-s*b;y[...,1]=s*a+c*b
    return y


class LayerKV(nn.Module):
    METHODS=("mha","mlkv2","mlkv1","mirror","gate")
    def __init__(self,cfg:Config,method:str):
        super().__init__()
        if method not in self.METHODS:raise ValueError(method)
        self.cfg,self.method=cfg,method
        self.groups={"mha":cfg.layers,"mlkv2":2,"mlkv1":1,"mirror":1,"gate":1}[method]
        self.wk=nn.Parameter(torch.empty(self.groups,cfg.model_dim,cfg.head_dim))
        self.wv=nn.Parameter(torch.empty(self.groups,cfg.model_dim,cfg.head_dim))
        nn.init.xavier_uniform_(self.wk);nn.init.xavier_uniform_(self.wv)
        if method=="mirror":self.angles=nn.Parameter(torch.zeros(cfg.layers,2))
        if method=="gate":self.gates=nn.Parameter(torch.ones(cfg.layers,2))

    def materialized_cache(self,mem:torch.Tensor):
        return (torch.einsum("btd,gdf->bgtf",mem,self.wk),torch.einsum("btd,gdf->bgtf",mem,self.wv))

    def measured_cache_bytes(self,mem:torch.Tensor):
        return sum(t.numel()*t.element_size() for t in self.materialized_cache(mem))

    def cache_bytes(self,tokens:int|None=None):
        n=tokens or self.cfg.memory_tokens
        return self.groups*2*n*self.cfg.head_dim*4

    def forward(self,query:torch.Tensor,mem:torch.Tensor):
        # Project/cache once per physical MLKV group, then expose layer views.
        key_cache,value_cache=self.materialized_cache(mem)
        outs=[]
        for layer in range(self.cfg.layers):
            group=layer//(self.cfg.layers//self.groups)
            key,value=key_cache[:,group],value_cache[:,group]
            if self.method=="mirror":
                key=rotate(key,self.angles[layer,0]);value=rotate(value,self.angles[layer,1])
            elif self.method=="gate":
                key=key*self.gates[layer,0];value=value*self.gates[layer,1]
            score=(query[:,layer,None,:]*key).sum(-1)/(self.cfg.head_dim**0.5)
            outs.append((score.softmax(-1).unsqueeze(-1)*value).sum(1))
        return torch.stack(outs,dim=1)

    def parameter_count(self):return sum(p.numel() for p in self.parameters())

    def inference_payload_bytes(self):
        meta=json.dumps({**asdict(self.cfg),"method":self.method},sort_keys=True,separators=(",",":")).encode()
        buf=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in self.state_dict().items()},buf)
        return len(meta)+len(buf.getvalue())


class Teacher(nn.Module):
    """One K/V pair, transformed into distinct layer-specific role views."""
    def __init__(self,cfg:Config,seed:int):
        super().__init__();g=torch.Generator().manual_seed(seed);self.cfg=cfg
        self.register_buffer("wk",torch.empty(cfg.model_dim,cfg.head_dim).normal_(0,0.25,generator=g))
        self.register_buffer("wv",torch.empty(cfg.model_dim,cfg.head_dim).normal_(0,0.25,generator=g))
        self.register_buffer("angles",torch.empty(cfg.layers,2).uniform_(-0.8,0.8,generator=g))
    @torch.no_grad()
    def forward(self,query,mem):
        k0,v0=mem@self.wk,mem@self.wv;out=[]
        for l in range(self.cfg.layers):
            k,v=rotate(k0,self.angles[l,0]),rotate(v0,self.angles[l,1])
            score=(query[:,l,None,:]*k).sum(-1)/(self.cfg.head_dim**0.5)
            out.append((score.softmax(-1).unsqueeze(-1)*v).sum(1))
        return torch.stack(out,dim=1)


def mac_proxy_per_example(cfg:Config,method:str):
    groups={"mha":cfg.layers,"mlkv2":2,"mlkv1":1,"mirror":1,"gate":1}[method]
    projection=cfg.memory_tokens*cfg.model_dim*cfg.head_dim*2*groups
    attention=cfg.layers*cfg.memory_tokens*cfg.head_dim*2
    if method=="mirror":projection+=cfg.layers*2*cfg.memory_tokens*8
    if method=="gate":projection+=cfg.layers*2*cfg.memory_tokens*cfg.head_dim
    return projection+attention
