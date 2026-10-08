"""Recursive tied-block, step-LoRA, hyper-LoRA, and Mirror-view models."""
from __future__ import annotations
import io,json
from dataclasses import asdict,dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    input_dim:int=8
    hidden_dim:int=16
    depth:int=4
    rank:int=1


def rotate(x:torch.Tensor,angle:torch.Tensor,inverse:bool=False):
    y=x.clone();s=torch.sin(angle)*(-1 if inverse else 1);c=torch.cos(angle);a,b=x[...,0],x[...,1]
    y[...,0]=c*a-s*b;y[...,1]=s*a+c*b
    return y


class RecurrentModel(nn.Module):
    METHODS=("untied","tied","scalar_gate","static_lora","hyper_lora","mirror")
    def __init__(self,cfg:Config,method:str):
        super().__init__()
        if method not in self.METHODS:raise ValueError(method)
        self.cfg,self.method=cfg,method
        if method=="untied":
            self.w1=nn.Parameter(torch.empty(cfg.depth,cfg.input_dim,cfg.hidden_dim))
            self.w2=nn.Parameter(torch.empty(cfg.depth,cfg.hidden_dim,cfg.input_dim))
        else:
            self.w1=nn.Parameter(torch.empty(cfg.input_dim,cfg.hidden_dim))
            self.w2=nn.Parameter(torch.empty(cfg.hidden_dim,cfg.input_dim))
        for p in (self.w1,self.w2):nn.init.xavier_uniform_(p)
        # A shared residual gate is present for every recurrence variant.
        self.residual_gate_logits=nn.Parameter(torch.zeros(cfg.depth))
        if method=="scalar_gate":self.step_scale_logits=nn.Parameter(torch.zeros(cfg.depth))
        if method=="mirror":self.angles=nn.Parameter(torch.zeros(cfg.depth))
        if method=="static_lora":
            self.lora_a=nn.Parameter(torch.randn(cfg.depth,cfg.hidden_dim,cfg.rank)*0.02)
            self.lora_b=nn.Parameter(torch.zeros(cfg.depth,cfg.rank,cfg.input_dim))
        if method=="hyper_lora":
            self.basis_a=nn.Parameter(torch.randn(cfg.hidden_dim,cfg.rank)*0.02)
            self.basis_b=nn.Parameter(torch.zeros(cfg.rank,cfg.input_dim))
            self.controller_x=nn.Parameter(torch.randn(cfg.input_dim,cfg.rank)*0.02)
            self.controller_step=nn.Parameter(torch.zeros(cfg.depth,cfg.rank))
            self.controller_bias=nn.Parameter(torch.zeros(cfg.rank))

    def forward(self,x:torch.Tensor):
        h=x
        for step in range(self.cfg.depth):
            if self.method=="untied":w1,w2=self.w1[step],self.w2[step]
            else:w1,w2=self.w1,self.w2
            source=rotate(h,self.angles[step]) if self.method=="mirror" else h
            z=F.gelu(source@w1)
            block=z@w2
            if self.method=="mirror":block=rotate(block,self.angles[step],inverse=True)
            elif self.method=="scalar_gate":block=block*torch.sigmoid(self.step_scale_logits[step])
            elif self.method=="static_lora":block=block+(z@self.lora_a[step])@self.lora_b[step]
            elif self.method=="hyper_lora":
                coeff=torch.tanh(h@self.controller_x+self.controller_step[step]+self.controller_bias)
                block=block+(z@self.basis_a)@self.basis_b*coeff
            h=h+torch.sigmoid(self.residual_gate_logits[step])*block
        return h

    def parameter_count(self):return sum(p.numel() for p in self.parameters())

    def inference_payload_bytes(self):
        meta=json.dumps({**asdict(self.cfg),"method":self.method},sort_keys=True,separators=(",",":")).encode()
        buf=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in self.state_dict().items()},buf)
        return len(meta)+len(buf.getvalue())


class Teacher(nn.Module):
    def __init__(self,cfg:Config,seed:int):
        super().__init__();g=torch.Generator().manual_seed(seed);self.cfg=cfg
        self.register_buffer("w1",torch.empty(cfg.input_dim,cfg.hidden_dim).normal_(0,0.18,generator=g))
        self.register_buffer("w2",torch.empty(cfg.hidden_dim,cfg.input_dim).normal_(0,0.18,generator=g))
        self.register_buffer("angles",torch.empty(cfg.depth).uniform_(-0.75,0.75,generator=g))
        self.register_buffer("gates",torch.full((cfg.depth,),0.5))
    @torch.no_grad()
    def forward(self,x):
        h=x
        for step in range(self.cfg.depth):
            source=rotate(h,self.angles[step]);z=F.gelu(source@self.w1)
            block=rotate(z@self.w2,self.angles[step],inverse=True)
            h=h+self.gates[step]*block
        return h


def mac_proxy_per_example(cfg:Config,method:str):
    macs=cfg.depth*cfg.input_dim*cfg.hidden_dim+cfg.depth*cfg.hidden_dim*cfg.input_dim
    if method=="mirror":macs+=cfg.depth*16
    elif method=="scalar_gate":macs+=cfg.depth*cfg.input_dim
    elif method=="static_lora":macs+=cfg.depth*cfg.rank*(cfg.hidden_dim+cfg.input_dim)
    elif method=="hyper_lora":macs+=cfg.depth*(cfg.hidden_dim*cfg.rank+cfg.rank*cfg.input_dim+cfg.input_dim*cfg.rank)
    return macs
