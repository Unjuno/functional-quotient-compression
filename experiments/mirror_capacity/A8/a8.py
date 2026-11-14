# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A8: rank-certified 1D multi-function representation benchmark.

All x/y, code and weights are dimensionless. Stored weights are little-endian
FP32 and exact byte accounting includes every learned tensor, 32-B header,
and 32-B SHA-256. Program bytes and optimizer states are not included.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

K = 16
RANK = 2
KINDS = ('mirror', 'pre', 'film', 'shift', 'input_affine', 'basis', 'multihead')
MAGIC = b'A8CAP\0\0\0'
HEADER = struct.Struct('<8s6I')
assert len(MAGIC) == 8 and HEADER.size == 32


def make_world(world: int, family: str) -> dict:
    if type(world) is not int or world < 0 or family not in ('hinge','ramp'):
        raise ValueError('world or family invalid')
    rng = np.random.default_rng(810012 + world * 9973)
    thresholds = np.linspace(-.95, .75, K, dtype=np.float64) + rng.uniform(-.015,.015,K)
    thresholds = np.sort(thresholds).astype(np.float32)
    right = thresholds + rng.uniform(.20,.55,K).astype(np.float32)

    def make_split(n:int,seed:int,lower:float,upper:float) -> tuple[torch.Tensor,torch.Tensor]:
        x = np.random.default_rng(seed).uniform(lower,upper,(n,1)).astype(np.float32)
        y = np.maximum(x - thresholds[None,:],0.)
        if family=='ramp': y -= np.maximum(x-right[None,:],0.)
        return torch.tensor(x.copy()),torch.tensor(y.astype(np.float32,copy=True))

    samples={}
    for offset,name,n,lo,hi in [(11,'train',1024,-1.5,1.5),(23,'dev',384,-1.5,1.5),
                                 (37,'iid',384,-1.5,1.5),(59,'ood',384,-2.,2.)]:
        samples[name]=make_split(n,world*100003 + offset + (0 if family=='hinge' else 10000019),lo,hi)
    data = thresholds.tobytes() + right.tobytes()+family.encode('ascii')
    for name in ('train','dev','iid','ood'):
        data += samples[name][0].numpy().tobytes() + samples[name][1].numpy().tobytes()
    samples.update(world=world,family=family,thresholds=thresholds,right=right,
                   fingerprint=hashlib.sha256(data).hexdigest())
    return samples


def num_params(kind: str, width: int) -> int:
    if kind not in KINDS or width < 1 or (kind in ('mirror','pre') and (width < 2 or width % 2)):
        raise ValueError('invalid kind or width')
    if kind=='multihead': return (K+2)*width+K
    base=3*width + 1 + K*RANK
    return base + (4*width if kind=='film' else 2*width if kind in ('shift','basis') else 0)


class Student(nn.Module):
    def __init__(self,kind: str,width: int):
        super().__init__()
        if kind not in KINDS or width < 1 or (kind in ('mirror','pre') and (width<2 or width%2)):
            raise ValueError('invalid kind or width')
        self.kind,self.width=kind,width
        self.up=nn.Linear(1,width)
        if kind=='multihead':
            self.head=nn.Linear(width,K)
        else:
            self.down=nn.Linear(width,1)
            self.code=nn.Embedding(K,RANK)
            nn.init.uniform_(self.code.weight,-.4,.4)
            if kind in ('shift','film'): self.shift_w=nn.Parameter(torch.randn(RANK,width)*.1)
            if kind=='film': self.gain_w=nn.Parameter(torch.randn(RANK,width)*.1)
            if kind=='basis': self.basis_w=nn.Parameter(torch.randn(RANK,width)*.1)
        assert sum(p.numel() for p in self.parameters())==num_params(kind,width)

    def forward(self,x:torch.Tensor,task:torch.Tensor)->torch.Tensor:
        if x.ndim!=2 or x.shape[1]!=1 or task.shape!=(len(x),) or task.dtype!=torch.long:
            raise ValueError('expected x=[batch,1], task=[batch] int64')
        if task.min()<0 or task.max()>=K: raise ValueError('task out of range')
        if self.kind=='multihead':
            return self.head(F.relu(self.up(x))).gather(1,task[:,None]).flatten()
        code=self.code(task)
        if self.kind=='input_affine':
            x=x*(1.+code[:,0:1])+code[:,1:2]
        v=self.up(x)
        if self.kind in ('mirror','pre'):
            ab=code[:,torch.arange(self.width//2,device=x.device)%RANK]
            v=v.reshape(-1,self.width//2,2)
            z0=F.relu(v[:,:,0]+ab*v[:,:,1]); z1=F.relu(v[:,:,1])
            if self.kind=='mirror':z0=z0-ab*z1
            h=torch.stack((z0,z1),-1).reshape(-1,self.width)
        else:
            if self.kind in ('shift','film'):v=v+code@self.shift_w
            if self.kind=='film':v=v*(1.+code@self.gain_w)
            h=F.relu(v)
        y=self.down(h).flatten()
        if self.kind=='basis':y=y+(code*(h@self.basis_w.T)).sum(-1)
        return y


def create(kind: str, cap: int) -> Student:
    if kind not in KINDS or type(cap) is not int or cap<64:
        raise ValueError('invalid kind or cap')
    candidates=[h for h in range(1,300) if (kind not in ('mirror','pre') or h%2==0)
                and (kind not in ('mirror','pre') or h>=2) and 64 + 4*num_params(kind,h)<=cap] if cap>=200 else []
    if not candidates: raise ValueError('budget too small')
    return Student(kind,max(candidates))


def serialize(m:Student)->bytes:
    if any(p.dtype!=torch.float32 or p.device.type!='cpu' or not bool(torch.isfinite(p).all())
           for p in m.parameters()):raise ValueError('expected finite CPU FP32 parameters')
    payload=b''.join(p.detach().contiguous().numpy().astype('<f4',copy=False).tobytes()
                     for p in m.parameters())
    header=HEADER.pack(MAGIC,1,KINDS.index(m.kind),m.width,1,1,K)
    raw=header+payload
    assert len(raw)==32+4*num_params(m.kind,m.width)
    return raw+hashlib.sha256(raw).digest()


def deserialize(raw:bytes)->Student:
    if len(raw)<64 or hashlib.sha256(raw[:-32]).digest()!=raw[-32:]:
        raise ValueError('bad model length/checksum')
    magic,vers,kind_id,h,d,o,k=HEADER.unpack(raw[:32])
    if magic!=MAGIC or vers!=1 or kind_id>=len(KINDS) or d!=1 or o!=1 or k!=K:
        raise ValueError('bad model header')
    m=Student(KINDS[kind_id],h)
    if len(raw)!=64+4*num_params(m.kind,m.width):raise ValueError('bad payload length')
    vals=np.frombuffer(raw[32:-32],dtype='<f4')
    if not np.isfinite(vals).all():raise ValueError('nonfinite parameters')
    start=0
    with torch.no_grad():
        for p in m.parameters():
            n=p.numel();p.copy_(torch.from_numpy(vals[start:start+n].copy()).reshape_as(p));start+=n
    assert start==len(vals)
    return m


def exact_construction(kind:str, world:dict)->Student:
    if kind not in ('mirror','pre'):raise ValueError('only mirror and pre')
    h=2 if world['family']=='hinge' else 4
    m=Student(kind,h)
    with torch.no_grad():
        m.up.weight.zero_();m.up.bias.zero_()
        m.down.weight.zero_();m.down.bias.zero_()
        for q in range(h//2):
            m.up.weight[2*q,0]=1.
            m.up.bias[2*q+1]=-1.  # inactive ReLU branch, inverse does not matter
            m.down.weight[0,2*q]=1. if q==0 else -1.
        m.code.weight[:,0]=torch.tensor(world['thresholds'])
        m.code.weight[:,1]=torch.tensor(world['right'])
    return m


def rank_certificate(world:dict,width:int) -> dict:
    if not 0<=width<K:raise ValueError('width must be in [0,K)')
    t=world['thresholds'].astype(np.float64)
    right=world['right'].astype(np.float64)
    eps=.75*float(np.diff(t).min())
    points=np.r_[float(t.min())-2*eps,t+eps]
    y=np.maximum(points[:,None]-t[None,:],0.)
    if world['family']=='ramp':y-=np.maximum(points[:,None]-right[None,:],0.)
    yc=y-y.mean(axis=0,keepdims=True)
    singular=np.linalg.svd(yc,compute_uv=False)
    numerical_rank=int(np.count_nonzero(singular>singular[0]*1e-10))
    floor=float((singular[width:]**2).sum()/yc.size)
    return {'width':width,'points':points,'centered_matrix':yc,'singular_values':singular,
            'target_rank_centered':numerical_rank,'mse_floor':floor,
            'relative_frobenius_floor':float(np.sqrt(np.sum(singular[width:]**2)/np.sum(singular**2)))}


def quality(m:Student,split:tuple[torch.Tensor,torch.Tensor])->dict:
    xx,yy=split
    if xx.ndim!=2 or xx.shape[1]!=1 or yy.shape!=(len(xx),K):raise ValueError('data shape')
    with torch.no_grad():
        inp=xx.repeat(K,1)
        ids=torch.arange(K,dtype=torch.long).repeat_interleave(len(xx))
        pred=m.eval()(inp,ids).reshape(K,-1).T.double()
        target=yy.double()
        dif=pred-target
        mse=float(dif.square().mean())
        denom=float((target-target.mean(dim=0,keepdim=True)).square().mean())
        if denom<=0:raise ValueError('invalid denominator')
        return {'mse':mse,'nrmse':float(np.sqrt(mse/denom)),
                'max_abs':float(dif.abs().max()),'target_var':denom}


def train_one(world:dict,kind:str,cap:int,seed:int,steps:int=600,batch:int=256)->dict:
    if steps<=0 or batch<=0:raise ValueError('steps/batch')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(510000+world['world']*199+seed*797)
    m=create(kind,cap)
    xx,yy=world['train']
    batchgen=torch.Generator().manual_seed(951000+world['world']*491+seed*37)
    opt=torch.optim.AdamW(m.parameters(),lr=.012,weight_decay=0.)
    m.train()
    for s in range(steps):
        ix=torch.randint(len(xx),(batch,),generator=batchgen)
        ids=torch.randint(K,(batch,),generator=batchgen)
        opt.zero_grad(set_to_none=True)
        output=m(xx[ix],ids)
        loss=(output-yy[ix,ids]).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(m.parameters(),5.)
        if not torch.isfinite(grad):raise RuntimeError('nonfinite gradients')
        opt.step()
    m.eval()
    return {'model':m,'train_final_mse':float(loss.item())}
