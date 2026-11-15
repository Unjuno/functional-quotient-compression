# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A11 multi-family meta-learning capacity screen. Teacher parameters never enter model init.

All tensors and outputs are dimensionless. Weight-file byte accounting includes
all learned FP32 tensors and a 64-byte header/trailer (NOT executable decoder).
"""
from __future__ import annotations
import hashlib
import math
import struct
import time
from typing import Dict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

K_TRAIN, K_NEW, RANK = 16, 8, 2
METHODS = ('mirror','pre','shift','film','basis','multihead')
CAP_OPTIONS = (512,640,768,1024)
MAGIC=b'A11META\0'
HEADER=struct.Struct('<8s6I')
assert HEADER.size==32 and len(MAGIC)==8


FAMILIES = ('ramp','tent','sine')


def make_world(world_id:int,family:str)->dict:
    """Generate labels ONLY from an external synthetic teacher; no Mirror in teacher.

    World and family identify a task distribution. Train/support/dev/IID/OOD
    use independent RNG streams, and teacher parameters are never passed to models.
    """
    if type(world_id) is not int or world_id < 0 or family not in FAMILIES:
        raise ValueError('invalid world id or teacher family')
    tag=FAMILIES.index(family)
    rng=np.random.default_rng(417211+world_id*19937+tag*170981)
    count=K_TRAIN+K_NEW
    if family=='ramp':
        a=rng.uniform(-.95,.55,count).astype('float32')
        b=(a+rng.uniform(.24,.52,count)).astype('float32')
        def target(x):
            return np.maximum(x[...,None]-a,0)-np.maximum(x[...,None]-b,0)
        teacher=np.stack((a,b),1)
    elif family=='tent':
        center=rng.uniform(-.65,.65,count).astype('float32')
        width=rng.uniform(.20,.46,count).astype('float32')
        def target(x):
            return (np.maximum(x[...,None]-(center-width),0)
                    -2*np.maximum(x[...,None]-center,0)
                    +np.maximum(x[...,None]-(center+width),0))/width
        teacher=np.stack((center,width),1)
    else:
        freq=rng.uniform(.75,2.5,count).astype('float32')
        phase=rng.uniform(-np.pi,np.pi,count).astype('float32')
        scale=rng.uniform(.65,1.15,count).astype('float32')
        def target(x):
            return scale*np.sin(freq*x[...,None]+phase)
        teacher=np.stack((freq,phase,scale),1)
    def rand_x(n,stream,lo,hi):
        return np.random.default_rng(world_id*117+tag*70001+stream).uniform(lo,hi,(n,1)).astype('float32')
    tx=rand_x(1024,11,-1.5,1.5)
    dev=rand_x(256,23,-1.5,1.5)
    iid=rand_x(256,37,-1.5,1.5)
    ood=rand_x(256,59,-1.9,1.9)
    support=np.stack([np.random.default_rng(world_id*10003+tag*2999+800+t).uniform(-1.5,1.5,(32,1)).astype('float32') for t in range(K_NEW)])
    # Evaluate all teacher functions and select held-out task indices afterwards.
    fy=lambda xx:target(xx).squeeze(axis=-2).astype('float32')
    meta_y=fy(tx)[:,:K_TRAIN];dev_y=fy(dev)[:,:K_TRAIN]
    iid_y=fy(iid)[:,K_TRAIN:];ood_y=fy(ood)[:,K_TRAIN:]
    support_y=np.stack([fy(support[t])[:,K_TRAIN+t] for t in range(K_NEW)])
    payload=teacher.tobytes()+b''.join(a.tobytes() for a in (tx,meta_y,dev,dev_y,support,support_y,iid,iid_y,ood,ood_y))
    return dict(family=family,world_id=world_id,fingerprint=hashlib.sha256(payload).hexdigest(),
        meta_x=torch.from_numpy(tx),train_y=torch.from_numpy(meta_y),
        meta_dev_x=torch.from_numpy(dev),meta_dev_y=torch.from_numpy(dev_y),
        support_x=torch.from_numpy(support),support_y=torch.from_numpy(support_y),
        new_iid_x=torch.from_numpy(iid),new_iid_y=torch.from_numpy(iid_y),
        new_ood_x=torch.from_numpy(ood),new_ood_y=torch.from_numpy(ood_y))

def nparams(method:str,h:int)->int:
    if method not in METHODS or type(h)!=int or h<1 or (method in ('mirror','pre') and h%2):
        raise ValueError('invalid model width')
    if method=='multihead':return 26*h+24
    return (3*h+1+(K_TRAIN+K_NEW)*RANK + (4*h if method=='film' else 2*h if method in ('shift','basis') else 0))


class Student(nn.Module):
    def __init__(self,method:str,width:int):
        super().__init__()
        nparams(method,width)
        self.method,self.width=method,width
        self.up=nn.Linear(1,width)
        if method=='multihead':
            self.head_train=nn.Linear(width,K_TRAIN)
            self.head_new=nn.Linear(width,K_NEW)
        else:
            self.down=nn.Linear(width,1)
            self.train_codes=nn.Parameter(torch.empty(K_TRAIN,RANK).uniform_(-.4,.4))
            self.new_codes=nn.Parameter(torch.empty(K_NEW,RANK).uniform_(-.4,.4))
            if method in ('shift','film'):
                self.shift_w=nn.Parameter(torch.randn(RANK,width)*.1)
            if method=='film':self.gain_w=nn.Parameter(torch.randn(RANK,width)*.1)
            if method=='basis':self.basis_w=nn.Parameter(torch.randn(RANK,width)*.1)
        assert sum(p.numel() for p in self.parameters())==nparams(method,width)

    def forward(self,x:torch.Tensor,ids:torch.Tensor)->torch.Tensor:
        if x.ndim!=2 or x.shape[1]!=1 or ids.shape!=(len(x),) or ids.dtype!=torch.long:
            raise ValueError('x=[batch,1], ids=[batch] long')
        if ids.numel()==0 or ids.min()<0 or ids.max()>=K_TRAIN+K_NEW:
            raise ValueError('task address out of range')
        v=self.up(x)
        if self.method=='multihead':
            h=F.relu(v)
            return torch.cat((self.head_train(h),self.head_new(h)),dim=-1).gather(1,ids[:,None]).flatten()
        code=torch.cat((self.train_codes,self.new_codes),dim=0)[ids]
        if self.method in ('mirror','pre'):
            pair=v.reshape(-1,self.width//2,2)
            a=code[:,torch.arange(self.width//2,device=x.device)%RANK]
            z0=F.relu(pair[:,:,0]+a*pair[:,:,1])
            z1=F.relu(pair[:,:,1])
            if self.method=='mirror': z0=z0-a*z1
            h=torch.stack((z0,z1),-1).reshape(-1,self.width)
        else:
            if self.method in ('shift','film'):v=v+code@self.shift_w
            if self.method=='film':v=v*(1.+code@self.gain_w)
            h=F.relu(v)
        y=self.down(h).flatten()
        if self.method=='basis':y=y+(code*(h@self.basis_w.T)).sum(-1)
        return y

    def enter_meta(self):
        for p in self.parameters():p.requires_grad_(True)
        if self.method=='multihead':
            for p in self.head_new.parameters():p.requires_grad_(False)
        else:self.new_codes.requires_grad_(False)

    def enter_adaptation(self):
        for p in self.parameters():p.requires_grad_(False)
        if self.method=='multihead':
            for p in self.head_new.parameters():p.requires_grad_(True)
        else:self.new_codes.requires_grad_(True)


def make_model(method:str,cap:int,seed:int)->Student:
    if method not in METHODS or type(cap)!=int or cap not in CAP_OPTIONS or type(seed)!=int or seed<0:
        raise ValueError('unsupported model or capacity')
    choices=[h for h in range(1,200) if (method not in ('mirror','pre') or h%2==0)
             and 64+4*nparams(method,h)<=cap]
    if not choices:raise ValueError('budget too small')
    torch.manual_seed(seed)
    return Student(method,max(choices))


def count_bytes(m:Student)->int:
    return 64+4*sum(p.numel() for p in m.parameters())


def serialize(m:Student)->bytes:
    if any(p.dtype!=torch.float32 or p.device.type!='cpu' or not bool(torch.isfinite(p).all()) for p in m.parameters()):
        raise ValueError('CPU finite FP32 only')
    head=HEADER.pack(MAGIC,1,METHODS.index(m.method),m.width,K_TRAIN,K_NEW,RANK)
    payload=b''.join(p.detach().contiguous().numpy().astype('<f4',copy=False).tobytes() for p in m.parameters())
    data=head+payload
    assert len(data)+32==count_bytes(m)
    return data+hashlib.sha256(data).digest()


def deserialize(raw:bytes)->Student:
    if len(raw)<64 or hashlib.sha256(raw[:-32]).digest()!=raw[-32:]:
        raise ValueError('digest mismatch')
    magic,version,kind,width,ntrain,nnew,rank=HEADER.unpack(raw[:32])
    if magic!=MAGIC or version!=1 or kind>=len(METHODS) or (ntrain,nnew,rank)!=(K_TRAIN,K_NEW,RANK):
        raise ValueError('bad header')
    m=Student(METHODS[kind],width)
    if count_bytes(m)!=len(raw):raise ValueError('wrong payload length')
    values=np.frombuffer(raw[32:-32],dtype='<f4').copy()
    i=0
    with torch.no_grad():
        for p in m.parameters():
            n=p.numel()
            p.copy_(torch.from_numpy(values[i:i+n].reshape(p.shape)))
            i+=n
    assert i==len(values)
    return m.eval()


def metric(pred:torch.Tensor,target:torch.Tensor)->dict:
    if pred.shape!=target.shape or target.ndim!=2:
        raise ValueError('evaluation shapes mismatch')
    a=pred.detach().to(torch.float64)-target.detach().to(torch.float64)
    y=target.detach().to(torch.float64)
    mse=float(a.square().mean())
    variance=float((y-y.mean(0,keepdim=True)).square().mean())
    if variance<=0:raise ValueError('zero target variance')
    return {'mse':mse,'nrmse':math.sqrt(mse/variance),'target_variance':variance}


def evaluate(m:Student,x:torch.Tensor,y:torch.Tensor,offset:int)->dict:
    k=y.shape[1]
    with torch.no_grad():
        pred=m.eval()(x.repeat(k,1),torch.arange(offset,offset+k,dtype=torch.long).repeat_interleave(len(x))).reshape(k,-1).T
    return metric(pred,y)


def fit_two_stage(world:dict,method:str,cap:int,seed:int, *, meta_steps:int=1000, adapt_steps:int=300,batch:int=128)->dict:
    if meta_steps<1 or adapt_steps<1 or batch<1:raise ValueError('invalid optimizer schedule')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model=make_model(method,cap,seed=world['world_id']*1009+seed*71+method_free_seed(method))
    model.enter_meta()
    # Match minibatch *indices* between methods, but method initialization is separate.
    gen=torch.Generator().manual_seed(630001+world['world_id']*107+seed*347)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.01,weight_decay=0)
    start=time.perf_counter()
    x,y=world['meta_x'],world['train_y']
    for step in range(meta_steps):
        ix=torch.randint(len(x),(batch,),generator=gen)
        task=torch.randint(K_TRAIN,(batch,),generator=gen)
        opt.zero_grad(set_to_none=True)
        loss=(model(x[ix],task)-y[ix,task]).square().mean()
        if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite meta loss')
        loss.backward()
        gn=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),5.)
        if not bool(torch.isfinite(gn)):raise RuntimeError('nonfinite meta gradients')
        opt.step()
    meta_seconds=time.perf_counter()-start
    meta_final_batch_mse=float(loss.detach())
    # Snapshot and then hard-freeze all shared parameter values.
    model.enter_adaptation()
    frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
    adap_optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.01,weight_decay=0)
    gx=torch.Generator().manual_seed(740001+world['world_id']*107+seed*347)
    sx,sy=world['support_x'],world['support_y']
    start=time.perf_counter()
    for step in range(adapt_steps):
        task=torch.randint(K_NEW,(batch,),generator=gx)
        ix=torch.randint(32,(batch,),generator=gx)
        adap_optimizer.zero_grad(set_to_none=True)
        loss=(model(sx[task,ix],task+K_TRAIN)-sy[task,ix]).square().mean()
        if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite adaptation loss')
        loss.backward()
        gn=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),5.)
        if not bool(torch.isfinite(gn)):raise RuntimeError('nonfinite adaptation gradients')
        adap_optimizer.step()
    adapt_seconds=time.perf_counter()-start
    if any(not torch.equal(p.detach(),frozen[name]) for name,p in model.named_parameters() if name in frozen):
        raise AssertionError('adaptation changed shared weights')
    model.eval()
    return dict(model=model,meta_seconds=meta_seconds,adapt_seconds=adapt_seconds,
                adaptation_trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                meta_final_batch_mse=meta_final_batch_mse, # training minibatch only; no audit access
                adapt_final_batch_mse=float(loss.detach()))


def method_free_seed(method:str)->int:
    return METHODS.index(method)*1409+19
