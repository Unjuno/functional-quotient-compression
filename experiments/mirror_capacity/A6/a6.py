# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A6: conditional-function capacity on the bundled handwritten digits dataset.

Only condition-indexed students are trainable. No model contains a Mirror teacher.
FP32 tensor payload is strictly size-accounted, with a SHA-256 trailer.
"""
from __future__ import annotations
import hashlib
import math
import struct

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F

D, O, TASKS, RANK = 64, 1, 16, 4
KINDS = ('mirror4', 'pre4', 'film4', 'basis4', 'lowrank4', 'dense4', 'multihead')
MAGIC = b'A6CAP\0\0\0'
HEADER = struct.Struct('<8s6I')
assert len(MAGIC) == HEADER.size - 6*4 == 8


def param_count(kind: str, width: int) -> int:
    if kind not in KINDS or width < 2 or (kind in ('mirror4', 'pre4') and width % 2):
        raise ValueError('invalid model kind/width')
    if kind == 'multihead':
        return (D + 1) * width + TASKS * (width + 1)
    shared = (D + 1) * width + (width + 1) + TASKS * RANK
    extra = {
        'mirror4': 0, 'pre4': 0,
        'film4': 2 * RANK * width,
        'basis4': RANK * width,
        'lowrank4': RANK * (D + width),
        'dense4': RANK * width,
    }[kind]
    return shared + extra


class Student(nn.Module):
    def __init__(self, kind: str, width: int):
        super().__init__()
        if kind not in KINDS or width < 2 or (kind in ('mirror4', 'pre4') and width % 2):
            raise ValueError('invalid kind/width')
        self.kind, self.width = kind, width
        if kind == 'multihead':
            self.up = nn.Linear(D, width)
            self.head = nn.Linear(width, TASKS)
        else:
            self.up = nn.Linear(D + (RANK if kind == 'dense4' else 0), width)
            self.down = nn.Linear(width, O)
            self.code = nn.Embedding(TASKS, RANK)
            nn.init.uniform_(self.code.weight, -0.5, 0.5)
            if kind == 'film4':
                self.gain = nn.Parameter(torch.randn(RANK, width) * 0.18)
                self.shift = nn.Parameter(torch.randn(RANK, width) * 0.18)
            elif kind == 'basis4':
                self.basis = nn.Parameter(torch.randn(width, RANK) * (0.2/math.sqrt(width)))
            elif kind == 'lowrank4':
                self.lora_in = nn.Parameter(torch.randn(RANK, D) * 0.1)
                self.lora_out = nn.Parameter(torch.randn(RANK, width) * 0.1)
        assert sum(x.numel() for x in self.parameters()) == param_count(kind, width)

    def forward(self, x: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != D or task.shape != (len(x),) or task.dtype != torch.long:
            raise ValueError('expected x=[batch,64], task=[batch] int64')
        if self.kind == 'multihead':
            h = F.gelu(self.up(x))
            return self.head(h).gather(1, task[:, None]).squeeze(1)
        code = self.code(task)
        v = self.up(torch.cat((x, code), -1) if self.kind == 'dense4' else x)
        if self.kind in ('mirror4', 'pre4'):
            paired = v.reshape(-1, self.width // 2, 2)
            a = code[:, torch.arange(self.width // 2, device=x.device) % RANK]
            z0 = F.gelu(paired[..., 0] + a * paired[..., 1])
            z1 = F.gelu(paired[..., 1])
            if self.kind == 'mirror4': z0 = z0 - a * z1
            h = torch.stack((z0,z1),dim=-1).reshape(-1,self.width)
        else:
            if self.kind == 'film4':
                v = v * (1 + code @ self.gain) + code @ self.shift
            elif self.kind == 'lowrank4':
                v = v + ((x @ self.lora_in.T) * code) @ self.lora_out
            h = F.gelu(v)
        result = self.down(h).flatten()
        if self.kind == 'basis4':
            result = result + (h @ self.basis * code).sum(-1)
        return result


def create(kind: str, cap: int) -> Student:
    if kind not in KINDS or cap < 128:
        raise ValueError('invalid kind or cap')
    step = 2 if kind in ('mirror4','pre4') else 1
    choices = [h for h in range(2, ((cap-64)//4)+1) if h % step == 0 and 64+4*param_count(kind,h) <= cap]
    if not choices:
        raise ValueError('insufficient bytes')
    return Student(kind,max(choices))


def model_bytes(model: Student) -> int:
    return HEADER.size + 4*sum(p.numel() for p in model.parameters()) + 32


def serialize(model: Student) -> bytes:
    if any(p.dtype != torch.float32 or p.device.type != 'cpu' or not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('only finite FP32 CPU weights')
    hdr = HEADER.pack(MAGIC,1,KINDS.index(model.kind),model.width,D,O,TASKS)
    raw = hdr + b''.join(p.detach().contiguous().numpy().astype('<f4',copy=False).tobytes() for p in model.parameters())
    assert len(raw)+32 == model_bytes(model)
    return raw + hashlib.sha256(raw).digest()


def deserialize(raw: bytes) -> Student:
    if len(raw) < 64 or hashlib.sha256(raw[:-32]).digest() != raw[-32:]:
        raise ValueError('invalid model length/digest')
    magic,vers,kind_id,width,d,o,tasks=HEADER.unpack(raw[:32])
    if magic != MAGIC or vers != 1 or kind_id >= len(KINDS) or d!=D or o!=O or tasks!=TASKS:
        raise ValueError('invalid model header')
    model=Student(KINDS[kind_id],width)
    if len(raw) != model_bytes(model): raise ValueError('invalid model payload length')
    values=np.frombuffer(raw[32:-32],dtype='<f4')
    if not np.isfinite(values).all(): raise ValueError('invalid model tensor')
    offset=0
    with torch.no_grad():
        for p in model.parameters():
            n=p.numel(); p.copy_(torch.from_numpy(values[offset:offset+n].copy()).reshape_as(p));offset+=n
    assert offset==len(values)
    return model


def explicit_shear(model: Student, x: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
    if model.kind not in ('mirror4','pre4'):raise ValueError('shear models only')
    out=[]
    for i in range(len(x)):
        code=model.code(task[i:i+1])[0]
        a=code[torch.arange(model.width//2) % RANK]
        w=model.up.weight.reshape(-1,2,D)
        b=model.up.bias.reshape(-1,2)
        wu=torch.stack((w[:,0] + a[:,None]*w[:,1],w[:,1]),1).reshape(-1,D)
        bu=torch.stack((b[:,0] + a*b[:,1],b[:,1]),1).reshape(-1)
        h=F.gelu(wu@x[i] + bu)
        wd=model.down.weight.reshape(1,-1,2)
        if model.kind=='mirror4': wd=torch.stack((wd[...,0],wd[...,1]-a[None,:]*wd[...,0]),-1)
        out.append((wd.reshape(1,-1)@h).flatten()[0]+model.down.bias[0])
    return torch.stack(out)


def make_world(world: int, family: str) -> dict:
    """Each world has unique masks and stratified IMAGE-disjoint train/val/test splits."""
    if world < 0 or family not in ('mask','xorink'):
        raise ValueError('world/family')
    sample=load_digits()
    x_np=np.asarray(sample.data,dtype=np.float32)/16.0
    targets=np.asarray(sample.target,dtype=np.int64)
    ids=np.arange(len(targets))
    train_ids,temp=train_test_split(ids,test_size=.4,stratify=targets,random_state=world*17+7)
    val_ids,test_ids=train_test_split(temp,test_size=.5,stratify=targets[temp],random_state=world*17+13)
    generator=np.random.default_rng(world*11027+13)
    masks=[]
    while len(masks)<TASKS:
        take=generator.choice(10,5,replace=False)
        mask=np.zeros(10,dtype=np.int8);mask[take]=1
        if not any(np.array_equal(mask,old) for old in masks): masks.append(mask)
    masks=np.stack(masks)
    threshold=float(np.median(x_np[train_ids].sum(axis=1)))
    def split(which,idx):
        base=x_np[idx]
        pos=masks[:,targets[idx]].copy()
        if family=='xorink':
            ink=(base.sum(axis=1)>threshold).astype(np.int8)
            pos=np.logical_xor(pos,ink[None,:]).astype(np.int8)
        if which=='ood':
            noise_rng=np.random.default_rng(900000 + world * 313)
            features=np.clip(base + noise_rng.normal(0,.15,size=base.shape).astype(np.float32),0,1)
        else:features=base
        return torch.tensor(features.copy()),torch.tensor(pos.copy(),dtype=torch.float32)
    outputs={s:split(s,idx) for s,idx in (('train',train_ids),('val',val_ids),('iid',test_ids),('ood',test_ids))}
    outputs['masks']=masks
    outputs['threshold']=threshold
    outputs['split_ids']={'train':train_ids.tolist(),'val':val_ids.tolist(),'iid':test_ids.tolist()}
    outputs['fingerprint']=hashlib.sha256(x_np.tobytes() + targets.tobytes() + masks.tobytes() + np.array([threshold],dtype='<f8').tobytes()).hexdigest()
    return outputs


def eval_model(model: Student, data: tuple[torch.Tensor,torch.Tensor]) -> dict:
    x,y=data
    if x.ndim!=2 or x.shape[1]!=D or y.shape!=(TASKS,len(x)): raise ValueError('dataset shape')
    with torch.no_grad():
        xx=x.repeat(TASKS,1)
        tasks=torch.arange(TASKS,dtype=torch.long).repeat_interleave(len(x))
        logits=model.eval()(xx,tasks).reshape_as(y).double()
        yd=y.double()
        bce=float(F.binary_cross_entropy_with_logits(logits,yd))
        pred=logits>=0
        acc=float((pred==yd.bool()).double().mean())
        bac=[]
        for k in range(TASKS):
            positive=(yd[k]==1); negative=~positive
            if positive.sum()==0 or negative.sum()==0: raise ValueError('single class in task')
            tpr=(pred[k][positive]).double().mean()
            tnr=(~pred[k][negative]).double().mean()
            bac.append(float((tpr+tnr)/2))
        return {'bce':bce,'balanced_accuracy':float(np.mean(bac)),'accuracy':acc,
                'worst_task_bce':max(float(F.binary_cross_entropy_with_logits(logits[k],yd[k])) for k in range(TASKS))}


def train_model(model: Student,world: dict,steps: int,seed: int,batch:int=256) -> float:
    if steps<=0 or batch<=0: raise ValueError('steps/batch')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    x,y=world['train']; model.train()
    random_gen=torch.Generator().manual_seed(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=0.0)
    for _ in range(steps):
        ix=torch.randint(len(x),(batch,),generator=random_gen)
        tk=torch.randint(TASKS,(batch,),generator=random_gen)
        optimizer.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(model(x[ix],tk),y[tk,ix])
        if not torch.isfinite(loss):raise ValueError('loss nonfinite')
        loss.backward()
        gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
        if not torch.isfinite(gradnorm): raise ValueError('grad nonfinite')
        optimizer.step()
    model.eval()
    return float(loss.item())
