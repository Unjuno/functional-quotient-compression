# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A13: black-box ES for the new-task 2D codes of exactly frozen A12 meta scaffolds.

No test split is consulted here. Teacher coefficients are never exposed to the optimizers.
The cached frozen weights have been learned on A12's old 16 tasks; code-only
optimization is evaluated on 32 labeled examples per new task.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import sys
import time
import zipfile
from pathlib import Path
from typing import Any
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
if (HERE.parent/'A11'/'a11.py').exists():
    A11_DIR=HERE.parent/'A11'
else:
    A11_DIR=HERE/'source'/'repo'/'experiments'/'mirror_capacity'/'A11'
sys.path.insert(0,str(A11_DIR))
from a11 import deserialize, serialize, Student, K_NEW, K_TRAIN, make_world, METHODS, count_bytes

METHODS_A13=('mirror','shift','film')
POP=32
GENERATIONS=30
ELITE=8
SEARCH_BOUND=4.0


def _arr(x)->np.ndarray:
    if isinstance(x,torch.Tensor):return x.detach().cpu().numpy().astype(np.float64,copy=False)
    return np.asarray(x,dtype=np.float64)


def candidate_predict(model:Student,x:torch.Tensor|np.ndarray,codes:np.ndarray)->np.ndarray:
    """Vectorized P candidate 2D codes against the same N inputs: returns [P,N]."""
    if model.method not in METHODS_A13:raise ValueError('A13 supports mirror/shift/film only')
    xx=_arr(x)
    cc=_arr(codes)
    if xx.ndim!=2 or xx.shape[1]!=1 or cc.ndim!=2 or cc.shape[1]!=2 or cc.shape[0]<1:
        raise ValueError('expected x[N,1], codes[P,2]')
    if not np.isfinite(cc).all() or not np.isfinite(xx).all():raise ValueError('nonfinite')
    with torch.no_grad():
        ww=_arr(model.up.weight).T  # [1,H]
        bb=_arr(model.up.bias)[None,:]
        down=_arr(model.down.weight).reshape(-1)
        db=float(model.down.bias.item())
        v=xx@ww+bb  # [N,H]
        if model.method=='mirror':
            pair=v.reshape(len(xx),model.width//2,2)
            a=cc[:,np.arange(model.width//2)%2]
            z1=np.maximum(pair[None,:,:,1],0)
            z0=np.maximum(pair[None,:,:,0]+a[:,None,:]*pair[None,:,:,1],0)-a[:,None,:]*z1
            h=np.stack((z0,np.broadcast_to(z1,z0.shape)),-1).reshape(len(cc),len(xx),model.width)
        else:
            shift=cc@_arr(model.shift_w)
            u=v[None,:,:]+shift[:,None,:]
            if model.method=='film':
                u=u*(1+cc@_arr(model.gain_w))[:,None,:]
            h=np.maximum(u,0)
        return np.einsum('pnh,h->pn',h,down,optimize=True)+db


def loss_batch(model:Student,x,y,codes):
    yy=_arr(y).reshape(-1)
    pp=candidate_predict(model,x,np.asarray(codes).reshape(-1,2))
    if pp.shape[1]!=len(yy):raise ValueError('bad target shape')
    return ((pp-yy[None,:])**2).mean(1)


def score_support(model,x,y,code):return float(loss_batch(model,x,y,np.asarray(code).reshape(1,2))[0])


def es_optimize(model:Student,x,y,start,seed:int,pop:int=POP,generations:int=GENERATIONS)->dict[str,Any]:
    """Bounded diagonal-variance (mu,lambda)-ES, with strict support-only selection."""
    if pop<ELITE or generations<1 or pop%2:raise ValueError('invalid ES population or generations')
    rng=np.random.default_rng(int(seed))
    mean=np.clip(_arr(start).copy().reshape(2),-SEARCH_BOUND,SEARCH_BOUND)
    sigma=np.full((2,),1.35)
    best=mean.copy()
    bestloss=score_support(model,x,y,best)
    trace=[];bound_hits=0
    for generation in range(generations):
        # Fixed zero-mean antithetic population removes some noise in mean updates.
        z=rng.standard_normal((pop//2,2));z=np.concatenate((z,-z),axis=0)
        proposed=mean[None,:]+z*sigma[None,:]
        batch=np.clip(proposed,-SEARCH_BOUND,SEARCH_BOUND)
        bound_hits+=int(np.any((proposed!=batch),axis=1).sum())
        errors=loss_batch(model,x,y,batch)
        top=np.argsort(errors,kind='stable')[:ELITE]
        winner=int(top[0])
        if float(errors[winner])<bestloss:
            bestloss=float(errors[winner]);best=batch[winner].copy()
        elite=batch[top,:]
        center=elite.mean(0)
        mean=.45*mean+.55*center
        elite_sd=np.sqrt(((elite-center[None,:])**2).mean(0))
        sigma=np.clip(.50*sigma+.50*elite_sd,.08,2.50)
        trace.append({'generation':generation+1,'best_support_mse':bestloss,
                      'population_best_mse':float(errors[winner]),
                      'mean':mean.tolist(),'sigma':sigma.tolist(),
                      'elite_mean_mse':float(errors[top].mean())})
    return {'code':best,'best_support_mse':bestloss,'evaluations':pop*generations,
            'bound_hit_rate':bound_hits/(pop*generations),'trace':trace}


def random_optimize(model,x,y,start,seed:int,evaluations:int=POP*GENERATIONS)->dict[str,Any]:
    if evaluations<1:raise ValueError('bad evaluation budget')
    rng=np.random.default_rng(int(seed))
    # Equal budget with ES, select only by support loss.
    candidates=rng.uniform(-SEARCH_BOUND,SEARCH_BOUND,(evaluations,2))
    errors=loss_batch(model,x,y,candidates)
    idx=int(np.argmin(errors))
    best=np.asarray(start,dtype=np.float64).reshape(2).copy()
    bestloss=score_support(model,x,y,best)
    if float(errors[idx])<bestloss:
        bestloss=float(errors[idx]);best=candidates[idx].copy()
    return {'code':best,'best_support_mse':bestloss,'evaluations':evaluations,'bound_hit_rate':0.0,'trace':[]}


def make_combined_model(parent:Student,new_codes:np.ndarray)->Student:
    code=np.asarray(new_codes,dtype=np.float32)
    if code.shape!=(K_NEW,2) or not np.isfinite(code).all():raise ValueError('bad final code table')
    m=copy.deepcopy(parent).cpu().eval()
    with torch.no_grad():m.new_codes.copy_(torch.from_numpy(code))
    return m


def load_archived_parent(zip_file:Path,family:str,world_id:int,seed:int,method:str)->tuple[Student,bytes,dict]:
    """Validate parent SHA and roundtrip against the A12 locked meta checklist."""
    path=Path(zip_file)
    if not path.is_file():raise ValueError('A12 ZIP is missing')
    with zipfile.ZipFile(path) as z:
        ledger=json.loads(z.read('main/train.lock.json'))
        rows=[q for q in ledger['meta_models'] if q['family']==family and q['world']==world_id and q['seed']==seed and q['method']==method and q['meta_steps']==2400]
        if len(rows)!=1:raise ValueError('no unique meta parent')
        row=rows[0]
        raw=z.read('main/'+row['path'])
        if hashlib.sha256(raw).hexdigest()!=row['sha256'] or len(raw)!=row['stored_bytes']:
            raise RuntimeError('archived parent hash and size mismatch')
        model=deserialize(raw)
        if serialize(model)!=raw or model.method!=method or count_bytes(model)>1024:
            raise RuntimeError('archived parent schema or roundtrip mismatch')
    return model,raw,row


def adam_optimize(parent:Student,support_x:torch.Tensor,support_y:torch.Tensor,steps:int=POP*GENERATIONS)->dict:
    """All eight codes optimized jointly on exactly 32 per-task support labels."""
    torch.set_num_threads(1)
    model=copy.deepcopy(parent).cpu()
    model.enter_adaptation()
    old={name:p.detach().clone() for name,p in model.named_parameters() if name!='new_codes'}
    sx=support_x.reshape(K_NEW*len(support_x[0]),1)
    ys=support_y.reshape(-1)
    task=torch.arange(K_TRAIN,K_TRAIN+K_NEW,dtype=torch.long).repeat_interleave(len(support_x[0]))
    opt=torch.optim.AdamW([model.new_codes],lr=.01,weight_decay=0)
    trace=[]
    for step in range(1,steps+1):
        opt.zero_grad(set_to_none=True)
        pred=model(sx,task)
        loss=(pred-ys).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite Adam loss')
        loss.backward()
        opt.step()
        if step in (1,60,180,360,600,steps):
            trace.append({'step':step,'support_mse':float(loss.detach())})
    for name,p in model.named_parameters():
        if name in old and not torch.equal(old[name],p.detach()):
            raise RuntimeError('Adam modified frozen shared or old task weights')
    codes=model.new_codes.detach().cpu().numpy().copy()
    values=[score_support(parent,support_x[t],support_y[t],codes[t]) for t in range(K_NEW)]
    return {'codes':codes,'support_mses':values,'evaluations':steps,'trace':trace}


def local_jacobian_singular_values(model:Student,x:torch.Tensor,code:np.ndarray,delta:float=1e-3)->tuple[float,float]:
    """Finite difference support Jacobian singular values for diagnostic only."""
    cc=np.asarray(code,dtype=np.float64).copy().reshape(2)
    diffs=[]
    for i in range(2):
        v=np.zeros((2,));v[i]=delta
        hi=candidate_predict(model,x,(cc+v)[None,:])[0]
        lo=candidate_predict(model,x,(cc-v)[None,:])[0]
        diffs.append((hi-lo)/(2*delta))
    s=np.linalg.svd(np.stack(diffs,axis=1),compute_uv=False)
    return float(s[-1]),float(s[0])
