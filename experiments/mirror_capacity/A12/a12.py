# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A12 shared-scaffold vs support vs adaptation-optimizer factorial study.

Reuse the frozen A11 function generators, model, and exact FP32 serializer; no
teacher parameters enter any model state or learned initialization.
"""
from __future__ import annotations
import copy, csv, hashlib, json, math, os, platform, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn

A11_ROOT = Path(__file__).resolve().parents[1] / 'A11'
if str(A11_ROOT) not in sys.path:
    sys.path.insert(0, str(A11_ROOT))
from a11 import (METHODS as A11_METHODS, FAMILIES as A11_FAMILIES, K_TRAIN, K_NEW,
                 make_world, make_model, method_free_seed, Student,
                 serialize, deserialize, evaluate, count_bytes)

FAMILIES=('tent','sine')
METHODS=('mirror','shift','film')
PROTOCOL=Path(__file__).with_name('PROTOCOL.md')


def make_support(world_id:int,family:str,n:int)->tuple[torch.Tensor,torch.Tensor]:
    """Labels from A11 teacher. n=32 is bit-identical to the A11 support set.

    Teacher parameters are not returned and are used only inside target generation.
    The n=256 set contains the n=32 set as the prefix for each new task.
    """
    if family not in FAMILIES or type(world_id) is not int or type(n) is not int or n not in (32,256,1024):
        raise ValueError('unsupported support design')
    tag=A11_FAMILIES.index(family)
    rng=np.random.default_rng(417211+world_id*19937+tag*170981)
    count=K_TRAIN+K_NEW
    if family=='tent':
        center=rng.uniform(-.65,.65,count).astype('float32')
        width=rng.uniform(.20,.46,count).astype('float32')
        def target(x):
            return (np.maximum(x[...,None]-(center-width),0)
                    -2*np.maximum(x[...,None]-center,0)
                    +np.maximum(x[...,None]-(center+width),0))/width
    else:
        freq=rng.uniform(.75,2.5,count).astype('float32')
        phase=rng.uniform(-np.pi,np.pi,count).astype('float32')
        scale=rng.uniform(.65,1.15,count).astype('float32')
        def target(x):return scale*np.sin(freq*x[...,None]+phase)
    x=np.stack([np.random.default_rng(world_id*10003+tag*2999+800+t).uniform(-1.5,1.5,(n,1)).astype('float32') for t in range(K_NEW)])
    y=np.stack([target(x[t]).squeeze(axis=-2).astype('float32')[:,K_TRAIN+t] for t in range(K_NEW)])
    return torch.from_numpy(x),torch.from_numpy(y)


def new_code_start(world_id:int,method:str,seed:int,cap:int)->torch.Tensor:
    m=make_model(method,cap,world_id*1009+seed*71+method_free_seed(method))
    return m.new_codes.detach().clone()


def fit_meta_checkpoints(w:dict,method:str,cap:int,seed:int,steps:tuple[int,...]=(600,2400),batch:int=128)->dict[int,Student]:
    if method not in METHODS or not steps or any(type(s)!=int or s<1 for s in steps) or sorted(set(steps))!=list(steps):
        raise ValueError('bad meta checkpoint sequence')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model=make_model(method,cap,w['world_id']*1009+seed*71+method_free_seed(method))
    model.enter_meta()
    gen=torch.Generator().manual_seed(630001+w['world_id']*107+seed*347)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.01,weight_decay=0)
    x,y=w['meta_x'],w['train_y'];saved={}
    for step in range(1,max(steps)+1):
        ix=torch.randint(len(x),(batch,),generator=gen)
        task=torch.randint(K_TRAIN,(batch,),generator=gen)
        opt.zero_grad(set_to_none=True)
        loss=(model(x[ix],task)-y[ix,task]).square().mean()
        if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite meta loss')
        loss.backward()
        g=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),5.)
        if not bool(torch.isfinite(g)):raise RuntimeError('nonfinite meta gradient')
        opt.step()
        if step in steps:
            saved[step]=copy.deepcopy(model).eval()
    return saved


def adapt_branch(meta_model:Student,w:dict,world_id:int,family:str,seed:int,method:str,
                 support_n:int,steps:tuple[int,...]=(200,800),batch:int=128)->tuple[str,dict[int,Student]]:
    if not steps or any(type(i)!=int or i<1 for i in steps) or sorted(set(steps))!=list(steps):
        raise ValueError('bad adaptation steps')
    if support_n==32:
        sx,sy=w['support_x'],w['support_y']
    else:
        sx,sy=make_support(world_id,family,support_n)
    model=copy.deepcopy(meta_model)
    model.enter_adaptation()
    frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
    frozen_sha=hashlib.sha256(b''.join(p.contiguous().numpy().tobytes() for p in frozen.values())).hexdigest()
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.01,weight_decay=0)
    gen=torch.Generator().manual_seed(740001+world_id*107+seed*347)
    saved={}
    for step in range(1,max(steps)+1):
        task=torch.randint(K_NEW,(batch,),generator=gen)
        ix=torch.randint(support_n,(batch,),generator=gen)
        opt.zero_grad(set_to_none=True)
        loss=(model(sx[task,ix],task+K_TRAIN)-sy[task,ix]).square().mean()
        if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite adaptation loss')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),5.)
        if not bool(torch.isfinite(grad)):raise RuntimeError('nonfinite adaptation gradient')
        opt.step()
        if step in steps:
            if any(not torch.equal(p.detach(),frozen[name]) for name,p in model.named_parameters() if name in frozen):
                raise RuntimeError('adaptation illegally modified frozen shared weights')
            saved[step]=copy.deepcopy(model).eval()
    return frozen_sha,saved


def _sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_run(out:Path,recompute:bool=False)->dict:
    """Verify *all* frozen bytes and config before evaluating held-out labels."""
    out=Path(out)
    lock=json.loads((out/'train.lock.json').read_text(encoding='utf-8'))
    if lock['protocol_sha256']!=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest():
        raise RuntimeError('protocol SHA mismatch')
    cfg_file=out/'config.json'
    if lock['config_sha256']!=_sha(cfg_file):
        raise RuntimeError('config SHA mismatch')
    cfg=json.loads(cfg_file.read_text())
    expected=(len(cfg['families'])*len(cfg['worlds'])*len(cfg['seeds'])*len(cfg['methods'])*
              len(cfg['meta_checkpoints'])*len(cfg['support_counts'])*len(cfg['adapt_checkpoints']))
    if len(lock['models'])!=expected:
        raise RuntimeError('locked checkpoint count mismatch')
    if (out/'audit.csv').exists() and not recompute:
        raise FileExistsError('audit already exists')
    expected_meta=(len(cfg['families'])*len(cfg['worlds'])*len(cfg['seeds'])*len(cfg['methods'])*len(cfg['meta_checkpoints']))
    if len(lock.get('meta_models',[]))!=expected_meta:
        raise RuntimeError('meta checkpoint count mismatch')
    for meta in lock['meta_models']:
        raw=(out/meta['path']).read_bytes()
        if len(raw)!=meta['stored_bytes'] or hashlib.sha256(raw).hexdigest()!=meta['sha256']:
            raise RuntimeError('meta checkpoint hash/size mismatch')
        m=deserialize(raw)
        if serialize(m)!=raw or m.method!=meta['method'] or count_bytes(m)>cfg['cap']:
            raise RuntimeError('meta checkpoint roundtrip/method/budget mismatch')
    # No scoring until all hashes and complete reserialization roundtrips pass.
    all_models=[]
    for r in lock['models']:
        raw=(out/r['path']).read_bytes()
        if len(raw)!=r['stored_bytes'] or hashlib.sha256(raw).hexdigest()!=r['sha256']:
            raise RuntimeError('checkpoint hash/length mismatch')
        m=deserialize(raw)
        if serialize(m)!=raw:
            raise RuntimeError('checkpoint reserialization mismatch')
        if m.method!=r['method'] or m.width!=r['width'] or r['stored_bytes']>cfg['cap']:
            raise RuntimeError('checkpoint method/width/cap mismatch')
        all_models.append((r,m))
    worlds={(family,world):make_world(world,family) for family in cfg['families'] for world in cfg['worlds']}
    scored=[]
    for r,m in all_models:
        w=worlds[(r['family'],r['world'])]
        if w['fingerprint']!=r['world_fingerprint']:
            raise RuntimeError('world fingerprint mismatch')
        old=evaluate(m,w['meta_dev_x'],w['meta_dev_y'],0)
        iid=evaluate(m,w['new_iid_x'],w['new_iid_y'],K_TRAIN)
        ood=evaluate(m,w['new_ood_x'],w['new_ood_y'],K_TRAIN)
        scored.append({**r,'old_dev_nrmse':old['nrmse'],'new_iid_nrmse':iid['nrmse'],
                       'new_ood_nrmse':ood['nrmse'],'old_dev_mse':old['mse'],
                       'new_iid_mse':iid['mse'],'new_ood_mse':ood['mse']})
    with (out/'audit.csv').open('w' if recompute else 'x',newline='',encoding='utf-8') as f:
        z=csv.DictWriter(f,fieldnames=scored[0].keys());z.writeheader();z.writerows(scored)
    grouped=defaultdict(list)
    for r in scored:grouped[(r['family'],r['world'],r['method'],r['meta_steps'],r['support_n'],r['adapt_steps'])].append(r)
    world_rows=[]
    for k,rows in sorted(grouped.items()):
        rr=dict(zip(('family','world','method','meta_steps','support_n','adapt_steps'),k))
        rr['seeds']=len(rows)
        rr['stored_bytes']=sum(x['stored_bytes'] for x in rows)/len(rows)
        for met in ('old_dev_nrmse','new_iid_nrmse','new_ood_nrmse'):
            rr[met]=sum(r[met] for r in rows)/len(rows)
        rr['quality_pass']=int(max(rr['old_dev_nrmse'],rr['new_iid_nrmse'],rr['new_ood_nrmse'])<=.20)
        world_rows.append(rr)
    with (out/'world_means.csv').open('w' if recompute else 'x',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=world_rows[0].keys());wr.writeheader();wr.writerows(world_rows)
    mapping={(r['family'],r['world'],r['method'],r['meta_steps'],r['support_n'],r['adapt_steps']):r for r in world_rows}
    effects=[]
    for fam in cfg['families']:
      for wid in cfg['worlds']:
       for method in cfg['methods']:
        # Conditions must all be available; test different factors at fixed others.
        best=mapping[(fam,wid,method,max(cfg['meta_checkpoints']),max(cfg['support_counts']),max(cfg['adapt_checkpoints']))]
        for axis,alt in [('meta',mapping[(fam,wid,method,min(cfg['meta_checkpoints']),max(cfg['support_counts']),max(cfg['adapt_checkpoints']))]),
                         ('support',mapping[(fam,wid,method,max(cfg['meta_checkpoints']),min(cfg['support_counts']),max(cfg['adapt_checkpoints']))]),
                         ('adapt',mapping[(fam,wid,method,max(cfg['meta_checkpoints']),max(cfg['support_counts']),min(cfg['adapt_checkpoints']))])]:
            effects.append(dict(family=fam,world=wid,method=method,axis=axis,
                                baseline_iid=alt['new_iid_nrmse'],intervention_iid=best['new_iid_nrmse'],
                                delta_iid=alt['new_iid_nrmse']-best['new_iid_nrmse'],
                                relative_iid_gain=1-best['new_iid_nrmse']/max(alt['new_iid_nrmse'],1e-15),
                                baseline_old=alt['old_dev_nrmse'],intervention_old=best['old_dev_nrmse'],
                                delta_old=alt['old_dev_nrmse']-best['old_dev_nrmse'],
                                delta_ood=alt['new_ood_nrmse']-best['new_ood_nrmse']))
    with (out/'factor_effects.csv').open('w' if recompute else 'x',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=effects[0].keys());wr.writeheader();wr.writerows(effects)
    response={'models':len(all_models),'split_scores':len(scored)*3,'world_means':world_rows,'factor_effects':effects,
              'interpretation':'Three-world synthetic paired exploratory factorial; no statistically general capacity or hardware claims.'}
    (out/'summary.json').write_text(json.dumps(response,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return response
