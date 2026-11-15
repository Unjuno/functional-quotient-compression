# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Post-discovery A13b: multistart gradient and low-discrepancy search controls."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
import copy
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from scipy.stats import qmc
from a13 import (make_world,deserialize,serialize,make_combined_model,loss_batch,score_support,
                 K_TRAIN,K_NEW,SEARCH_BOUND)

ROOT=Path(__file__).resolve().parent


def sha(b):return hashlib.sha256(b).hexdigest()


def json_native(value):
    """Convert NumPy numeric scalars in evidence records to JSON-safe Python values."""
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {key:json_native(val) for key,val in value.items()}
    if isinstance(value,(tuple,list)):return [json_native(v) for v in value]
    return value

def enrich_parents(parents,rows):
    """Join parent SHA256 to locked A13 setup metadata; avoid path parsing."""
    mapping={}
    for r in rows:
        key=r['parent_sha256']
        val={k:r[k] for k in ('family','world','seed','method')}
        if key in mapping and mapping[key]!=val:
            raise RuntimeError('parent SHA maps to conflicting setups')
        mapping[key]=val
    if len(parents)!=len(mapping):raise RuntimeError('parent ledger mismatch')
    results=[]
    for p in parents:
        if p['sha256'] not in mapping:raise RuntimeError('orphan parent')
        results.append({**p,**mapping[p['sha256']]})
    return results

def single_parent(record,main:Path,out:Path):
    torch.set_num_threads(1)
    family,world,seed,method=(record[k] for k in ('family','world','seed','method'))
    meta_path=main/record['path']
    raw=meta_path.read_bytes()
    if sha(raw)!=record['sha256']:raise RuntimeError('parent changed')
    parent=deserialize(raw);w=make_world(world,family)
    key=f'{family}_w{world}_s{seed}_{method}'
    starts=parent.new_codes.detach().numpy().copy()
    # Four independent 240-step optimizations, only new_codes learnable.
    runs=[]
    start_t=time.perf_counter()
    for rep in range(4):
        model=copy.deepcopy(parent);model.enter_adaptation()
        if rep:
            rng=np.random.default_rng(190135+world*977+seed*31+('mirror','shift','film').index(method)*191+rep*2057)
            c=rng.uniform(-SEARCH_BOUND,SEARCH_BOUND,size=(K_NEW,2)).astype('float32')
            with torch.no_grad():model.new_codes.copy_(torch.from_numpy(c))
        support_x=w['support_x'].reshape(-1,1)
        support_y=w['support_y'].reshape(-1)
        ids=torch.arange(K_TRAIN,K_TRAIN+K_NEW).repeat_interleave(32)
        opt=torch.optim.AdamW([model.new_codes],lr=.01,weight_decay=0)
        for step in range(240):
            opt.zero_grad(set_to_none=True)
            l=(model(support_x,ids)-support_y).square().mean()
            l.backward();opt.step()
        runs.append(model.new_codes.detach().numpy().copy())
    selected=np.empty((K_NEW,2),np.float32)
    multistr=[]
    for task in range(K_NEW):
        options=np.stack([starts[task]]+[z[task] for z in runs])
        losses=loss_batch(parent,w['support_x'][task],w['support_y'][task],options)
        i=int(np.argmin(losses));selected[task]=options[i]
        multistr.append(dict(task=task,restart=i if i else 'start',support_mse=float(losses[i])))
    result={'adam4':selected}
    sobol=np.empty((K_NEW,2),np.float32)
    for task in range(K_NEW):
        seed_q=800413+world*157+seed*19+('mirror','shift','film').index(method)*397+task*137
        sampler=qmc.Sobol(d=2,scramble=True,seed=seed_q)
        candidates=(2*SEARCH_BOUND*sampler.random_base2(10)[:960]-SEARCH_BOUND).astype('float64')
        losses=loss_batch(parent,w['support_x'][task],w['support_y'][task],candidates)
        i=int(np.argmin(losses));start_loss=score_support(parent,w['support_x'][task],w['support_y'][task],starts[task])
        sobol[task]=candidates[i] if losses[i]<start_loss else starts[task]
    result['sobol']=sobol
    elapsed=time.perf_counter()-start_t
    rows=[]
    for mode,code in result.items():
        m=make_combined_model(parent,code)
        data=serialize(m)
        target=out/'models'/f'{key}_{mode}.a11w';target.parent.mkdir(exist_ok=True)
        with target.open('xb') as f:f.write(data)
        rows.append(dict(path=str(target.relative_to(out)),family=family,world=world,seed=seed,method=method,algorithm=mode,
                         stored_bytes=len(data),sha256=sha(data),parent_sha256=record['sha256'],
                         world_fingerprint=w['fingerprint'],support_mse=np.mean([score_support(parent,w['support_x'][i],w['support_y'][i],code[i]) for i in range(K_NEW)]),
                         elapsed_s=elapsed,code_l2_mean=np.mean(np.linalg.norm(code,axis=1))))
    return rows,multistr


def run(main:Path,out:Path,workers:int=4):
    main=main.resolve();out=out.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('refusing overwrite')
    out.mkdir(exist_ok=True)
    main_lock=json.loads((main/'train.lock.json').read_text())
    if main_lock['experiment']!='A13_ES':raise ValueError('not A13 main evidence')
    parent_records=enrich_parents(main_lock['parents'],main_lock['models'])
    cfg={"experiment":"A13B_EXPLORATORY","design_after_A13_results":True,"parent_main_sha256":sha((main/'train.lock.json').read_bytes()),
         "protocol_sha256":sha((ROOT/'PROTOCOL_A13B.md').read_bytes()),"source_sha256":sha((ROOT/'followup.py').read_bytes()),
         "restart_count":4,"steps_per_restart":240,"srmse_selection":"support only","sobol_points":960,"bound":SEARCH_BOUND}
    conf=json.dumps(cfg,indent=2,sort_keys=True).encode();(out/'config.json').write_bytes(conf)
    allrows=[];tic=time.perf_counter();rets=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures=[ex.submit(single_parent,r,main,out) for r in parent_records]
        for f in as_completed(futures):
            rows,sel=f.result();allrows.extend(rows);rets.append(sel)
            print('FOLLOWUP',len(allrows),'/',len(parent_records)*2,rows[0]['family'],rows[0]['method'],flush=True)
    allrows.sort(key=lambda r:(r['family'],r['world'],r['seed'],r['method'],r['algorithm']))
    if len(allrows)!=72:raise RuntimeError('missing followup')
    lock=dict(config_sha256=sha(conf),models=allrows,training_seconds=time.perf_counter()-tic)
    (out/'train.lock.json').write_text(json.dumps(json_native(lock),indent=2,sort_keys=True))
    print('FOLLOWUP_LOCKED',len(allrows),'models, seconds',lock['training_seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',type=Path,default=ROOT/'main');p.add_argument('--out',type=Path,default=ROOT/'followup');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.main,a.out,a.workers)
