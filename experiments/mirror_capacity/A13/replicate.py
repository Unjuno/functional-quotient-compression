# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A13c fresh worlds, separate post-discovery replication ledger."""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from a13 import (A11_DIR,serialize,deserialize,make_world,score_support,es_optimize,random_optimize,
                 adam_optimize,make_combined_model,POP,GENERATIONS)

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'/'repo'/'experiments'/'mirror_capacity'/'A12'))
from a12 import fit_meta_checkpoints
FAMILIES=('tent','sine')
WORLDS=(1300,1301,1302)
METHODS=('mirror','shift','film')
SEEDS=(0,1)

def sha(b):return hashlib.sha256(b).hexdigest()


def one_meta(args):
    out,fam,world,seed,method=args
    torch.set_num_threads(1)
    w=make_world(world,fam)
    t=time.perf_counter()
    checkpoint=fit_meta_checkpoints(w,method,1024,seed,steps=(2400,),batch=128)[2400]
    raw=serialize(checkpoint)
    path=Path(out)/'parents'/f'{fam}_w{world}_s{seed}_{method}_meta2400.a11w'
    path.parent.mkdir(exist_ok=True)
    with path.open('xb') as f:f.write(raw)
    return dict(path=str(path.relative_to(out)),family=fam,world=world,seed=seed,method=method,
                stored_bytes=len(raw),sha256=sha(raw),world_fingerprint=w['fingerprint'],meta_steps=2400,
                meta_elapsed_seconds=time.perf_counter()-t)


def one_es(args):
    out,rec=args
    torch.set_num_threads(1)
    raw=(Path(out)/rec['path']).read_bytes()
    if sha(raw)!=rec['sha256']:raise RuntimeError('parent digest changed')
    parent=deserialize(raw)
    w=make_world(rec['world'],rec['family'])
    if w['fingerprint']!=rec['world_fingerprint']:raise RuntimeError('world changed')
    es_codes=[];rnd_codes=[]
    for task in range(8):
        start=parent.new_codes[task].detach().numpy().copy()
        seed=40000011+rec['world']*1973+rec['seed']*17+METHODS.index(rec['method'])*101+task*109
        es=es_optimize(parent,w['support_x'][task],w['support_y'][task],start,seed,pop=POP,generations=GENERATIONS)
        rnd=random_optimize(parent,w['support_x'][task],w['support_y'][task],start,seed+190001,evaluations=POP*GENERATIONS)
        es_codes.append(es['code']);rnd_codes.append(rnd['code'])
    adam=adam_optimize(parent,w['support_x'],w['support_y'],steps=POP*GENERATIONS)
    codes={'es':np.stack(es_codes),'random':np.stack(rnd_codes),'adam':adam['codes']}
    rows=[]
    for key,c in codes.items():
        m=make_combined_model(parent,c)
        data=serialize(m)
        path=Path(out)/'models'/f"{rec['family']}_w{rec['world']}_s{rec['seed']}_{rec['method']}_{key}.a11w"
        path.parent.mkdir(exist_ok=True)
        with path.open('xb') as f:f.write(data)
        rows.append(dict(path=str(path.relative_to(out)),algorithm=key,family=rec['family'],world=rec['world'],seed=rec['seed'],method=rec['method'],
                         sha256=sha(data),stored_bytes=len(data),parent_sha256=rec['sha256'],
                         world_fingerprint=rec['world_fingerprint'],
                         support_mse_mean=float(np.mean([score_support(parent,w['support_x'][t],w['support_y'][t],c[t]) for t in range(8)]))))
    return rows


def run(out:Path,workers=4):
    out=out.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('do not overwrite')
    out.mkdir(exist_ok=True)
    config=dict(experiment='A13C_POSTDISCOVERY',families=FAMILIES,worlds=WORLDS,seeds=SEEDS,methods=METHODS,
                search_pop=POP,search_generations=GENERATIONS,support_per_task=32,
                meta_steps=2400,cap=1024,
                source_sha256={name:sha((HERE/name).read_bytes()) for name in ('a13.py','replicate.py','PROTOCOL_A13C.md')},
                a11_source_sha256=sha((A11_DIR/'a11.py').read_bytes()),
                a12_source_sha256=sha((HERE/'source'/'repo'/'experiments'/'mirror_capacity'/'A12'/'a12.py').read_bytes()),
                optimization_data='support only, never old-dev or IID/OOD')
    raw_cfg=json.dumps(config,sort_keys=True,indent=2).encode();(out/'config.json').write_bytes(raw_cfg)
    t0=time.perf_counter()
    inputs=[(str(out),fam,world,seed,method) for fam,world,seed,method in itertools.product(FAMILIES,WORLDS,SEEDS,METHODS)]
    parents=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one_meta,x) for x in inputs]):
            r=fut.result();parents.append(r)
            print('META',len(parents),'/',len(inputs),r['family'],r['world'],r['seed'],r['method'],round(r['meta_elapsed_seconds'],2),flush=True)
    parents.sort(key=lambda r:(r['family'],r['world'],r['seed'],r['method']))
    if len(parents)!=36:raise RuntimeError('missing parent')
    # Explicit parent barrier: all meta weights are written and hashed before any new-code search.
    (out/'meta.lock.json').write_text(json.dumps({'config_sha256':sha(raw_cfg),'parents':parents},sort_keys=True,indent=2))
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one_es,(str(out),p)) for p in parents]):
            rs=fut.result();results.extend(rs)
            print('ADAPT',len(results),'/',len(parents)*3,rs[0]['family'],rs[0]['method'],flush=True)
    results.sort(key=lambda r:(r['family'],r['world'],r['seed'],r['method'],r['algorithm']))
    if len(results)!=108:raise RuntimeError('missing final models')
    lock={'config_sha256':sha(raw_cfg),'parents':parents,'models':results,'meta_lock_sha256':sha((out/'meta.lock.json').read_bytes()),
          'train_elapsed_seconds':time.perf_counter()-t0}
    (out/'train.lock.json').write_text(json.dumps(lock,sort_keys=True,indent=2))
    print('A13C_LOCKED',len(parents),'parents',len(results),'models','seconds',lock['train_elapsed_seconds'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=HERE/'replication');ap.add_argument('--workers',type=int,default=4)
    x=ap.parse_args();run(x.out,x.workers)
