# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Train and lock A7 final checkpoints WITHOUT reading held-out audit labels."""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import torch
import numpy as np
import sklearn
import a7
a6=a7.source_core()

HERE=Path(__file__).resolve().parent


def digest(b:bytes)->str: return hashlib.sha256(b).hexdigest()


@lru_cache(maxsize=50)
def world_cache(w:int,family:str)->dict:
    return a7.make_world(w,family)


def one(item:tuple)->dict:
    world,family,seed,cap,kind,steps,outdir=item
    torch.set_num_threads(1)
    torch.manual_seed(210000+world*1009+seed*37)
    model=a6.create(kind,cap)
    data=world_cache(world,family)
    start=time.perf_counter()
    ending=a6.train_model(model,data,steps,seed=10003000+world*73+seed*19001)
    sec=time.perf_counter()-start
    raw=a6.serialize(model)
    path=f'{family}_w{world}_s{seed}_b{cap}_{kind}.a7w'
    (Path(outdir)/path).write_bytes(raw)
    return dict(world=world,family=family,seed=seed,cap=cap,kind=kind,width=model.width,
                num_parameters=sum(p.numel() for p in model.parameters()),bytes=len(raw),
                digest=digest(raw),weight_file=path,train_final_batch_loss=ending,seconds=sec,
                dataset_fingerprint=data['fingerprint'])


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--worlds',nargs='+',type=int,default=list(range(700,706)))
    p.add_argument('--families',nargs='+',choices=['pair_and','pair_xor'],default=['pair_and','pair_xor'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1])
    p.add_argument('--caps',nargs='+',type=int,default=[6000,12000])
    p.add_argument('--kinds',nargs='+',choices=a6.KINDS,default=list(a6.KINDS))
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--jobs',type=int,default=3)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('out path already exists')
    if args.steps<1 or args.jobs<1 or any(v<0 for v in args.worlds):raise ValueError('invalid arguments')
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    args.out.mkdir(parents=True)
    (args.out/'models').mkdir()
    protocol_sha=digest((HERE/'PROTOCOL.md').read_bytes())
    script_sha=digest((HERE/'a7.py').read_bytes())
    core_sha=digest((HERE.parent/'A6'/'a6.py').read_bytes())
    cases=[(w,f,s,c,k,args.steps,str(args.out/'models')) for w in args.worlds for f in args.families
           for s in args.seeds for c in args.caps for k in args.kinds]
    config={'worlds':args.worlds,'families':args.families,'seeds':args.seeds,'caps':args.caps,'kinds':args.kinds,
            'steps':args.steps,'jobs':args.jobs,'batch':256,'optimizer':'AdamW','lr':.005,'weight_decay':0.0,
            'model_protocol_sha256':protocol_sha,'implementation_sha256':script_sha,'a6_core_sha256':core_sha,
            'platform':platform.platform(),'processor':platform.processor(),
            'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,
            'cpu_threads':os.cpu_count(),'duration_note':'no fixed CPU frequency, model times include input prep/optimizer; not matched-FLOP'}
    (args.out/'config.json').write_text(json.dumps(config,indent=2,default=str)+'\n')
    results=[]; tic=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures={executor.submit(one,c):c for c in cases}
        for fut in concurrent.futures.as_completed(futures):
            row=fut.result();results.append(row)
            if len(results)%24==0 or len(results)==len(cases):
                print(f'TRAIN_PROGRESS {len(results)}/{len(cases)} seconds {time.perf_counter()-tic:.1f}',flush=True)
    results.sort(key=lambda r:(r['family'],r['world'],r['seed'],r['cap'],r['kind']))
    with (args.out/'train.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    lock={'schema':'a7-lock-v1','model_count':len(results),'created_from_config':config,
          'rows':[{'weight_file':r['weight_file'],'digest':r['digest'],'bytes':r['bytes'],
                   'dataset_fingerprint':r['dataset_fingerprint']} for r in results],
          'train_csv_sha256':digest((args.out/'train.csv').read_bytes())}
    (args.out/'train.lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    print('TRAIN_LOCKED',len(results),'seconds',round(time.perf_counter()-tic,2),flush=True)


if __name__=='__main__': main()
