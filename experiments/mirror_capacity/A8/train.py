# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Lock A8 weights before reading development or test scores."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch
import a8

HERE=Path(__file__).resolve().parent


def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()

@lru_cache(maxsize=30)
def world(world_id:int,family:str): return a8.make_world(world_id,family)


def one(record:tuple)->dict:
    w,f,s,c,k,steps,model_dir=record
    torch.set_num_threads(1)
    env=world(w,f)
    t=time.perf_counter()
    model=a8.train_one(env,k,c,s,steps=steps)
    sec=time.perf_counter()-t
    raw=a8.serialize(model['model'])
    path=f'{f}_w{w}_s{s}_b{c}_{k}.a8w'
    (Path(model_dir)/path).write_bytes(raw)
    return dict(world=w,family=f,seed=s,cap=c,kind=k,width=model['model'].width,
                parameters=sum(p.numel() for p in model['model'].parameters()),bytes=len(raw),
                sha256=sha(raw),weight_file=path,train_final_batch_mse=model['train_final_mse'],
                train_seconds=sec,dataset_sha256=env['fingerprint'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worlds',nargs='+',type=int,default=list(range(810,816)))
    p.add_argument('--families',nargs='+',choices=['hinge','ramp'],default=['hinge','ramp'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1])
    p.add_argument('--caps',nargs='+',type=int,default=[256,512])
    p.add_argument('--kinds',nargs='+',choices=a8.KINDS,default=list(a8.KINDS))
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--jobs',type=int,default=4)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('refusing to overwrite an existing run')
    if args.steps<1 or args.jobs<1 or any(w<0 for w in args.worlds):raise ValueError('invalid arguments')
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['MKL_NUM_THREADS']='1'
    os.environ['OPENBLAS_NUM_THREADS']='1'
    args.out.mkdir(parents=True)
    (args.out/'models').mkdir()
    cfg={
        'worlds':args.worlds,'families':args.families,'seeds':args.seeds,'caps':args.caps,
        'kinds':args.kinds,'steps':args.steps,'jobs':args.jobs,'batch':256,'optimizer':'AdamW',
        'lr':0.012,'weight_decay':0,'gradient_clip':5,
        'protocol_sha256':sha((HERE/'PROTOCOL.md').read_bytes()),
        'source_sha256':sha((HERE/'a8.py').read_bytes()),
        'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),
        'cpu_logical_count':os.cpu_count(),'CPU_freq_unlocked':True,
        'note':'exact same updates, NOT matched FLOPs, no audit score was read during training'}
    (args.out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    cases=list(itertools.product(args.worlds,args.families,args.seeds,args.caps,args.kinds))
    cases=[(*record,args.steps,str(args.out/'models')) for record in cases]
    rows=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(one,record) for record in cases]
        for fut in as_completed(futures):
            rows.append(fut.result())
            if len(rows)%28==0 or len(rows)==len(cases):
                print(f'A8_TRAIN_PROGRESS {len(rows)}/{len(cases)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    rows.sort(key=lambda r:(r['family'],r['world'],r['seed'],r['cap'],r['kind']))
    with (args.out/'train.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lock={'format':'a8-lock-v1','n':len(rows),'config':cfg,
          'train_csv_sha256':sha((args.out/'train.csv').read_bytes()),
          'files':{r['weight_file']:{'sha256':r['sha256'],'bytes':r['bytes'],
                                    'dataset_sha256':r['dataset_sha256']} for r in rows}}
    (args.out/'train.lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    print('A8_TRAIN_LOCKED',len(rows),f'seconds={time.perf_counter()-start:.1f}',flush=True)

if __name__=='__main__': main()
