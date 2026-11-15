# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Train A12 and LOCK all checkpoint bytes before opening held-out labels."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, multiprocessing as mp, platform, time
from pathlib import Path
import torch
from a12 import METHODS,FAMILIES,PROTOCOL,fit_meta_checkpoints,adapt_branch,make_world,serialize


def _one(job):
    out,fam,wid,seed,method,cfg=job
    torch.set_num_threads(1)
    t=time.perf_counter()
    w=make_world(wid,fam)
    meta=fit_meta_checkpoints(w,method,cfg['cap'],seed,steps=tuple(cfg['meta_checkpoints']),batch=cfg['batch'])
    receipts=[];meta_receipts=[]
    for n,model in meta.items():
        raw=serialize(model)
        name=f'{fam}_w{wid}_s{seed}_{method}_meta{n}.a11w'
        with (Path(out)/'meta'/name).open('xb') as f:f.write(raw)
        meta_receipts.append(dict(family=fam,world=wid,seed=seed,method=method,meta_steps=n,
                                  path='meta/'+name,sha256=hashlib.sha256(raw).hexdigest(),stored_bytes=len(raw)))
        for support_n in cfg['support_counts']:
            frozen,snap=adapt_branch(model,w,wid,fam,seed,method,support_n,steps=tuple(cfg['adapt_checkpoints']),batch=cfg['batch'])
            for step,fit in snap.items():
                data=serialize(fit)
                name=f'{fam}_w{wid}_s{seed}_{method}_meta{n}_support{support_n}_adapt{step}.a11w'
                with (Path(out)/'models'/name).open('xb') as f:f.write(data)
                receipts.append(dict(family=fam,world=wid,seed=seed,method=method,meta_steps=n,support_n=support_n,
                                     adapt_steps=step,world_fingerprint=w['fingerprint'],path='models/'+name,
                                     sha256=hashlib.sha256(data).hexdigest(),stored_bytes=len(data),width=fit.width,
                                     frozen_shared_sha256=frozen))
    return receipts,meta_receipts,time.perf_counter()-t


def run_experiment(out:Path,cfg:dict)->dict:
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    if not cfg['families'] or not cfg['worlds'] or not cfg['methods'] or not cfg['seeds']:
        raise ValueError('empty experiment factor')
    if any(f not in FAMILIES for f in cfg['families']) or any(m not in METHODS for m in cfg['methods']):
        raise ValueError('unsupported family or method')
    out.mkdir(parents=True,exist_ok=False)
    (out/'models').mkdir();(out/'meta').mkdir()
    (out/'config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    jobs=[(str(out),fam,wid,seed,method,cfg) for fam in cfg['families'] for wid in cfg['worlds'] for seed in cfg['seeds'] for method in cfg['methods']]
    all_models=[];all_meta=[]
    start=time.perf_counter()
    if cfg['workers']==1:
        result=map(_one,jobs)
        for i,(models,meta,elapsed) in enumerate(result,1):
            all_models.extend(models);all_meta.extend(meta)
            print('TRAINED',i,'/',len(jobs),'models',len(all_models),'job-sec',round(elapsed,2),flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn')) as pool:
            for i,(models,meta,elapsed) in enumerate(pool.map(_one,jobs,chunksize=1),1):
                all_models.extend(models);all_meta.extend(meta)
                if i%3==0:print('TRAINED',i,'/',len(jobs),'models',len(all_models),flush=True)
    all_models.sort(key=lambda r:(r['family'],r['world'],r['method'],r['seed'],r['meta_steps'],r['support_n'],r['adapt_steps']))
    all_meta.sort(key=lambda r:(r['family'],r['world'],r['method'],r['seed'],r['meta_steps']))
    cpu='unknown';mhz=[]
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name') and cpu=='unknown':cpu=line.split(':')[-1].strip()
            if line.startswith('cpu MHz'):mhz.append(float(line.split(':')[-1]))
    except OSError:pass
    lock=dict(experiment=cfg['experiment'],config_sha256=hashlib.sha256((out/'config.json').read_bytes()).hexdigest(),
              protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),models=all_models,meta_models=all_meta,
              wall_seconds=time.perf_counter()-start,
              environment=dict(python=platform.python_version(),torch=torch.__version__,CPU=cpu,
                               observed_mhz_range=[min(mhz),max(mhz)] if mhz else None,clocks_locked=False,
                               workers=cfg['workers'],threads_per_worker=1,dtype='FP32'))
    (out/'train.lock.json').write_text(json.dumps(lock,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('FROZEN',len(all_models),'models +',len(all_meta),'meta checkpoints',flush=True)
    return lock

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    cfg=(dict(experiment='A12_SMOKE',families=['tent'],worlds=[1199],seeds=[0],methods=['mirror'],cap=1024,
              meta_checkpoints=[2,4],support_counts=[32,256],adapt_checkpoints=[2,4],batch=16,workers=1)
         if args.smoke else dict(experiment='A12_MAIN',families=list(FAMILIES),worlds=[1200,1201,1202],seeds=[0,1],methods=list(METHODS),cap=1024,
              meta_checkpoints=[600,2400],support_counts=[32,256],adapt_checkpoints=[200,800],batch=128,workers=args.workers))
    run_experiment(args.out,cfg)
