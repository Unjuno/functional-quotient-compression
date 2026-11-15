# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A11 multi-family training phase only. Do NOT open dev/IID/OOD targets here."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, multiprocessing as mp
import os, platform, time
from pathlib import Path
import torch
from a11 import METHODS,FAMILIES,CAP_OPTIONS,make_world,fit_two_stage,serialize


def sha(p:Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _train_one(args):
    folder,world,family,seed,kind,cap,meta_steps,adapt_steps,batch=args
    import torch
    torch.set_num_threads(1)
    w=make_world(world,family)
    fit=fit_two_stage(w,kind,cap,seed,meta_steps=meta_steps,adapt_steps=adapt_steps,batch=batch)
    name=f'{family}_w{world}_s{seed}_{kind}_b{cap}.a11w'
    data=serialize(fit['model'])
    path=Path(folder)/'models'/name
    with path.open('xb') as f:
        f.write(data)
    return dict(world=world,family=family,seed=seed,method=kind,cap=cap,path='models/'+name,
                world_fingerprint=w['fingerprint'],stored_bytes=len(data),
                model_sha256=hashlib.sha256(data).hexdigest(),
                n_params=sum(p.numel() for p in fit['model'].parameters()),
                adapt_trainable=fit['adaptation_trainable_params'],
                meta_seconds=fit['meta_seconds'],adapt_seconds=fit['adapt_seconds'],
                meta_last_batch_mse=fit['meta_final_batch_mse'],
                adapt_last_batch_mse=fit['adapt_final_batch_mse'])


def run_training(out:Path,config:dict,workers:int=4)->dict:
    out=Path(out)
    if out.exists(): raise FileExistsError(f'refusing overwrite: {out}')
    if workers<1 or workers>8:raise ValueError('workers outside [1,8]')
    if not config.get('worlds') or not config.get('seeds') or not config.get('methods') or not config.get('caps') or not config.get('families'):
        raise ValueError('empty experiment dimension')
    if any(x not in METHODS for x in config['methods']) or any(c not in CAP_OPTIONS for c in config['caps']) or any(f not in FAMILIES for f in config['families']):
        raise ValueError('unsupported method/cap')
    if len(set(config['worlds']))!=len(config['worlds']) or len(set(config['methods']))!=len(config['methods']):
        raise ValueError('duplicated worlds or methods')
    out.mkdir(parents=True,exist_ok=False)
    (out/'models').mkdir()
    (out/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    jobs=[(str(out),w,f,s,m,c,config['meta_steps'],config['adapt_steps'],config['batch'])
          for f in config['families'] for w in config['worlds'] for s in config['seeds'] for c in config['caps'] for m in config['methods']]
    data=[]
    t0=time.perf_counter()
    if workers==1:
        for job in jobs:
            data.append(_train_one(job))
            if len(data)%24==0:print('TRAINED',len(data),'/',len(jobs),flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
            for item in pool.map(_train_one,jobs,chunksize=1):
                data.append(item)
                if len(data)%24==0:print('TRAINED',len(data),'/',len(jobs),flush=True)
    data.sort(key=lambda a:(a['family'],a['world'],a['seed'],a['cap'],a['method']))
    cpu='unavailable';mhz=[]
    try:
        for l in Path('/proc/cpuinfo').read_text().splitlines():
            if l.startswith('model name') and cpu=='unavailable':cpu=l.split(':',1)[-1].strip()
            if l.startswith('cpu MHz'):mhz.append(float(l.split(':',1)[-1]))
    except OSError:pass
    receipt=dict(experiment=config['experiment'],protocol_sha256=hashlib.sha256((Path(__file__).parent/'PROTOCOL.md').read_bytes()).hexdigest(),
        config_sha256=sha(out/'config.json'),models=data,wall_seconds=time.perf_counter()-t0,
        environment=dict(python=platform.python_version(),torch=torch.__version__,device='CPU',
            cpu=cpu,cpu_clock_observed_mhz_range=[min(mhz),max(mhz)] if mhz else None,
            clocks_locked=False,workers=workers,threads_per_worker=1,precision='FP32'))
    (out/'train.lock.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('FROZEN',len(data),'models',str(out/'train.lock.json'),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true',help='world 899 single case, never pooled')
    args=parser.parse_args()
    conf=(dict(worlds=[1099],families=['ramp'],seeds=[0],methods=['mirror','shift'],caps=[512],meta_steps=20,adapt_steps=10,batch=32,workers=1,experiment='A11_SMOKE')
          if args.smoke else dict(worlds=[1100,1101],families=list(FAMILIES),seeds=[0,1],methods=list(METHODS),caps=list(CAP_OPTIONS),
                meta_steps=600,adapt_steps=200,batch=128,workers=4,experiment='A11_MAIN'))
    run_training(args.out,conf,workers=conf['workers'])

if __name__=='__main__':main()
