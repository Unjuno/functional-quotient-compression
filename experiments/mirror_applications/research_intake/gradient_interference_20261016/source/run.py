"""Run calibration, freeze selections, then predeclared fresh GIC001 comparisons."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any
import numpy as np
import core

ROOT=Path(__file__).resolve().parents[1]
GROUPS={
    'shared':['shared','shared_pcgrad'],
    'rank2':['mirror_r2','mirror_r2_pcgrad','native_lowrank'],
    'private':['selected_private','selected_private_pcgrad','random_private','random_private_pcgrad'],
    'rank3':['rank3'], 'film':['film'], 'independent':['independent']}

def state_hash(p:dict[str,np.ndarray]) -> str:
    h=hashlib.sha256()
    for k in sorted(p): h.update(k.encode());h.update(p[k].tobytes())
    return h.hexdigest()

def metrics(w:dict[str,Any],name:str,p:dict[str,np.ndarray],split:str) -> np.ndarray:
    return np.mean((w['B'+split]@core.coefficients(name,p).T-w['Y'+split])**2,axis=0)

def fit(w:dict[str,Any],method:str,init:int,lr:float,steps:int=600,
        prefix:int=150,dead:bool=False) -> tuple[dict[str,np.ndarray],dict[str,Any],list[dict[str,Any]]]:
    is_private='private' in method
    p=core.initialize('mirror_r2' if is_private else method,8,17,init)
    if dead:
        p['c0'][:]=core.oracle(w['Btrain'],w['Ytrain'],0)[0];p['D'][:]=0;p['M'][:]=0
    opt=core.Adam(p,lr)
    rng=np.random.default_rng(np.random.SeedSequence([w['world'],init,700]))
    random_ids=sorted(np.random.default_rng(np.random.SeedSequence([w['world'],init,701])).choice(8,2,replace=False).tolist())
    history=[]; prefix_hash=''; allocation=[]
    start=time.perf_counter()
    for step in range(steps+1):
        g,shared,_=core.gradients(method,p,w['H'],w['Q'])
        checkpoint=step in set([0,prefix,steps//2,steps])
        if checkpoint:
            tr=metrics(w,method,p,'train'); val=metrics(w,method,p,'val')
            entry={'step':step,'train_mean':float(tr.mean()),'train_worst':float(tr.max()),
                   'val_mean':float(val.mean()),'val_worst':float(val.max()),**core.diagnostics(shared)}
            history.append(entry)
        if step==steps: break
        if is_private and step==prefix:
            prefix_hash=state_hash(p)
            if method.startswith('selected'):
                allocation=np.argsort(-metrics(w,method,p,'val'),kind='stable')[:2].tolist()
            else: allocation=random_ids
            p['P']=np.zeros((2,17));p['ids']=np.array(allocation,dtype=np.int64)
            opt=core.Adam(p,lr)
            g,shared,_=core.gradients(method,p,w['H'],w['Q'])
        if 'pcgrad' in method and (not is_private or step>=prefix):
            g=core.project_into(g,shared,method,rng)
        if checkpoint:
            old={k:p[k].copy() for k in core.shared_keys(method)}
            before=metrics(w,method,p,'train')
        opt.update(p,g)
        if checkpoint:
            delta=np.concatenate([(p[k]-old[k]).ravel() for k in core.shared_keys(method)]) if old else np.empty(0)
            slopes=shared@delta
            history[-1]['actual_shared_step_harm_fraction']=float(np.mean(slopes>1e-12)) if old else None
            after=metrics(w,method,p,'train')
            history[-1]['actual_full_step_mean_loss_change']=float(after.mean()-before.mean())
    elapsed=time.perf_counter()-start
    mode=core.kind(method)
    r=7 if mode=='independent' else (0 if mode=='shared' else (3 if method=='rank3' else 2))
    # FiLM is not a general affine rank2 family; its SVD2 bound is only a relaxation.
    oc=core.oracle(w['Btrain'],w['Ytrain'],r,private_ids=p.get('ids'))
    train=metrics(w,method,p,'train');val=metrics(w,method,p,'val');test=metrics(w,method,p,'test')
    diag=core.diagnostics(core.gradients(method,p,w['H'],w['Q'])[1])
    row={'family':w['family'],'world':w['world'],'init':init,'method':method,'lr':lr,'steps':steps,
         'train_mse':float(train.mean()),'val_mean_mse':float(val.mean()),
         'test_mean_mse':float(test.mean()),'test_worst_task_mse':float(test.max()),
         'oracle_train_mse':float(np.mean((w['Btrain']@oc.T-w['Ytrain'])**2)),
         'oracle_test_mean_mse':float(np.mean((w['Btest']@oc.T-w['Ytest'])**2)),
         'oracle_bound_type':'rank2_relaxation' if mode=='film' else 'exact_empirical_class_optimum',
         'train_minus_oracle':0.,'train_seconds':elapsed,
         'parameters':sum(v.size for k,v in p.items() if k!='ids'),
         'private_ids':','.join(map(str,allocation)),
         'private_truth':','.join(map(str,w['private_truth'])),
         'private_overlap':len(set(allocation)&set(w['private_truth'])) if w['private_truth'] else None,
         'prefix_hash':prefix_hash,'state_hash':state_hash(p),**diag}
    row['train_minus_oracle']=row['train_mse']-row['oracle_train_mse']
    for t in range(8): row[f'test_task_{t}']=float(test[t])
    return p,row,history

def write_csv(path:Path,rows:list[dict[str,Any]]) -> None:
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)

def source_hashes() -> dict[str,str]:
    paths=[ROOT/'PROTOCOL.json']+sorted((ROOT/'source').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def environment() -> dict[str,Any]:
    cpu=Path('/proc/cpuinfo').read_text()
    return {'python':platform.python_version(),'numpy':np.__version__,'cpu_model':next((l.split(':',1)[1].strip() for l in cpu.splitlines() if l.startswith('model name')),'unknown'),
            'cpu_mhz_snapshots':[float(l.split(':')[1]) for l in cpu.splitlines() if l.startswith('cpu MHz')][:8],
            'clock_locked':False,'threads':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
            'dtype':'float64','gpu_used':False,'platform':platform.platform(),'numpy_blas':str(np.__config__.CONFIG.get('Build Dependencies',{}))}

def run_stage(stage:str) -> None:
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    out=ROOT/'results';out.mkdir(exist_ok=True)
    if stage=='calibration':
        worlds=protocol['calibration_worlds'];inits=protocol['calibration_initializations'];lrs=protocol['learning_rates']
    else:
        lock=json.loads((out/'FRESH_LOCK.json').read_text())
        if lock['source_hashes']!=source_hashes(): raise RuntimeError('science source changed after freeze')
        worlds=protocol['fresh_worlds'];inits=protocol['fresh_initializations'];selected=lock['learning_rates']
    rows=[];hist=[]
    for world in worlds:
        t=time.perf_counter()
        for family in protocol['families']:
            w=core.make_world(family,world)
            if stage=='fresh':
                np.savez(ROOT/'artifacts'/f'world_{world}_{family}.npz',**{k:v for k,v in w.items() if isinstance(v,np.ndarray)})
            for init in inits:
                for method in protocol['methods']:
                    for lr in lrs if stage=='calibration' else [selected[method]]:
                        p,row,h=fit(w,method,init,lr)
                        stem=f'{stage}_{family}_{world}_{init}_{method}_{lr}'
                        path=ROOT/'artifacts'/(stem+'.gic')
                        row['inference_bytes']=core.save_model(path,method,p)
                        row['payload_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                        row['payload']=str(path.relative_to(ROOT))
                        row['stage']=stage
                        rows.append(row)
                        hist.extend({**{k:row[k] for k in ['family','world','init','method','lr','stage']},**z} for z in h)
        write_csv(out/f'{stage.upper()}_RESULTS.csv',rows)
        write_csv(out/f'{stage.upper()}_HISTORY.csv',hist)
        print(json.dumps({'stage':stage,'world_finished':world,'records':len(rows),'seconds':round(time.perf_counter()-t,3)}),flush=True)
    if stage=='calibration':
        selection={};scores={}
        for group,methods in GROUPS.items():
            sc={str(lr):float(np.mean([r['val_mean_mse'] for r in rows if r['method'] in methods and r['lr']==lr])) for lr in lrs}
            best=min(lrs,key=lambda v:(sc[str(v)],v));scores[group]=sc
            for method in methods: selection[method]=best
        lock={'source_hashes':source_hashes(),'learning_rates':selection,'calibration_scores':scores,
              'environment':environment(),'fresh_not_yet_run':True,
              'selection_file_hash':hashlib.sha256((out/'CALIBRATION_RESULTS.csv').read_bytes()).hexdigest()}
        (out/'FRESH_LOCK.json').write_text(json.dumps(lock,indent=2))
        print(json.dumps({'selected':selection,'scores':scores}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['calibration','fresh'],required=True)
    run_stage(ap.parse_args().stage)
