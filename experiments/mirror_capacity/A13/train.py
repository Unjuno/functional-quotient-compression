# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A13 search stage. Writes and cryptographically freezes *all* models before audit.

Example: python train.py --a12-zip /path/mirror_capacity_a12_20261115.zip --out output
For a smoke test use --smoke with an entirely separate empty directory.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
from a13 import (load_archived_parent,make_world,make_combined_model,
                 score_support,es_optimize,random_optimize,adam_optimize,
                 local_jacobian_singular_values,serialize,POP,GENERATIONS,SEARCH_BOUND)

ROOT=Path(__file__).resolve().parent


def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()


def _one(args):
    zip_file,out,fam,world_id,seed,method,smoke=args
    torch.set_num_threads(1)
    start_time=time.perf_counter()
    m,raw,original=load_archived_parent(zip_file,fam,world_id,seed,method)
    w=make_world(world_id,fam)
    if len(w['support_x'])!=8 or w['support_x'].shape[1]!=32:
        raise RuntimeError('unsupported world support grain')
    pop,gens=(8,3) if smoke else (POP,GENERATIONS)
    budget=pop*gens
    final={'es':[],'random':[]}
    diagnostics=[];trace=[]
    for t in range(8):
        sx,sy=w['support_x'][t],w['support_y'][t]
        start=m.new_codes[t].detach().numpy().copy()
        seed_es=40000011+world_id*1973+seed*17+('mirror','shift','film').index(method)*101+t*109
        es=es_optimize(m,sx,sy,start,seed=seed_es,pop=pop,generations=gens)
        rnd=random_optimize(m,sx,sy,start,seed=seed_es+190001,evaluations=budget)
        final['es'].append(es['code'])
        final['random'].append(rnd['code'])
        smin,smax=local_jacobian_singular_values(m,sx,start)
        diagnostics.append(dict(family=fam,world=world_id,seed=seed,method=method,task=t,
            start_c0=float(start[0]),start_c1=float(start[1]),
            start_support_mse=score_support(m,sx,sy,start),
            es_support_mse=float(es['best_support_mse']),random_support_mse=float(rnd['best_support_mse']),
            es_c0=float(es['code'][0]),es_c1=float(es['code'][1]),
            random_c0=float(rnd['code'][0]),random_c1=float(rnd['code'][1]),
            es_bound_hit_rate=float(es['bound_hit_rate']),
            jac_s_min=smin,jac_s_max=smax,
            es_evaluations=es['evaluations'],random_evaluations=rnd['evaluations']))
        trace.append({'task':t,'es':es['trace']})
    # Adam optimization uses only the same training support tensor/labels.
    t1=time.perf_counter()
    adam=adam_optimize(m,w['support_x'],w['support_y'],steps=budget)
    t_adam=time.perf_counter()-t1
    for d in diagnostics:
        d['adam_support_mse']=float(adam['support_mses'][d['task']])
        d['adam_c0']=float(adam['codes'][d['task'],0]);d['adam_c1']=float(adam['codes'][d['task'],1])
    final['adam']=adam['codes']
    key=f'{fam}_w{world_id}_s{seed}_{method}'
    rows=[]
    for mode in ('es','random','adam'):
        model=make_combined_model(m,np.asarray(final[mode]))
        payload=serialize(model)
        p=Path(out)/'models'/f'{key}_{mode}.a11w'
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(payload)
        if p.stat().st_size!=len(payload):raise RuntimeError('model byte count mismatch')
        rows.append(dict(family=fam,world=world_id,seed=seed,method=method,algorithm=mode,
            path=str(p.relative_to(out)),stored_bytes=len(payload),sha256=sha(payload),
            parent_sha256=original['sha256'],world_fingerprint=w['fingerprint'],
            search_evaluations=budget,support_example_forwards=budget*8*32,
            target_codes=8,code_dim=2,code_min=float(np.min(final[mode])),
            code_max=float(np.max(final[mode])),code_l2_mean=float(np.linalg.norm(final[mode],axis=1).mean()),
            support_mse_mean=float(np.mean([score_support(m,w['support_x'][t],w['support_y'][t],final[mode][t]) for t in range(8)]))))
    trace_path=Path(out)/'traces'/f'{key}.json'
    trace_path.parent.mkdir(exist_ok=True)
    trace_payload=json.dumps({'family':fam,'world':world_id,'seed':seed,'method':method,'es':trace,'adam':adam['trace']},sort_keys=True).encode()
    with trace_path.open('xb') as f:f.write(trace_payload)
    parent_path=Path(out)/'parents'/f'{key}_meta2400.a11w'
    parent_path.parent.mkdir(exist_ok=True)
    with parent_path.open('xb') as f:f.write(raw)
    return dict(rows=rows,diagnostics=diagnostics,
                trace=dict(path=str(trace_path.relative_to(out)),sha256=sha(trace_payload)),
                parent=dict(path=str(parent_path.relative_to(out)),sha256=sha(raw),stored_bytes=len(raw)),
                family=fam,world=world_id,seed=seed,method=method,
                elapsed_s=time.perf_counter()-start_time,adam_s=t_adam)


def run(out:Path,a12_zip:Path,workers:int=4,smoke:bool=False)->dict:
    out=Path(out).resolve()
    if out.exists():
        if any(out.iterdir()):raise FileExistsError('refusing to overwrite a nonempty experiment directory')
    else:out.mkdir(parents=True)
    if not a12_zip.is_file():raise FileNotFoundError(a12_zip)
    import a13
    world_ids=(1200,) if smoke else (1200,1201,1202)
    seeds=(0,) if smoke else (0,1)
    families=('tent',) if smoke else ('tent','sine')
    methods=('mirror',) if smoke else ('mirror','shift','film')
    config=dict(experiment='A13_SMOKE' if smoke else 'A13_ES',source_a12_zip_sha256=sha(a12_zip.read_bytes()),
                family=list(families),worlds=list(world_ids),seeds=list(seeds),methods=list(methods),
                pop=8 if smoke else POP,generations=3 if smoke else GENERATIONS,code_domain=[-SEARCH_BOUND,SEARCH_BOUND],
                es_elite=8,es_sigma_init=1.35,es_sigma_floor=.08,es_sigma_ceiling=2.50,es_alpha=.55,
                random_control='independent uniform box, same population candidate-evaluations',
                adam='960 or 24 full support steps, lr0.01, no weight decay',
                support_per_new_task=32,new_tasks=8,parent_meta_steps=2400,capacity_limit_bytes=1024,
                optimizer_selection_data='support only; no old dev or new IID/OOD',
                source_sha256={k:sha((ROOT/k).read_bytes()) for k in ('PROTOCOL.md','a13.py','train.py')},
                inherited_a11_sha256=sha((a13.A11_DIR/'a11.py').read_bytes()),
                environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                                 machine=platform.machine(),processor=platform.processor(),cpu_count=os.cpu_count(),
                                 torch_threads=1,worker_processes=workers))
    cfg_bytes=json.dumps(config,sort_keys=True,indent=2).encode()
    (out/'config.json').write_bytes(cfg_bytes)
    inputs=[(str(a12_zip),str(out),fam,w,s,m,smoke)
            for fam,w,s,m in itertools.product(families,world_ids,seeds,methods)]
    results=[];tic=time.perf_counter()
    if workers==1:
        for args in inputs:
            r=_one(args);results.append(r)
            print('TRAIN',r['family'],r['world'],r['seed'],r['method'],f"{r['elapsed_s']:.2f}s",flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future=[pool.submit(_one,args) for args in inputs]
            for f in as_completed(future):
                r=f.result();results.append(r)
                print('TRAIN',len(results),'/',len(inputs),r['family'],r['world'],r['seed'],r['method'],f"{r['elapsed_s']:.2f}s",flush=True)
    results.sort(key=lambda x:(x['family'],x['world'],x['seed'],x['method']))
    rows=[z for r in results for z in r['rows']]
    diags=[z for r in results for z in r['diagnostics']]
    expected=3*len(inputs)
    if len(rows)!=expected or len({r['path'] for r in rows})!=expected:raise RuntimeError('incomplete model ledger')
    with (out/'policy_diagnostics.csv').open('x',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=diags[0].keys());wr.writeheader();wr.writerows(diags)
    ledger=dict(experiment=config['experiment'],config_sha256=sha(cfg_bytes),
                source_sha256=config['source_sha256'],
                models=rows,parents=[r['parent'] for r in results],
                traces=[r['trace'] for r in results],
                setup=[{k:r[k] for k in ('family','world','seed','method','elapsed_s','adam_s')} for r in results],
                training_wall_seconds=time.perf_counter()-tic)
    (out/'train.lock.json').write_text(json.dumps(ledger,indent=2,sort_keys=True))
    print('TRAIN_LOCKED',len(rows),'files',len(diags),'task-policies',len(results),'parent models', 'seconds',ledger['training_wall_seconds'],flush=True)
    return ledger


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--a12-zip',type=Path,default=Path('/mnt/data/mirror_capacity_a12_20261115.zip'))
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    run(args.out,args.a12_zip,args.workers,args.smoke)
