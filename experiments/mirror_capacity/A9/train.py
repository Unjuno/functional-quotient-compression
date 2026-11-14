# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A9 frozen last-step training, no development/audit evaluation."""
from __future__ import annotations
import argparse,csv,hashlib,itertools,json,os,platform,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from experiments.mirror_capacity.A8 import a8
from experiments.mirror_capacity.A9 import a9

HERE=Path(__file__).resolve().parent

def sha(p:Path):return hashlib.sha256(p.read_bytes()).hexdigest()

def record_one(t):
    w,s,kind,regime,steps,batch,folder=t
    torch.set_num_threads(1)
    data=a8.make_world(w,'ramp')
    t0=time.perf_counter()
    out=a9.fit(data,kind,regime,s,steps=steps,batch=batch)
    sec=time.perf_counter()-t0
    raw=a8.serialize(out['model'])
    name=f'ramp_w{w}_s{s}_{kind}_{regime}.a8w'
    file=Path(folder)/name
    file.write_bytes(raw)
    return {'world':w,'seed':s,'family':'ramp','kind':kind,'regime':regime,
            'width':out['model'].width,'params':a8.num_params(kind,out['model'].width),
            'serialized_bytes':len(raw),'model_file':name,'sha256':hashlib.sha256(raw).hexdigest(),
            'dataset_sha256':data['fingerprint'],
            'train_last_batch_mse':out['train_last_batch_mse'],
            'initial_code_grad_norm':out['initial_code_grad_norm'],
            'initial_active_fraction':out['initial_active_fraction'],
            'train_seconds':sec}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worlds',nargs='+',type=int,default=list(range(820,826)))
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1])
    p.add_argument('--kinds',nargs='+',choices=a9.KINDS,default=list(a9.KINDS))
    p.add_argument('--regimes',nargs='+',choices=a9.REGIMES,default=list(a9.REGIMES))
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--batch',type=int,default=256)
    p.add_argument('--jobs',type=int,default=4)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('refusing to overwrite')
    if a.steps<1 or a.batch<1 or a.jobs<1 or len(set(a.worlds))!=len(a.worlds):raise ValueError('invalid args')
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key]='1'
    a.out.mkdir(parents=True)
    (a.out/'models').mkdir()
    cfg={'worlds':a.worlds,'seeds':a.seeds,'kinds':a.kinds,'regimes':a.regimes,'steps':a.steps,
         'batch':a.batch,'jobs':a.jobs,'optimizer':'AdamW','learning_rate':0.012,
         'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),
         'CPU_frequency_not_locked':True,
         'protocol_sha256':sha(HERE/'PROTOCOL.md'),
         'a9_source_sha256':sha(HERE/'a9.py'),
         'a8_source_sha256':sha(HERE.parent/'A8'/'a8.py'),
         'scope':'CPU synthetic, fixed steps, not matched training FLOPs'}
    (a.out/'config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n')
    jobs=[(*c,a.steps,a.batch,str(a.out/'models')) for c in itertools.product(a.worlds,a.seeds,a.kinds,a.regimes)]
    rows=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futures={pool.submit(record_one,c):c for c in jobs}
        for fut in as_completed(futures):
            rows.append(fut.result())
            if len(rows)%24==0 or len(rows)==len(jobs):
                print('A9_TRAIN',len(rows),'/',len(jobs),'elapsed_seconds',round(time.perf_counter()-start,1),flush=True)
    rows.sort(key=lambda r:(r['world'],r['seed'],r['kind'],r['regime']))
    with (a.out/'train.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lock={'format':'a9-lock-v1','num_models':len(rows),'config':cfg,
          'train_csv_sha256':sha(a.out/'train.csv'),
          'models':{r['model_file']:{'sha256':r['sha256'],'bytes':r['serialized_bytes'],
                                     'dataset_sha256':r['dataset_sha256']} for r in rows}}
    (a.out/'train.lock.json').write_text(json.dumps(lock,indent=2,ensure_ascii=False)+'\n')
    print('A9_TRAIN_LOCKED',len(rows),flush=True)

if __name__=='__main__':main()
