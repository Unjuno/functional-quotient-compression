"""Run locked GIC002 batches; no test-driven hyperparameter selection."""
import argparse,datetime,json,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from engine import ROOT,CFG,fit,world,sha

def choose_lr(rows):
    return min(sorted({r['lr'] for r in rows}),key=lambda lr:np.mean([r['val_noisy_mse']/r['validation_scale'] for r in rows if r['lr']==lr]))

def task(job):
    seed,family,n,method,inits,lr,stage=job
    out=ROOT/'results'/stage/f'w{seed}_{family}_n{n}_lr{lr}'
    f=out/f'{method}.json'
    if f.exists():r=json.loads(f.read_text())
    else:r=fit(seed,family,n,method,inits,lr,out)
    scale=float(np.mean(world(seed,family)['val_y']**2))
    for row in r['records']:row['validation_scale']=scale
    return job,r['records']

def main(stage,workers):
    lock=json.loads((ROOT/'PRE_RUN_LOCK.json').read_text())
    for p,h in lock['files'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('source drift: '+p)
    if stage=='calibration':ws=CFG['development_worlds'];seeds=CFG['development_inits'];lrs=CFG['learning_rates']
    else:
        frozen=json.loads((ROOT/'FRESH_LOCK.json').read_text())
        ws=CFG['fresh_worlds'];seeds=CFG['fresh_inits'];lrs=[frozen['learning_rate']]
    jobs=[(w,f,n,m,seeds,lr,stage) for w in ws for f in CFG['families'] for n in CFG['train_counts'] for m in CFG['methods'] for lr in lrs]
    records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(task,j) for j in jobs]
        for i,fut in enumerate(as_completed(futures),1):
            job,rows=fut.result();records.extend(rows)
            print(f'{stage} {i}/{len(jobs)} w={job[0]} {job[1]} n={job[2]} {job[3]} lr={job[5]} val={np.mean([r["val_noisy_mse"] for r in rows]):.6g}',flush=True)
    records.sort(key=lambda x:(x['world'],x['family'],x['ntrain'],x['method'],x['lr'],x['init']))
    (ROOT/f'{stage.upper()}_RECORDS.json').write_text(json.dumps(records,sort_keys=True,indent=2))
    if stage=='calibration':
        best=choose_lr(records)
        scores={str(lr):float(np.mean([r['val_noisy_mse']/r['validation_scale'] for r in records if r['lr']==lr])) for lr in lrs}
        obj={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'learning_rate':best,'validation_scores':scores,'record_sha256':sha(ROOT/'CALIBRATION_RECORDS.json'),'source_lock_sha256':sha(ROOT/'PRE_RUN_LOCK.json'),'fresh_generated':False}
        (ROOT/'FRESH_LOCK.json').write_text(json.dumps(obj,sort_keys=True,indent=2))
        print('FRESH LOCK',json.dumps(obj),flush=True)
    print('COMPLETE',stage,len(records),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['calibration','fresh']);ap.add_argument('--workers',type=int,default=2)
    a=ap.parse_args();main(a.stage,a.workers)
