"""Frozen, paired CPU grokking sweeps. One JSON+payload per independent fit."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,platform,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import torch
from model import Config
from train import train_one,source_sha

BASE=Path(__file__).resolve().parent
PLAN=[('mirror',0.5),('gain',0.5),('pair_mirror',0.5),('independent',0.5),('mirror',0.0)]
WORLDS=[87011,87012,87013]
SEEDS=[101,202]
CFG=Config(p=31,h=64,q=16)
STEPS=8000
LR=0.005
FRAC=0.60
INTERVAL=200


def worker(job):
    world,seed,index=job
    torch.set_num_threads(1)
    method,wd=PLAN[index]
    result=train_one(world,seed,method,CFG,LR,wd,FRAC,STEPS,INTERVAL,True)
    return job,result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=3)
    a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    proof={'training_hashes':source_sha(),'batch_py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'config':CFG.__dict__,'steps':STEPS,'lr':LR,'train_fraction':FRAC,'interval':INTERVAL,
           'worlds':WORLDS,'initialization_seeds':SEEDS,'method_weight_decay':PLAN,
           'dtype':'float32','device':'cpu','python':platform.python_version(),'torch':torch.__version__,
           'world_unit':'split of same finite modular addition universe','eval_rule':'checkpoint at fixed steps, no test-based stopping'}
    (out/'setup.json').write_text(json.dumps(proof,indent=2)+'\n')
    jobs=[(w,s,j) for w in WORLDS for s in SEEDS for j in range(len(PLAN))]
    t0=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs={pool.submit(worker,job):job for job in jobs}
        for fut in as_completed(futs):
            job,result=fut.result()
            world,seed,index=job;method,wd=PLAN[index]
            lbl='mirror_no_decay' if method=='mirror' and wd==0.0 else method
            raw=result.pop('raw')
            (out/f'model_{world}_{seed}_{lbl}.pt').write_bytes(raw)
            history=result.pop('history')
            (out/f'history_{world}_{seed}_{lbl}.json').write_text(json.dumps(history,separators=(',',':'))+'\n')
            result['label']=lbl;result['weight_decay']=wd
            (out/f'result_{world}_{seed}_{lbl}.json').write_text(json.dumps(result,indent=2)+'\n')
            print(f'finished {world}/{seed}/{lbl} train={result["final_train_accuracy"]:.3f} test={result["final_test_accuracy"]:.3f} elapsed={result["elapsed_seconds"]:.1f}s [total wall {time.perf_counter()-t0:.1f}s]',flush=True)
    print('COMPLETE',len(jobs),'fits; wall',time.perf_counter()-t0)
if __name__=='__main__':main()
