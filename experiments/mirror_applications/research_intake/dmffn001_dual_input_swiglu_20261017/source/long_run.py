"""Post-primary locked LONG stage. New worlds only; source model/train unchanged."""
import argparse,json,time,os,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from train import train_single
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    p=json.loads((ROOT/'LONG_PROTOCOL.json').read_text())
    jobs=[(w,s,f,m,p['training_updates'],p['learning_rate'],str(ROOT/'results'/'long'))
          for w in p['new_worlds'] for s in p['initialization_seeds'] for f in p['conditions'] for m in p['methods']]
    errors=[];completed=0;t0=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs={pool.submit(train_single,j):j for j in jobs}
        for f in as_completed(futs):
            try:
                result=f.result();completed+=1
                if completed%10==0 or completed==len(jobs):print(json.dumps({'done':completed,'total':len(jobs),'elapsed_s':round(time.perf_counter()-t0,1),'latest':result}),flush=True)
            except Exception as e:
                errors.append((futs[f],str(e)));print('ERROR',futs[f],e,flush=True)
    print(json.dumps({'total':len(jobs),'completed':completed,'errors':len(errors),'wall_s':time.perf_counter()-t0}),flush=True)
    if errors:(ROOT/'logs/long_ERRORS.json').write_text(json.dumps(errors,indent=2));sys.exit(1)
if __name__=='__main__':main()
