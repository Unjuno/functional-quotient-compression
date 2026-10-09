#!/usr/bin/env python3
import argparse,csv,io,json,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SEEDS=(55801,55802);L=3;W=(2,4,8);HOLD=((0,0,2),(0,2,0),(1,0,2),(1,2,0),(2,0,2),(2,2,0))
def pack(a):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_STORED) as z:
  for k in sorted(a):
   q=io.BytesIO();np.save(q,a[k],allow_pickle=False);z.writestr(zipfile.ZipInfo(k+'.npy',(1980,1,1,0,0,0)),q.getvalue())
 return b.getvalue()
def run(seed):
 rng=np.random.default_rng(seed);base=rng.normal(size=(8,8)).astype(np.float32);lf=rng.normal(size=(L,8)).astype(np.float32);wf=rng.normal(size=(3,8)).astype(np.float32)
 target={}
 for a in range(3):
  for b in range(3):
   for c in range(3):
    cfg=(a,b,c); mats=[base[:W[g],:W[g]]+np.diag(lf[i,:W[g]]*wf[g,:W[g]]) for i,g in enumerate(cfg)]
    target[cfg]=mats
 # bundle controls
 flatarr={'meta':np.frombuffer(b'flat',dtype=np.uint8)}
 for i,(k,vs) in enumerate(target.items()):
  for j,v in enumerate(vs):flatarr[f'cfg{i}_{j}']=v
 flat=pack(flatarr)
 shared=pack({'base':base,'meta':np.frombuffer(b'native-mix',dtype=np.uint8)})
 fac=pack({'base':base,'layer':lf,'width':wf,'meta':np.frombuffer(b'factorized',dtype=np.uint8)})
 rows=[]
 for method in ('native_prefix','flat_table','mirror_factorized','direct_factorized'):
  errs=[];distinct=[]
  for cfg in HOLD:
   true=target[cfg]
   if method=='native_prefix':pred=[base[:W[g],:W[g]] for g in cfg]
   elif method=='flat_table':pred=true
   else:pred=[base[:W[g],:W[g]]+np.diag(lf[i,:W[g]]*wf[g,:W[g]]) for i,g in enumerate(cfg)]
   errs.append(sum(np.mean((p-t)**2) for p,t in zip(pred,true))/max(sum(np.mean(t**2) for t in true),1e-12));distinct.append(b''.join(np.round(p,5).tobytes() for p in pred))
  b={'native_prefix':len(shared),'flat_table':len(flat),'mirror_factorized':len(fac),'direct_factorized':len(fac)}[method]
  rows.append({'world':seed,'method':method,'serialized_bytes':b,'train_examples':0,'optimizer_updates':0,'active_compute_proxy':len(HOLD)*8*8*sum(W[g] for cfg in HOLD for g in cfg),'wall_time_s':'0','heldout_nmse':f'{np.mean(errs):.9g}','distinct_functions':len(set(distinct)),'status_note':'oracle aligned layer-width composition'})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'artifacts'/'dev_results.csv'));a=ap.parse_args();r=[]
 for x in SEEDS:r+=run(x)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=r[0].keys());w.writeheader();w.writerows(r)
 print(json.dumps({'rows':len(r)},indent=2))
if __name__=='__main__':main()
