#!/usr/bin/env python3
"""MA-556 sparse-private residual frontier screen."""
import argparse,csv,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SEEDS=(55601,55602);RHOS=(0.0,.1,.3,.6);FRACS=(0,.05,.1,.25,.5,1.0);TASKS=8;D=8

def pack(a):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_STORED) as z:
  for k in sorted(a):
   q=io.BytesIO();np.save(q,a[k],allow_pickle=False);z.writestr(zipfile.ZipInfo(k+'.npy',(1980,1,1,0,0,0)),q.getvalue())
 return b.getvalue()
def run(seed):
 rng=np.random.default_rng(seed);base=rng.normal(size=(D,D));basis=rng.normal(size=(2,D,D));coeff=rng.normal(size=(TASKS,2));rows=[]
 for rho in RHOS:
  target=base[None,:,:]+np.einsum('tk,kij->tij',coeff,basis)
  # Each task has rho fraction of entries with private deviations; amplitude fixed to make heterogeneity explicit.
  mask=rng.random(target.shape)<rho;noise=rng.normal(0,.45,target.shape)*mask;target=target+noise
  # Best rank-2 shared basis by SVD, fitted from the complete bank (oracle mechanism screen).
  centered=target.reshape(TASKS,-1)-target.mean(axis=0).reshape(1,-1);u,s,v=np.linalg.svd(centered,full_matrices=False);B=(s[:2,None]*v[:2]).reshape(2,D,D);C=u[:,:2]
  pred=target.mean(axis=0)[None,:,:]+np.einsum('tk,kij->tij',C,B);res=target-pred
  start=time.perf_counter()
  for method in ('shared_base','mirror_rank2','direct_coeff','private_sparse','independent'):
   for frac in (FRACS if method=='private_sparse' else (0,)):
    a={'base':target.mean(axis=0).astype(np.float32),'metadata':np.frombuffer(method.encode(),dtype=np.uint8)}
    if method in ('mirror_rank2','direct_coeff','private_sparse'):
     a.update({'basis':B.astype(np.float32),'codes':C.astype(np.float32)})
    if method=='private_sparse' and frac>0:
     k=max(1,int(frac*res.size));idx=np.argpartition(np.abs(res).ravel(),-k)[-k:];a['private_indices']=idx.astype(np.uint16);a['private_values']=res.ravel()[idx].astype(np.float32)
    if method=='independent':a={'operators':target.astype(np.float32),'metadata':np.frombuffer(b'independent',dtype=np.uint8)}
    payload=pack(a)
    if method=='shared_base':rec=np.repeat(target.mean(axis=0)[None,:,:],TASKS,axis=0)
    elif method=='independent':rec=target.copy()
    else:
     rec=pred.copy()
     if method=='private_sparse' and frac>0:
      flat=rec.reshape(-1);flat[idx]+=res.ravel()[idx]
    nmse=float(np.mean((rec-target)**2)/max(np.mean(target**2),1e-12));macs=TASKS*D*D*(1 if method in ('shared_base','independent') else 3)+(len(a.get('private_indices',[]))*2)
    rows.append({'world':seed,'heterogeneity':rho,'method':method,'private_fraction':frac,'serialized_bytes':len(payload),'train_examples':0,'optimizer_updates':0,'active_compute_proxy':macs,'wall_time_s':f'{time.perf_counter()-start:.6f}','nmse':f'{nmse:.9g}','status_note':'oracle SVD shared basis; private entries selected by magnitude'})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'artifacts'/'dev_results.csv'));a=ap.parse_args();r=[]
 for s in SEEDS:r+=run(s)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=r[0].keys());w.writeheader();w.writerows(r)
 print(json.dumps({'rows':len(r)},indent=2))
if __name__=='__main__':main()
