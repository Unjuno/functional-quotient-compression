#!/usr/bin/env python3
"""Synthetic Fisher-weighted task merging in full parameter and shared-code spaces."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D,T,R,Q,K=256,8,8,16,4
SEEDS={'dev-71401':71401,'audit-71411':71411,'audit-71412':71412,'audit-71413':71413}
RHOS=(0.0,0.1,0.25)
def pack(**a):
 b=io.BytesIO();np.savez(b,**a);return b.getvalue()
def fisher_merge(ds,fs,ids):
 f=fs[ids];d=ds[ids];return (f*d).sum(0)/np.maximum(f.sum(0),1e-30)
def code_fisher(B,ds,fs,ids):
 A=np.zeros((R,R));b=np.zeros(R)
 for i in ids:
  f=fs[i];A+=B.T@(f[:,None]*B);b+=B.T@(f*ds[i])
 return np.linalg.solve(A+1e-10*np.eye(R),b)
def merge_loss(x,ds,fs,ids):
 num=den=0.0
 for i in ids:num+=float(np.sum(fs[i]*(x-ds[i])**2));den+=float(np.sum(fs[i]*ds[i]**2))
 return num/max(den,1e-30)
def nrmse(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
def run_one(label,seed,outdir,rhos=RHOS):
 rng=np.random.default_rng(seed);outdir.mkdir(parents=True,exist_ok=True);rows=[]
 B,_=np.linalg.qr(rng.normal(size=(D,R)));base=rng.normal(size=D).astype(np.float32)*.02
 requests=np.stack([rng.choice(T,size=K,replace=False) for _ in range(Q)]).astype(np.int16)
 for rho in rhos:
  codes=rng.normal(size=(T,R));ds=codes@B.T
  if rho:
   private=rng.normal(size=(T,D));private*=rho*np.linalg.norm(ds,axis=1,keepdims=True)/np.maximum(np.linalg.norm(private,axis=1,keepdims=True),1e-30);ds+=private
  fisher=np.exp(rng.normal(0,1.0,size=(T,D)))
  full=[];fcode=[];ucode=[];meanfull=[];t=time.perf_counter()
  for ids in requests:
   full.append(fisher_merge(ds,fisher,ids));fcode.append(code_fisher(B,ds,fisher,ids));ucode.append(codes[ids].mean(0));meanfull.append(ds[ids].mean(0))
  solve_wall=time.perf_counter()-t;full=np.stack(full);fcode=np.stack(fcode);ucode=np.stack(ucode);meanfull=np.stack(meanfull)
  payloads={'full_parameter_Fisher':pack(base=base,deltas=ds.astype(np.float32),fisher=fisher.astype(np.float32),requests=requests,merged=full.astype(np.float32),metadata=np.asarray([D,T,Q],np.int16)),'task_arithmetic_mean':pack(base=base,deltas=ds.astype(np.float32),requests=requests,merged=meanfull.astype(np.float32),metadata=np.asarray([D,T,Q],np.int16)),'Mirror_Fisher_code':pack(base=base,basis=B.astype(np.float32),task_codes=codes.astype(np.float32),fisher=fisher.astype(np.float32),requests=requests,merged_codes=fcode.astype(np.float32),metadata=np.asarray([D,T,R,Q],np.int16)),'native_projected_Fisher_code':pack(base=base,basis=B.astype(np.float32),task_codes=codes.astype(np.float32),fisher=fisher.astype(np.float32),requests=requests,merged_codes=fcode.astype(np.float32),metadata=np.asarray([D,T,R,Q],np.int16)),'unweighted_code_average':pack(base=base,basis=B.astype(np.float32),task_codes=codes.astype(np.float32),requests=requests,merged_codes=ucode.astype(np.float32),metadata=np.asarray([D,T,R,Q],np.int16))}
  for method,payload in payloads.items():
   (outdir/f'{label}_rho{rho:g}.{method}.npz').write_bytes(payload)
   with np.load(io.BytesIO(payload),allow_pickle=False) as p:
    if method in ('full_parameter_Fisher','task_arithmetic_mean'):merged=p['merged']
    else:merged=p['merged_codes']@p['basis'].T
   losses=[];errors=[]
   for q,ids in enumerate(requests):losses.append(merge_loss(merged[q],ds,fisher,ids));errors.append(nrmse(merged[q],full[q]))
   ops=R*D+R*R*K if 'code' in method.lower() else D*K
   rows.append(dict(condition=label,seed=seed,private_fraction=rho,method=method,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),mean_normalized_fisher_loss=float(np.mean(losses)),max_normalized_fisher_loss=float(np.max(losses)),nrmse_to_full_fisher_merge=float(np.mean(errors)),merge_requests=Q,tasks_per_request=K,examples=Q*K,optimizer_updates=0,solve_wall_s=solve_wall,active_compute_proxy=ops,status_note='synthetic diagonal Fisher; complete source-plus-output state serialized'))
 return rows
def main():
 rows=[]
 for label,seed in SEEDS.items():rows.extend(run_one(label,seed,ROOT/'source'/'payloads'))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('wrote',len(rows),'model-level rows')
if __name__=='__main__':main()
