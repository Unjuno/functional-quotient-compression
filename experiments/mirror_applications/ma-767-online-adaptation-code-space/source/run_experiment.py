#!/usr/bin/env python3
"""Online task adaptation in a compact functional-code space vs full weights."""
from __future__ import annotations
import csv,hashlib,io,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SEED=76701; D=16; R=2; TASKS=16; SUPPORTS=(4,8,16); TEST=512
def pack(**a):
 b=io.BytesIO();np.savez_compressed(b,**a);return b.getvalue()

def run():
 rng=np.random.default_rng(SEED)
 w0=rng.normal(scale=.3,size=D).astype(np.float32)
 basis,_=np.linalg.qr(rng.normal(size=(D,R)))
 basis=basis.astype(np.float32)
 true_codes=rng.normal(scale=.45,size=(TASKS,R)).astype(np.float32)
 weights=w0[None,:]+true_codes@basis.T
 xt=rng.normal(size=(TASKS,TEST,D)).astype(np.float32)
 yt=np.einsum('td,tnd->tn',weights,xt)
 rows=[]
 for k in SUPPORTS:
  x=rng.normal(size=(TASKS,k,D)).astype(np.float32)
  y=np.einsum('td,tnd->tn',weights,x)
  code_est=[];full_est=[];ops_code=ops_full=0
  for t in range(TASKS):
   z=x[t]@basis;res=y[t]-x[t]@w0
   code_est.append(np.linalg.lstsq(z,res,rcond=None)[0])
   full_est.append(w0+np.linalg.lstsq(x[t],res,rcond=None)[0])
   ops_code+=k*D*R+R**3;ops_full+=k*D*D+D**3
  code_est=np.asarray(code_est,np.float32);full_est=np.asarray(full_est,np.float32)
  blobs={
   'shared_base_no_adapt':pack(base=w0),
   'full_task_weight_copies':pack(task_weights=full_est,task_ids=np.arange(TASKS,dtype=np.int8)),
   'native_lowrank_adapter_bank':pack(base=w0,shared_basis=basis,task_codes=code_est,task_ids=np.arange(TASKS,dtype=np.int8)),
   'mirror_functional_code_bank':pack(base=w0,shared_basis=basis,task_codes=code_est,task_ids=np.arange(TASKS,dtype=np.int8))}
  preds={
   'shared_base_no_adapt':np.broadcast_to(w0,(TASKS,D)),
   'full_task_weight_copies':full_est,
   'native_lowrank_adapter_bank':w0[None,:]+code_est@basis.T,
   'mirror_functional_code_bank':w0[None,:]+code_est@basis.T}
  for name,blob in blobs.items():
   pred=np.einsum('td,tnd->tn',preds[name],xt)
   mse=np.mean((pred-yt)**2);norm=np.sqrt(mse/np.mean(yt**2))
   ops=0 if name=='shared_base_no_adapt' else (ops_full if name=='full_task_weight_copies' else ops_code)
   dynamic_macs=TASKS*TEST*(D+D*R+R if name in ('native_lowrank_adapter_bank','mirror_functional_code_bank') else D)
   precompose=TASKS*D*R if name in ('native_lowrank_adapter_bank','mirror_functional_code_bank') else 0
   rows.append(dict(seed=SEED,support_examples_per_task=k,method=name,serialized_bytes=len(blob),payload_sha256=hashlib.sha256(blob).hexdigest(),
    mean_test_mse=float(mse),relative_rmse=float(norm),retained_tasks=TASKS,adaptation_examples=TASKS*k,adaptation_ops=ops+precompose,
    precompose_macs=precompose,dynamic_code_inference_macs=dynamic_macs,precomposed_dense_inference_macs=TASKS*TEST*D,updates=0,
    status_note='exact native low-rank alias' if name=='mirror_functional_code_bank' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('tasks',TASKS,'supports',SUPPORTS,'dimensions',D,'code rank',R)
if __name__=='__main__':run()
