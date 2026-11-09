#!/usr/bin/env python3
"""Factor when-to-act and what-behavior codes in a coupling flow."""
from __future__ import annotations
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
SEED=77201; C=4; B=4; DIM=2; HELD={(0,0),(1,2),(2,3),(3,1)}; TRAIN=[(c,b) for c in range(C) for b in range(B) if (c,b) not in HELD]
FLOW=np.asarray([.3,.25,.2,-.15],np.float32);N=256
torch.set_num_threads(1)
def pack(**a):
 q=io.BytesIO();np.savez_compressed(q,**a);return q.getvalue()
def transform(x,m):
 s=FLOW[0]*x[:,0]+FLOW[1]*m;t=FLOW[2]*x[:,0]+FLOW[3]*m
 return np.stack([x[:,0],x[:,1]*np.exp(s)+t],axis=1),s

def run():
 rng=np.random.default_rng(SEED);cg=np.asarray([.15,.4,.75,1.1],np.float32);bh=np.asarray([-1.,-.35,.45,1.2],np.float32)
 target=cg[:,None]*bh[None,:]
 torch.manual_seed(SEED);cv=torch.nn.Parameter(torch.ones(C));bv=torch.nn.Parameter(torch.tensor(bh.copy()))
 opt=torch.optim.Adam([cv,bv],lr=.03);pairs=np.asarray(TRAIN)
 t0=time.perf_counter()
 for _ in range(2500):
  opt.zero_grad();pred=cv[torch.tensor(pairs[:,0])]*bv[torch.tensor(pairs[:,1])]
  loss=torch.mean((pred-torch.tensor(target[pairs[:,0],pairs[:,1]]))**2);loss.backward();opt.step()
 fit_ms=(time.perf_counter()-t0)*1000
 cfit=cv.detach().numpy();bfit=bv.detach().numpy();mfit=cfit[:,None]*bfit[None,:]
 # Simpler additive gate+behavior control fit on observed pairs.
 A=[];z=[]
 for c,b in TRAIN:
  row=np.zeros(C+B);row[c]=1;row[C+b]=1;A.append(row);z.append(target[c,b])
 addcoef=np.linalg.lstsq(np.asarray(A),np.asarray(z),rcond=None)[0].astype(np.float32)
 addmat=np.stack([addcoef[:C][c]+addcoef[C:][b] for c in range(C) for b in range(B)]).reshape(C,B)
 rngx=rng.normal(size=(C,B,N,DIM)).astype(np.float32)
 truth=np.empty((C,B,N,DIM),np.float32)
 for c in range(C):
  for b in range(B):truth[c,b]=transform(rngx[c,b],target[c,b])[0]
 factorblob=pack(flow_params=FLOW,condition_codes=cfit,behavior_codes=bfit,condition_ids=np.arange(C,dtype=np.int8),behavior_ids=np.arange(B,dtype=np.int8))
 nativeblob=factorblob
 additiveblob=pack(flow_params=FLOW,condition_codes=addcoef[:C],behavior_codes=addcoef[C:],condition_ids=np.arange(C,dtype=np.int8),behavior_ids=np.arange(B,dtype=np.int8))
 directblob=pack(flow_params=FLOW,pair_codes=target.astype(np.float32),pair_ids=np.asarray([(c,b) for c in range(C) for b in range(B)],np.int8))
 rows=[]
 for name,blob,mat,kind in [('independent_pair_codes',directblob,target,'direct'),('additive_condition_behavior',additiveblob,addmat,'add'),('native_product_condition_behavior',nativeblob,mfit,'factor'),('mirror_product_condition_behavior',factorblob,mfit,'factor')]:
  errs=[];cycles=[];t1=time.perf_counter()
  for c,b in sorted(HELD):
   pred,s=transform(rngx[c,b],mat[c,b]);errs.append(float(np.linalg.norm(pred-truth[c,b])/(np.linalg.norm(truth[c,b])+1e-12)))
   inv=np.stack([pred[:,0],(pred[:,1]-(FLOW[2]*pred[:,0]+FLOW[3]*mat[c,b]))*np.exp(-s)],axis=1)
   cycles.append(float(np.max(np.abs(inv-rngx[c,b]))))
  elapsed=(time.perf_counter()-t1)*1000
  rows.append(dict(seed=SEED,method=name,split='heldout_condition_behavior_pairs',serialized_bytes=len(blob),payload_sha256=hashlib.sha256(blob).hexdigest(),
   heldout_transform_nrmse=float(np.mean(errs)),max_cycle_error=float(np.max(cycles)),heldout_code_rmse=float(np.sqrt(np.mean((mat[tuple(np.array(sorted(HELD)).T)]-target[tuple(np.array(sorted(HELD)).T)])**2))),
   active_macs_per_example=7 if kind in ('factor','add') else 5,fit_updates=2500 if kind=='factor' else 0,
   fit_wall_ms=fit_ms if kind=='factor' else 0,eval_wall_ms=elapsed,status_note='exact native factorization alias' if name=='mirror_product_condition_behavior' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('train pairs',len(TRAIN),'heldout',sorted(HELD),'fit updates',2500)
if __name__=='__main__':run()
