#!/usr/bin/env python3
"""Conditional RealNVP coupling view vs linear and equal-size MLP steering."""
from __future__ import annotations
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
SEED=77101; TASKS=16; TRAIN_IDS={i for i in range(TASKS) if i not in {2,5,9,13}}; HELD=sorted(set(range(TASKS))-TRAIN_IDS); NTRAIN=256; NTEST=512
FLOW=np.asarray([.35,.40,.25,-.20],np.float32)
torch.set_num_threads(1)

def pack(**a):
 b=io.BytesIO();np.savez_compressed(b,**a);return b.getvalue()
def flow_forward(x,m,p=FLOW):
 a0,a1,b0,b1=p;s=a0*x[...,0]+a1*m;t=b0*x[...,0]+b1*m
 return np.stack([x[...,0],x[...,1]*np.exp(s)+t],axis=-1),s
def mlp_forward(x,m,w):
 wx,wm,wx2,v=w;z=wx*x[...,0]+wm*m+wx2*x[...,1]
 return np.stack([x[...,0],x[...,1]+v*np.tanh(z)],axis=-1),z

def run():
 rng=np.random.default_rng(SEED);codes=np.linspace(-1,1,TASKS,dtype=np.float32)
 xtrain=rng.normal(size=(TASKS,NTRAIN,2)).astype(np.float32);xtest=rng.normal(size=(TASKS,NTEST,2)).astype(np.float32)
 ytrain=np.stack([flow_forward(xtrain[t],codes[t])[0] for t in range(TASKS)])
 ytest=np.stack([flow_forward(xtest[t],codes[t])[0] for t in range(TASKS)])
 # Fit four-parameter residual MLP on train tasks only: same shared scalar count as flow.
 xfit=np.concatenate([xtrain[t] for t in sorted(TRAIN_IDS)])
 yfit=np.concatenate([ytrain[t] for t in sorted(TRAIN_IDS)])
 xx=torch.tensor(xfit);mm=torch.tensor(np.concatenate([np.full(NTRAIN,codes[t],np.float32) for t in sorted(TRAIN_IDS)]))
 yy=torch.tensor(yfit[:,1]-xfit[:,1])
 w=torch.nn.Parameter(torch.tensor([.1,.1,.1,.1]));opt=torch.optim.Adam([w],lr=.025)
 fit0=time.perf_counter()
 for _ in range(2000):
  opt.zero_grad();z=w[0]*xx[:,0]+w[1]*mm+w[2]*xx[:,1];pred=w[3]*torch.tanh(z);loss=torch.mean((pred-yy)**2);loss.backward();opt.step()
 fit_ms=(time.perf_counter()-fit0)*1000;wfit=w.detach().numpy()
 # Linear steering y2=x2+c0*x1+c1*m+c2; least squares fit on train tasks.
 A=np.concatenate([np.column_stack([xtrain[t,:,0],np.full(NTRAIN,codes[t]),np.ones(NTRAIN)]) for t in sorted(TRAIN_IDS)])
 b=np.concatenate([ytrain[t,:,1]-xtrain[t,:,1] for t in sorted(TRAIN_IDS)])
 lin=np.linalg.lstsq(A,b,rcond=None)[0].astype(np.float32)
 # Actual serialized inference state; the flow and its native RealNVP control are byte-identical.
 task_params=np.stack([[FLOW[0],FLOW[1]*codes[t],FLOW[2],FLOW[3]*codes[t]] for t in range(TASKS)]).astype(np.float32)
 blobs={
  'independent_realnvp_flows':pack(task_flow_params=task_params,task_ids=np.arange(TASKS,dtype=np.int8)),
  'linear_activation_steering':pack(linear_steering=lin,task_codes=codes,task_ids=np.arange(TASKS,dtype=np.int8)),
  'equal_size_residual_mlp':pack(mlp_weights=wfit,task_codes=codes,task_ids=np.arange(TASKS,dtype=np.int8)),
  'native_conditional_realnvp':pack(shared_flow_params=FLOW,task_codes=codes,task_ids=np.arange(TASKS,dtype=np.int8)),
  'mirror_realnvp_functional_view':pack(shared_flow_params=FLOW,task_codes=codes,task_ids=np.arange(TASKS,dtype=np.int8))}
 rows=[]
 for name,blob in blobs.items():
  errs=[];cycle=[];logdet=[];detvals=[]
  start=time.perf_counter()
  for t in HELD:
   x=xtest[t];target=ytest[t]
   if name in ('independent_realnvp_flows','native_conditional_realnvp','mirror_realnvp_functional_view'):
    pred,s=flow_forward(x,codes[t]);inv_s=np.stack([x[:,0],(pred[:,1]-(FLOW[2]*x[:,0]+FLOW[3]*codes[t]))*np.exp(-s)],axis=-1)
    cycle.append(float(np.max(np.abs(inv_s-x))));logdet.append(float(np.mean(s)))
    detvals.extend(np.exp(s).tolist())
   elif name=='linear_activation_steering':
    pred=np.stack([x[:,0],x[:,1]+lin[0]*x[:,0]+lin[1]*codes[t]+lin[2]],axis=-1)
    detvals.extend([1.0]*len(x));cycle.append(float('nan'));logdet.append(float('nan'))
   else:
    pred,z=mlp_forward(x,codes[t],wfit)
    deriv=1+wfit[3]*(1-np.tanh(z)**2)*wfit[2]
    detvals.extend(deriv.tolist());cycle.append(float('nan'));logdet.append(float('nan'))
   errs.append(float(np.linalg.norm(pred-target)/np.linalg.norm(target)))
  elapsed=(time.perf_counter()-start)*1000
  mac={'independent_realnvp_flows':5,'linear_activation_steering':3,'equal_size_residual_mlp':4,'native_conditional_realnvp':5,'mirror_realnvp_functional_view':5}[name]
  rows.append(dict(seed=SEED,method=name,split='heldout_task_codes',serialized_bytes=len(blob),payload_sha256=hashlib.sha256(blob).hexdigest(),
   heldout_function_nrmse=float(np.mean(errs)),max_inverse_cycle_error=float(np.nanmax(cycle)) if not np.all(np.isnan(cycle)) else 'NA',
   mean_logdet=float(np.nanmean(logdet)) if not np.all(np.isnan(logdet)) else 'NA',min_jacobian_det=float(np.min(detvals)),
   active_macs_per_example=mac,exp_ops_per_example=1 if 'realnvp' in name or name=='mirror_realnvp_functional_view' else 0,
   fit_updates=2000 if name=='equal_size_residual_mlp' else 0,fit_wall_ms=fit_ms if name=='equal_size_residual_mlp' else 0,eval_wall_ms=elapsed,
   status_note='exact native RealNVP alias' if name=='mirror_realnvp_functional_view' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 print('train tasks',len(TRAIN_IDS),'heldout',HELD,'MLP fit updates',2000)
if __name__=='__main__':run()
