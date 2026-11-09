#!/usr/bin/env python3
"""Context-selected local rule interpolation vs attention over rule features."""
from __future__ import annotations
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
SEED=76401; TASKS=32; TRAIN=20; HELD=12; SIZE=16; BATCH=8; STEPS=10
torch.set_num_threads(1)

def pack(**a):
 b=io.BytesIO();np.savez_compressed(b,**a);return b.getvalue()
def conv(x,k):return F.conv2d(x,k,padding=1)

def run():
 rng=np.random.default_rng(SEED)
 contexts=np.linspace(-1,1,TASKS,dtype=np.float32)
 alpha=1/(1+np.exp(-3*contexts))
 k0=rng.normal(scale=.11,size=(1,1,3,3)).astype(np.float32)
 k1=rng.normal(scale=.11,size=(1,1,3,3)).astype(np.float32)
 kernels=alpha[:,None,None,None,None]*k0+(1-alpha[:,None,None,None,None])*k1
 x=torch.tensor(rng.normal(scale=.35,size=(TASKS,BATCH,1,SIZE,SIZE)).astype(np.float32))
 # Fit attention-to-rule coordinate from train context/code pairs only.
 def sigmoid(z):return 1/(1+np.exp(-z))
 fit_start=time.perf_counter()
 design=np.stack([contexts[:TRAIN],np.ones(TRAIN)],axis=1)
 slope,bias=np.linalg.lstsq(design,np.log(alpha[:TRAIN]/(1-alpha[:TRAIN])),rcond=None)[0]
 router_fit_ms=(time.perf_counter()-fit_start)*1000
 pred_alpha=sigmoid(slope*contexts+bias).astype(np.float32)
 fitted_kernel=pred_alpha[:,None,None,None,None]*k0+(1-pred_alpha[:,None,None,None,None])*k1
 # Contextwise attention-feature control mixes nonlinear branch features.
 feature_mix=[];teacher=[]
 t0=time.perf_counter()
 for task in range(TASKS):
  xx=x[task]
  teacher.append(torch.tanh(conv(xx,torch.tensor(kernels[task]))))
  feature_mix.append(alpha[task]*torch.tanh(conv(xx,torch.tensor(k0)))+(1-alpha[task])*torch.tanh(conv(xx,torch.tensor(k1))))
 teacher=torch.stack(teacher);feature_mix=torch.stack(feature_mix)
 # Rollout compares the recurrent local rule for heldout contexts.
 testx=torch.tensor(rng.normal(scale=.35,size=(HELD,BATCH,1,SIZE,SIZE)).astype(np.float32))
 full=pack(task_kernels=kernels,context_ids=np.arange(TASKS,dtype=np.int8))
 shared=pack(rule_basis=np.stack([k0,k1]),attention_slope=np.asarray([slope,bias],np.float32),contexts=contexts,context_ids=np.arange(TASKS,dtype=np.int8))
 feature=pack(rule_basis=np.stack([k0,k1]),attention_slope=np.asarray([slope,bias],np.float32),contexts=contexts,context_ids=np.arange(TASKS,dtype=np.int8))
 hard=pack(rule_kernel=np.mean(kernels[:TRAIN],axis=0),context_ids=np.arange(TASKS,dtype=np.int8))
 rows=[]
 for name,blob,kind in [('independent_context_rules',full,'full'),('hard_tied_rule',hard,'hard'),('attention_feature_mixture',feature,'feature'),('native_attention_hypernetwork',shared,'view'),('mirror_attention_rule_view',shared,'view')]:
  errs=[];roll=[];norm=[];t0=time.perf_counter()
  for j in range(HELD):
   task=TRAIN+j;xx=testx[j].clone();yy=xx.clone();kernel=torch.tensor(kernels[task]);teacher_z=torch.tanh(conv(testx[j],kernel))
   if kind=='full':effective=kernel
   elif kind=='hard':effective=torch.tensor(np.mean(kernels[:TRAIN],axis=0))
   elif kind=='feature':
    # one-step nonlinear feature attention; separate branch activations
    z=alpha[task]*torch.tanh(conv(testx[j],torch.tensor(k0)))+(1-alpha[task])*torch.tanh(conv(testx[j],torch.tensor(k1)))
   else:effective=torch.tensor(fitted_kernel[task])
   if kind!='feature':z=torch.tanh(conv(xx,effective))
   errs.append(float(torch.linalg.vector_norm(z-teacher_z)/(torch.linalg.vector_norm(teacher_z)+1e-12)))
   ratios=[]
   for _ in range(STEPS):
    if kind=='feature':xx=alpha[task]*torch.tanh(conv(xx,torch.tensor(k0)))+(1-alpha[task])*torch.tanh(conv(xx,torch.tensor(k1)))
    else:xx=torch.tanh(conv(xx,effective))
    yy=torch.tanh(conv(yy,kernel));ratios.append(float(torch.linalg.vector_norm(xx)/(torch.linalg.vector_norm(yy)+1e-12)))
   roll.append(float(torch.linalg.vector_norm(xx-yy)/(torch.linalg.vector_norm(yy)+1e-12)));norm.append(max(ratios))
  elapsed=(time.perf_counter()-t0)*1000
  macs=HELD*BATCH*STEPS*SIZE*SIZE*9*(2 if kind in ('feature','view') else 1)
  rows.append(dict(seed=SEED,method=name,split='heldout_contexts',serialized_bytes=len(blob),payload_sha256=hashlib.sha256(blob).hexdigest(),
   heldout_update_nrmse=float(np.mean(errs)),rollout_nrmse=float(np.mean(roll)),max_state_norm_ratio=float(np.max(norm)),active_macs=macs,
   router_fit_examples=TRAIN,fit_wall_ms=router_fit_ms if name in ('native_attention_hypernetwork','mirror_attention_rule_view') else 0,eval_wall_ms=elapsed,
   status_note='exact native kernel-hypernetwork alias' if name=='mirror_attention_rule_view' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('train contexts',TRAIN,'heldout',HELD,'attention fit',slope,bias)
if __name__=='__main__':run()
