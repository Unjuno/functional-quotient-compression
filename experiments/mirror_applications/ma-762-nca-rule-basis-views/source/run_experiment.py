#!/usr/bin/env python3
"""Low-rank local-rule bank for synthetic cellular automata dynamics."""
from __future__ import annotations
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
SEED=76201; TASKS=96; TRAIN=64; HELD=32; RANK=3; SIZE=16; NMAP=8; STEPS=12
torch.set_num_threads(1)

def pack(**a):
    b=io.BytesIO();np.savez_compressed(b,**a);return b.getvalue()
def conv(x,k):return F.conv2d(x,k,padding=1)

def run():
    rng=np.random.default_rng(SEED)
    basis=rng.normal(scale=.10,size=(RANK,1,1,3,3)).astype(np.float32)
    codes=rng.normal(size=(TASKS,RANK)).astype(np.float32)
    kernels=np.einsum('tr,roihw->toihw',codes,basis).astype(np.float32)
    xcal=torch.tensor(rng.normal(scale=.4,size=(TASKS,NMAP,1,SIZE,SIZE)).astype(np.float32))
    teachers=[]
    for t in range(TASKS):teachers.append(conv(xcal[t].reshape(NMAP,1,SIZE,SIZE),torch.tensor(kernels[t])))
    teachers=torch.stack(teachers)
    # Recover a shared rank-3 kernel basis from training task kernels only.
    U,S,Vt=np.linalg.svd(kernels[:TRAIN].reshape(TRAIN,-1),full_matrices=False)
    learned=Vt[:RANK].reshape(RANK,1,1,3,3).astype(np.float32)
    # Fit every code from update demonstrations. Heldout task IDs are not used
    # to fit the basis; their coefficients use only the calibration maps.
    design=[];target=[]
    for t in range(TASKS):
      basis_out=[conv(xcal[t].reshape(NMAP,1,SIZE,SIZE),torch.tensor(learned[r])).numpy().reshape(-1) for r in range(RANK)]
      design.append(np.stack(basis_out,axis=1));target.append(teachers[t].numpy().reshape(-1))
    fitted=np.stack([np.linalg.lstsq(design[t],target[t],rcond=None)[0] for t in range(TASKS)]).astype(np.float32)
    reconstructed=np.einsum('tr,roihw->toihw',fitted,learned)
    # Cheapest scalar FiLM/gate control: one gain per task on the shared mean kernel.
    mean_kernel=np.mean(kernels[:TRAIN],axis=0)
    mean_outputs=[]
    for t in range(TASKS):mean_outputs.append(conv(xcal[t].reshape(NMAP,1,SIZE,SIZE),torch.tensor(mean_kernel)).numpy().reshape(-1))
    gains=np.asarray([np.dot(mean_outputs[t],target[t])/(np.dot(mean_outputs[t],mean_outputs[t])+1e-12) for t in range(TASKS)],np.float32)
    film_kernels=gains[:,None,None,None,None]*mean_kernel[None]
    ids=np.arange(TASKS,dtype=np.int8)
    full=pack(local_kernels=kernels,task_ids=ids)
    hard=pack(local_kernel=mean_kernel,task_ids=ids)
    film=pack(local_kernel=mean_kernel,task_gain=gains,task_ids=ids)
    factor=pack(shared_kernel_basis=learned,task_codes=fitted,task_ids=ids)
    methods=[('independent_rule_bank',full,kernels,'full'),('hard_tied_single_rule',hard,np.broadcast_to(mean_kernel,kernels.shape),'single'),
             ('task_gain_film',film,film_kernels,'film'),('native_rank3_rule_basis',factor,reconstructed,'basis'),('mirror_rule_views',factor,reconstructed,'basis')]
    rows=[]
    test_x=torch.tensor(rng.normal(scale=.4,size=(HELD,NMAP,1,SIZE,SIZE)).astype(np.float32))
    for name,blob,eff,kind in methods:
      op=[];roll=[];maxratio=[];start=time.perf_counter()
      for t in range(TRAIN,TASKS):
        kt=torch.tensor(eff[t]);teacher_k=torch.tensor(kernels[t])
        pred=conv(test_x[t-TRAIN],kt);truth=conv(test_x[t-TRAIN],teacher_k)
        op.append(float(torch.linalg.vector_norm(pred-truth)/(torch.linalg.vector_norm(truth)+1e-12)))
        xx=test_x[t-TRAIN].clone();yy=xx.clone();ratio=[]
        for _ in range(STEPS):
          if kind=='basis':
            z=sum(fitted[t,r]*conv(xx,torch.tensor(learned[r])) for r in range(RANK))
          else:z=conv(xx,kt)
          xx=torch.tanh(z);yy=torch.tanh(conv(yy,teacher_k));ratio.append(float(torch.linalg.vector_norm(xx)/(torch.linalg.vector_norm(yy)+1e-12)))
        roll.append(float(torch.linalg.vector_norm(xx-yy)/(torch.linalg.vector_norm(yy)+1e-12)))
        maxratio.append(max(ratio))
      elapsed=(time.perf_counter()-start)*1000
      # Include 12 heldout tasks x 8 maps x 12 recurrent steps.
      factor_ops=RANK if kind=='basis' else 1
      macs=HELD*NMAP*STEPS*SIZE*SIZE*9*factor_ops
      rows.append(dict(seed=SEED,method=name,split='heldout_rules',serialized_bytes=len(blob),payload_sha256=hashlib.sha256(blob).hexdigest(),
          heldout_update_nrmse=float(np.mean(op)),rollout_nrmse=float(np.mean(roll)),max_state_norm_ratio=float(np.max(maxratio)),
          active_macs=macs,fit_updates=0,eval_wall_ms=elapsed,status_note='exact native rule-basis alias' if name=='mirror_rule_views' else ''))
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('wrote',len(rows),'rows; train rules',TRAIN,'heldout rules',HELD)
if __name__=='__main__':run()
