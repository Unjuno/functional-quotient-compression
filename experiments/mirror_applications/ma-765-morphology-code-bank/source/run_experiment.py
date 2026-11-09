#!/usr/bin/env python3
"""Synthetic morphology bank and damage recovery using one local update rule."""
from __future__ import annotations
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
SEED=76501; N=16; TRAIN=12; SIZE=32; STEPS=24; ETA=.25
torch.set_num_threads(1)
def pack(**a):
 b=io.BytesIO();np.savez_compressed(b,**a);return b.getvalue()

def generate(code):
 angle,phase,freq,bias=code
 yy,xx=np.mgrid[0:SIZE,0:SIZE].astype(np.float32)
 x=(xx-(SIZE-1)/2)/SIZE;y=(yy-(SIZE-1)/2)/SIZE
 wave=np.sin(2*np.pi*freq*(np.cos(angle)*x+np.sin(angle)*y)+phase)
 return (1/(1+np.exp(-4*(wave+bias)))).astype(np.float32)

def evolve(state,target,steps=STEPS):
 x=torch.tensor(state)[None,None]
 t=torch.tensor(target)[None,None]
 avg=F.avg_pool2d(F.pad(x,(1,1,1,1),mode='replicate'),3,stride=1)
 for _ in range(steps):
  avg=F.avg_pool2d(F.pad(x,(1,1,1,1),mode='replicate'),3,stride=1)
  x=x+ETA*(t-x)+.04*(avg-x)
 return x.numpy()[0,0]

def run():
 rng=np.random.default_rng(SEED)
 codes=np.stack([rng.uniform(0,2*np.pi,N),rng.uniform(-np.pi,np.pi,N),rng.integers(1,5,N),rng.uniform(-.2,.2,N)],axis=1).astype(np.float32)
 target=np.stack([generate(c) for c in codes])
 # Damage 30% of each starting organism; shared local rule repairs it.
 states=[];masks=[]
 for i in range(N):
  damaged=rng.random((SIZE,SIZE))<.30
  state=target[i].copy();state[damaged]=rng.uniform(0,1,int(damaged.sum())).astype(np.float32)
  states.append(state);masks.append(damaged)
 states=np.stack(states);masks=np.stack(masks)
 # Full independent morphology map vs compact code bank plus deterministic generator metadata.
 full=pack(morphology_targets=target,task_ids=np.arange(N,dtype=np.int8))
 coords=pack(codes=codes,task_ids=np.arange(N,dtype=np.int8),grid=np.asarray([SIZE],np.int16),generator=np.asarray([1],np.int8))
 # Native analytic coordinate conditioning uses the same code/generator payload.
 hard=pack(shared_target=target[0],task_ids=np.arange(N,dtype=np.int8))
 methods=[('independent_target_bank',full,'full'),('hard_tied_single_morphology',hard,'hard'),('native_analytic_morphology_code',coords,'code'),('mirror_morphology_views',coords,'code')]
 rows=[]
 for name,blob,kind in methods:
  errs=[];before=[];wall=time.perf_counter();gen_ops=0
  for i in range(N):
   goal=target[i] if kind=='full' else (target[0] if kind=='hard' else generate(codes[i]))
   gen_ops += 0 if kind in ('full','hard') else SIZE*SIZE*12
   before.append(float(np.linalg.norm(states[i]-target[i])/np.linalg.norm(target[i])))
   out=evolve(states[i],goal)
   errs.append(float(np.linalg.norm(out-target[i])/np.linalg.norm(target[i])))
  elapsed=(time.perf_counter()-wall)*1000
  # Per-step stencil update proxy plus one target synthesis for code-based methods.
  macs=N*STEPS*SIZE*SIZE*12 + gen_ops
  rows.append(dict(seed=SEED,method=name,split='16_morphologies_with_damage',serialized_bytes=len(blob),payload_sha256=hashlib.sha256(blob).hexdigest(),
   damaged_initial_nrmse=float(np.mean(before)),final_target_nrmse=float(np.mean(errs)),recovery_fraction=float(1-np.mean(errs)/np.mean(before)),
   active_macs=macs,pattern_generation_ops=gen_ops,eval_wall_ms=elapsed,
   status_note='exact native analytic code alias' if name=='mirror_morphology_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('morphologies',N,'damage',.30,'steps',STEPS)
if __name__=='__main__':run()
