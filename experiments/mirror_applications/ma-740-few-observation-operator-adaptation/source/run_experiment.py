#!/usr/bin/env python3
"""Few-shot analytic Fourier operator adaptation screen."""
from __future__ import annotations
import argparse,csv,hashlib,io,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1]
FREQ=np.arange(1,5,dtype=np.float64);TASKS=16;AMP=np.asarray([1.,.72,.48,.31])
RIDGE=1e-4

def pack(**a):
 b=io.BytesIO();np.savez_compressed(b,**a);return b.getvalue()
def basis(x):return np.stack([np.cos(x[:,None]*FREQ),np.sin(x[:,None]*FREQ)],axis=-1).reshape(len(x),-1)
def phase_eval(x,phase):return np.cos(x[:,None]*FREQ+phase[None,:])@AMP
def nrmse(y,p):return float(np.linalg.norm(p-y)/max(np.linalg.norm(y),1e-30))
def run(seed=74001,outname='RESULTS_CORE.csv'):
 rng=np.random.default_rng(seed);phases=rng.uniform(-1.2,1.2,size=(TASKS,4));rows=[]
 test_x=rng.uniform(-np.pi,np.pi,size=(TASKS,256));truth=np.stack([phase_eval(test_x[t],phases[t]) for t in range(TASKS)])
 for nobs in (4,8,16):
  obs_x=rng.uniform(-np.pi,np.pi,size=(TASKS,nobs));obs_y=np.stack([phase_eval(obs_x[t],phases[t]) for t in range(TASKS)])+rng.normal(0,.01,size=(TASKS,nobs))
  fitted=np.zeros_like(phases);ctx=np.zeros((TASKS,8));fevals=0;proxy=0;start=time.perf_counter()
  for t in range(TASKS):
   opt=least_squares(lambda z:phase_eval(obs_x[t],z)-obs_y[t],np.zeros(4),max_nfev=200,xtol=1e-11,ftol=1e-11,gtol=1e-11)
   fitted[t]=opt.x;fevals+=int(opt.nfev*nobs);proxy+=int(opt.nfev*nobs*4*4)
  mirror_ms=(time.perf_counter()-start)*1000
  X=basis(obs_x[0]);context=np.zeros_like(ctx);ctx_evals=0;ctx_proxy=0;start=time.perf_counter()
  for t in range(TASKS):
   Xt=basis(obs_x[t]);context[t]=np.linalg.solve(Xt.T@Xt+RIDGE*np.eye(8),Xt.T@obs_y[t]) if RIDGE else np.linalg.lstsq(Xt,obs_y[t],rcond=None)[0];ctx_evals+=nobs;ctx_proxy+=nobs*8*8+8**3
  ctx_ms=(time.perf_counter()-start)*1000
  pred_phase=np.stack([phase_eval(test_x[t],fitted[t]) for t in range(TASKS)])
  pred_context=np.einsum('tni,ti->tn',np.stack([basis(x) for x in test_x]),context)
  pred_frozen=np.stack([phase_eval(test_x[t],np.zeros(4)) for t in range(TASKS)])
  phase_state=pack(frequencies=FREQ.astype(np.int8),amplitudes=AMP.astype(np.float32),phase_codes=fitted.astype(np.float32),task_ids=np.arange(TASKS,dtype=np.int8))
  native_phase=phase_state
  context_state=pack(frequencies=FREQ.astype(np.int8),coefficients=context.astype(np.float32),task_ids=np.arange(TASKS,dtype=np.int8))
  frozen=pack(frequencies=FREQ.astype(np.int8),amplitudes=AMP.astype(np.float32),task_ids=np.arange(TASKS,dtype=np.int8))
  rows.extend([
   dict(seed=seed,method='frozen_shared',observations_per_task=nobs,serialized_bytes=len(frozen),payload_sha256=hashlib.sha256(frozen).hexdigest(),task_state_bytes=0,mean_test_nrmse=np.mean([nrmse(truth[t],pred_frozen[t]) for t in range(TASKS)]),mean_function_evaluations=0,adaptation_multiply_proxy=0,wall_ms=0.,status_note='no task adaptation'),
   dict(seed=seed,method='mirror_phase_code',observations_per_task=nobs,serialized_bytes=len(phase_state),payload_sha256=hashlib.sha256(phase_state).hexdigest(),task_state_bytes=len(pack(phase_codes=fitted.astype(np.float32),task_ids=np.arange(TASKS,dtype=np.int8))),mean_test_nrmse=np.mean([nrmse(truth[t],pred_phase[t]) for t in range(TASKS)]),mean_function_evaluations=fevals,adaptation_multiply_proxy=proxy,wall_ms=mirror_ms,status_note=''),
   dict(seed=seed,method='native_phase_coordinate',observations_per_task=nobs,serialized_bytes=len(native_phase),payload_sha256=hashlib.sha256(native_phase).hexdigest(),task_state_bytes=len(pack(phase_codes=fitted.astype(np.float32),task_ids=np.arange(TASKS,dtype=np.int8))),mean_test_nrmse=np.mean([nrmse(truth[t],pred_phase[t]) for t in range(TASKS)]),mean_function_evaluations=fevals,adaptation_multiply_proxy=proxy,wall_ms=mirror_ms,status_note='exact mirror alias'),
   dict(seed=seed,method='cavia_linear_context',observations_per_task=nobs,serialized_bytes=len(context_state),payload_sha256=hashlib.sha256(context_state).hexdigest(),task_state_bytes=len(pack(coefficients=context.astype(np.float32),task_ids=np.arange(TASKS,dtype=np.int8))),mean_test_nrmse=np.mean([nrmse(truth[t],pred_context[t]) for t in range(TASKS)]),mean_function_evaluations=ctx_evals,adaptation_multiply_proxy=ctx_proxy,wall_ms=ctx_ms,status_note=f'ridge context fit lambda={RIDGE}'),
   dict(seed=seed,method='full_fourier_finetune',observations_per_task=nobs,serialized_bytes=len(context_state),payload_sha256=hashlib.sha256(context_state).hexdigest(),task_state_bytes=len(pack(coefficients=context.astype(np.float32),task_ids=np.arange(TASKS,dtype=np.int8))),mean_test_nrmse=np.mean([nrmse(truth[t],pred_context[t]) for t in range(TASKS)]),mean_function_evaluations=ctx_evals,adaptation_multiply_proxy=ctx_proxy,wall_ms=ctx_ms,status_note='same ridge-fitted 8 coefficients as context')
  ])
 with (ROOT/outname).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('wrote',len(rows),'rows seed',seed)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=74001);p.add_argument('--output',default='RESULTS_CORE.csv');a=p.parse_args();run(a.seed,a.output)
