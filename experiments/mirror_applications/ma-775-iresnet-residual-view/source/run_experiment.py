#!/usr/bin/env python3
"""Contractive invertible residual task family with shared basis Views."""
from __future__ import annotations
import csv,hashlib,io,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SEED=77501; D=4; TASKS=24; TRAIN=16; HELD=8; RANK=2; CAL=32; TEST=256; TOL=1e-6
def pack(**a):
 b=io.BytesIO();np.savez_compressed(b,**a);return b.getvalue()

def inverse(y,L,maxiter=500):
 x=y.copy()
 for i in range(maxiter):
  xn=y-x@L.T
  if np.max(np.abs(xn-x))<TOL:return xn,i+1,True
  x=xn
 return x,maxiter,False

def run():
 rng=np.random.default_rng(SEED)
 basis=rng.normal(size=(RANK,D,D)).astype(np.float32)
 basis*=.045
 codes=rng.normal(size=(TASKS,RANK)).astype(np.float32)
 mats=np.einsum('tr,rij->tij',codes,basis).astype(np.float32)
 lips=np.asarray([np.linalg.norm(a,2) for a in mats])
 # Training-task SVD basis and heldout code fit from residual output examples.
 _,_,vt=np.linalg.svd(mats[:TRAIN].reshape(TRAIN,-1),full_matrices=False)
 learned=vt[:RANK].reshape(RANK,D,D).astype(np.float32)
 xcal=rng.normal(size=(TASKS,CAL,D)).astype(np.float32)
 fitted=[]
 for t in range(TASKS):
  outs=np.stack([xcal[t]@learned[r].T for r in range(RANK)],axis=2).reshape(-1,RANK)
  target=(xcal[t]@mats[t].T).reshape(-1)
  fitted.append(np.linalg.lstsq(outs,target,rcond=None)[0])
 fitted=np.asarray(fitted,np.float32);recon=np.einsum('tr,rij->tij',fitted,learned)
 mean=np.mean(mats[:TRAIN],axis=0)
 gains=np.asarray([np.sum((xcal[t]@mean.T)*(xcal[t]@mats[t].T))/(np.sum((xcal[t]@mean.T)**2)+1e-12) for t in range(TASKS)],np.float32)
 film=gains[:,None,None]*mean[None]
 ids=np.arange(TASKS,dtype=np.int8)
 blobs={
  'independent_residual_blocks':pack(residual_matrices=mats,task_ids=ids),
  'hard_tied_residual':pack(residual_matrix=mean,task_ids=ids),
  'scalar_film_residual':pack(residual_matrix=mean,task_gains=gains,task_ids=ids),
  'native_rank2_residual_basis':pack(residual_basis=learned,task_codes=fitted,task_ids=ids),
  'mirror_invertible_residual_views':pack(residual_basis=learned,task_codes=fitted,task_ids=ids)}
 eff={'independent_residual_blocks':mats,'hard_tied_residual':np.broadcast_to(mean,mats.shape),
      'scalar_film_residual':film,'native_rank2_residual_basis':recon,'mirror_invertible_residual_views':recon}
 test_inputs=rng.normal(size=(HELD,TEST,D)).astype(np.float32)
 rows=[]
 for name,blob in blobs.items():
  fwd=[];inv=[];its=[];ok=[];ld=[];det=[];t0=time.perf_counter()
  for t in range(TRAIN,TASKS):
   x=test_inputs[t-TRAIN];L=eff[name][t];teacher=mats[t]
   pred=x+x@L.T;truth=x+x@teacher.T
   fwd.append(float(np.linalg.norm(pred-truth)/(np.linalg.norm(truth)+1e-12)))
   recovered,n,success=inverse(pred,L);inv.append(float(np.max(np.abs(recovered-x))));its.append(n);ok.append(success)
   jac=np.eye(D,dtype=np.float64)+L.astype(np.float64);ld.append(float(np.linalg.slogdet(jac)[1]));det.append(float(np.linalg.svd(jac,compute_uv=False).min()))
  elapsed=(time.perf_counter()-t0)*1000
  view=name in ('native_rank2_residual_basis','mirror_invertible_residual_views')
  macs=HELD*TEST*(D*RANK if view else D)
  rows.append(dict(seed=SEED,method=name,split='heldout_residual_blocks',serialized_bytes=len(blob),payload_sha256=hashlib.sha256(blob).hexdigest(),
   heldout_output_nrmse=float(np.mean(fwd)),max_inverse_error=float(np.max(inv)),inverse_iterations_mean=float(np.mean(its)),inverse_converged_fraction=float(np.mean(ok)),
   mean_logdet=float(np.mean(ld)),min_jacobian_singular=float(np.min(det)),max_teacher_lipschitz=float(np.max(lips[TRAIN:])),active_macs=macs,
   precompose_macs=(HELD*D*D*RANK if view else 0),eval_wall_ms=elapsed,status_note='exact native residual-basis alias' if name=='mirror_invertible_residual_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('train',TRAIN,'heldout',HELD,'teacher max Lipschitz',float(lips.max()))
if __name__=='__main__':run()
