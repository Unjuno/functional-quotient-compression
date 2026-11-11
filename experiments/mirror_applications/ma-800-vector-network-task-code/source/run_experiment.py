"""Synthetic compositional task coefficient codes over rank-1 atoms with input-local top-k."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[1];SEED=80001;D=8;K=8;TOPK=2;N=512
TASKS=np.array([(a,b) for a in (-1.,0.,1.) for b in (-1.,0.,1.)],np.float32)
HELD=np.array([0,8]);TRAIN=np.array([i for i in range(9) if i not in HELD])

def pack(**x):
 b=io.BytesIO();np.savez_compressed(b,**x);return b.getvalue()

def infer(x,u,v,coef):
 proj=x@v.T;idx=np.argpartition(np.abs(proj),-TOPK,axis=1)[:,-TOPK:]
 mask=np.zeros_like(proj);np.put_along_axis(mask,idx,1.,axis=1)
 y=np.einsum('nk,nk,nkd->nd',proj*mask,np.broadcast_to(coef,(len(x),K)),np.broadcast_to(u,(len(x),K,D)))
 return y,idx

def nrmse(p,y):return float(np.sqrt(np.mean((p-y)**2)/(np.mean(y*y)+1e-12)))

def run():
 rng=np.random.default_rng(SEED)
 u=rng.normal(0,1,(K,D)).astype(np.float32);v=rng.normal(0,1,(K,D)).astype(np.float32)
 u/=np.linalg.norm(u,axis=1,keepdims=True);v/=np.linalg.norm(v,axis=1,keepdims=True)
 c0=rng.normal(0,.35,K).astype(np.float32);b1=rng.normal(0,.2,K).astype(np.float32);b2=rng.normal(0,.2,K).astype(np.float32)
 teacher=np.stack([c0+a*b1+b*b2 for a,b in TASKS])
 # The two diagonal corners are held out from coefficient-basis fitting.
 design=np.concatenate([np.ones((len(TRAIN),1),np.float32),TASKS[TRAIN]],axis=1)
 coef=np.linalg.lstsq(design,teacher[TRAIN],rcond=None)[0].astype(np.float32)
 learned0=coef[0];learned_basis=coef[1:]
 predicted=np.stack([learned0+a*learned_basis[0]+b*learned_basis[1] for a,b in TASKS])
 xs=rng.normal(0,1,(9,N,D)).astype(np.float32)
 targets=[];routes=[]
 for t in range(9):
  y,idx=infer(xs[t],u,v,teacher[t]);targets.append(y);routes.append(idx)
 targets=np.stack(targets);routes=np.stack(routes)
 local_meta=dict(right_atoms=v,topk=np.array(TOPK,np.int8),task_ids=np.arange(9,dtype=np.int16))
 native_payload=pack(left_atoms=u,right_atoms=v,coefficient_intercept=learned0,coefficient_basis=learned_basis,task_codes=TASKS,topk=np.array(TOPK,np.int8),task_ids=np.arange(9,dtype=np.int16))
 shared=np.broadcast_to(teacher[TRAIN].mean(0),(9,K)).copy()
 payloads=[('independent_task_coefficients',teacher,pack(left_atoms=u,**local_meta,task_coefficients=teacher),0),
  ('shared_coefficients_no_task_code',shared,pack(left_atoms=u,**local_meta,task_coefficients=teacher[TRAIN].mean(0)),0),
  ('mirror_factorized_task_code',predicted,native_payload,9*3*K),
  ('native_lowrank_coefficient_factorization',predicted,native_payload,9*3*K)]
 rows=[]
 for name,ct,payload,decode in payloads:
  saved=np.load(io.BytesIO(payload))
  if name=='independent_task_coefficients':restored=saved['task_coefficients']
  elif name=='shared_coefficients_no_task_code':restored=np.broadcast_to(saved['task_coefficients'],(9,K))
  else:restored=np.stack([saved['coefficient_intercept']+a*saved['coefficient_basis'][0]+b*saved['coefficient_basis'][1] for a,b in saved['task_codes']])
  rt=float(np.max(np.abs(restored-ct)))
  if rt>1e-5:raise AssertionError(f'{name} payload reconstruction error {rt}')
  start=time.perf_counter();outs=[];idxs=[]
  for t in range(9):
   y,idx=infer(xs[t],u,v,ct[t]);outs.append(y);idxs.append(idx)
  wall=time.perf_counter()-start;outs=np.stack(outs);idxs=np.concatenate(idxs)
  unique={tuple(sorted(int(j) for j in row)) for row in idxs}
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),
   heldout_output_nrmse=nrmse(outs[HELD],targets[HELD]),all_task_output_nrmse=nrmse(outs,targets),
   average_active_atoms=float(np.mean(np.sum(np.ones_like(idxs),axis=1))),unique_local_atom_sets=len(unique),
   local_inference_examples=9*N,active_macs_per_example=K*D+TOPK*D,reconstruction_macs_total=decode,
   optimizer_updates=0,wall_time_s=wall,payload_roundtrip_max_abs=rt,status_note='exact native coefficient factorization alias' if name=='mirror_factorized_task_code' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows

if __name__=='__main__':
 for r in run():print(r)
