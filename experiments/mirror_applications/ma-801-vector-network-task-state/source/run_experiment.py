"""Synthetic task x state coefficient factorization with per-input atom selection."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[1];SEED=80101;D=8;K=8;TOPK=2;NT=4;NS=4;N=256
HELD=np.array([(t,s) for t in range(NT) for s in range(NS) if (t+s)%2==0],np.int16)
TRAIN=np.array([(t,s) for t in range(NT) for s in range(NS) if (t+s)%2!=0],np.int16)

def pack(**x):
 b=io.BytesIO();np.savez_compressed(b,**x);return b.getvalue()

def local(x,u,v,c):
 proj=x@v.T;idx=np.argpartition(np.abs(proj),-TOPK,axis=1)[:,-TOPK:]
 mask=np.zeros_like(proj);np.put_along_axis(mask,idx,1.,axis=1)
 y=np.einsum('nk,nk,nkd->nd',proj*mask,np.broadcast_to(c,(len(x),K)),np.broadcast_to(u,(len(x),K,D)))
 return y,idx

def nrmse(p,y):return float(np.sqrt(np.mean((p-y)**2)/(np.mean(y*y)+1e-12)))

def run():
 rng=np.random.default_rng(SEED)
 u=rng.normal(0,1,(K,D)).astype(np.float32);v=rng.normal(0,1,(K,D)).astype(np.float32)
 u/=np.linalg.norm(u,axis=1,keepdims=True);v/=np.linalg.norm(v,axis=1,keepdims=True)
 tf=rng.normal(0,1,(NT,2)).astype(np.float32);sf=rng.normal(0,1,(NS,2)).astype(np.float32);basis=rng.normal(0,.3,(2,K)).astype(np.float32)
 teacher=np.einsum('tr,sr,rk->tsk',tf,sf,basis)
 # Factor vectors are known/supplied; estimate the shared atom coefficients on observed task-state pairs only.
 design=np.stack([tf[t]*sf[s] for t,s in TRAIN]);Y=teacher[TRAIN[:,0],TRAIN[:,1]]
 fitted=np.linalg.lstsq(design,Y,rcond=None)[0].astype(np.float32)
 predicted=np.einsum('tr,sr,rk->tsk',tf,sf,fitted)
 task_only=np.broadcast_to(teacher.mean(axis=1)[:,None,:],teacher.shape).copy()
 state_only=np.broadcast_to(teacher.mean(axis=0)[None,:,:],teacher.shape).copy()
 centers=np.zeros((NS,D),np.float32)
 for s in range(NS):centers[s,s]=2.2
 xs=np.empty((NT,NS,N,D),np.float32);ys=np.empty((NT,NS,N,D),np.float32);routes=[]
 for t in range(NT):
  for s in range(NS):
   xs[t,s]=rng.normal(0,.55,(N,D)).astype(np.float32)+centers[s]
   ys[t,s]=local(xs[t,s],u,v,teacher[t,s])[0]
 coeff0=pack(left_atoms=u,right_atoms=v,topk=np.array(TOPK,np.int8),task_ids=np.arange(NT,dtype=np.int8),state_ids=np.arange(NS,dtype=np.int8))
 indep=pack(left_atoms=u,right_atoms=v,topk=np.array(TOPK,np.int8),task_state_coefficients=teacher,task_ids=np.arange(NT,dtype=np.int8),state_ids=np.arange(NS,dtype=np.int8))
 mirror=pack(left_atoms=u,right_atoms=v,topk=np.array(TOPK,np.int8),task_factors=tf,state_factors=sf,atom_coefficient_basis=fitted,task_ids=np.arange(NT,dtype=np.int8),state_ids=np.arange(NS,dtype=np.int8))
 payloads=[('independent_task_state_coefficients',teacher,indep,0),
  ('task_only_coefficients',task_only,pack(**{'left_atoms':u,'right_atoms':v,'topk':np.array(TOPK,np.int8),'task_coefficients':teacher.mean(axis=1),'task_ids':np.arange(NT,dtype=np.int8),'state_ids':np.arange(NS,dtype=np.int8)}),0),
  ('state_only_coefficients',state_only,pack(left_atoms=u,right_atoms=v,topk=np.array(TOPK,np.int8),state_coefficients=teacher.mean(axis=0),task_ids=np.arange(NT,dtype=np.int8),state_ids=np.arange(NS,dtype=np.int8)),0),
  ('mirror_task_state_views',predicted,mirror,NT*NS*2*K),
  ('native_cp_coefficient_factorization',predicted,mirror,NT*NS*2*K)]
 rows=[]
 for name,cs,payload,decode in payloads:
  saved=np.load(io.BytesIO(payload))
  if name=='independent_task_state_coefficients':restored=saved['task_state_coefficients']
  elif name=='task_only_coefficients':restored=np.broadcast_to(saved['task_coefficients'][:,None,:],(NT,NS,K))
  elif name=='state_only_coefficients':restored=np.broadcast_to(saved['state_coefficients'][None,:,:],(NT,NS,K))
  else:restored=np.einsum('tr,sr,rk->tsk',saved['task_factors'],saved['state_factors'],saved['atom_coefficient_basis'])
  rt=float(np.max(np.abs(restored-cs)))
  if rt>1e-5:raise AssertionError(f'{name} payload reconstruction error {rt}')
  start=time.perf_counter();out=np.empty_like(ys);active=[];allidx=[]
  for t in range(NT):
   for s in range(NS):
    out[t,s],idx=local(xs[t,s],u,v,cs[t,s]);active.extend([TOPK]*len(idx));allidx.extend(idx.tolist())
  wall=time.perf_counter()-start
  hpred=out[HELD[:,0],HELD[:,1]];htarget=ys[HELD[:,0],HELD[:,1]]
  sets={tuple(sorted(int(j) for j in q)) for q in allidx}
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),
   heldout_output_nrmse=nrmse(hpred,htarget),all_combination_output_nrmse=nrmse(out,ys),average_active_atoms=float(np.mean(active)),
   unique_local_atom_sets=len(sets),local_inference_iterations=1,active_macs_per_example=K*D+TOPK*D,
   reconstruction_macs_total=decode,test_examples=NT*NS*N,optimizer_updates=0,wall_time_s=wall,payload_roundtrip_max_abs=rt,
   status_note='exact native CP coefficient alias' if name=='mirror_task_state_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows

if __name__=='__main__':
 for r in run():print(r)
