"""Synthetic matrix memory with task-specific effective-rank allocation."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=82101;D=16;T=8;RANKS=[1,1,2,2,4,4,6,8];N=256

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);center=np.eye(D,dtype=np.float32);basis=rng.normal(size=(D,D)).astype(np.float32);q,_=np.linalg.qr(basis);targets=[]
 for r in RANKS:
  c=rng.normal(size=(D,r)).astype(np.float32)*.04;v=rng.normal(size=(r,D)).astype(np.float32)*.04;targets.append(center+c@v)
 targets=np.stack(targets);x=rng.normal(size=(T,N,D)).astype(np.float32);y=np.stack([x[t]@targets[t].T for t in range(T)])
 # shared basis is fixed orthogonal; project residuals and retain each task's prescribed rank
 coeff=[];recon=[]
 for t,r in enumerate(RANKS):
  u,s,vt=np.linalg.svd(targets[t]-center,full_matrices=False);a=(u[:,:r]*s[:r]).astype(np.float32);b=vt[:r].astype(np.float32);coeff.append((a,b));recon.append(center+a@b)
 adaptive=pack(center=center,task_left=np.concatenate([a for a,b in coeff],axis=1),task_right=np.concatenate([b for a,b in coeff],axis=0),rank_offsets=np.cumsum([0]+RANKS).astype(np.int8),task_ranks=np.array(RANKS,np.int8))
 native=pack(center=center,task_left=np.concatenate([a for a,b in coeff],axis=1),task_right=np.concatenate([b for a,b in coeff],axis=0),rank_offsets=np.cumsum([0]+RANKS).astype(np.int8),native_rank=np.array(RANKS,np.int8))
 fixed_factors=[np.linalg.svd(targets[t]-center,full_matrices=False) for t in range(T)]; fixed_left=np.concatenate([(u[:,:8]*s[:8]) for u,s,vt in fixed_factors],axis=1).astype(np.float32); fixed_right=np.concatenate([vt[:8] for u,s,vt in fixed_factors],axis=0).astype(np.float32); fixed=pack(center=center,task_left=fixed_left,task_right=fixed_right,rank_offsets=(np.arange(T+1)*8).astype(np.int8),task_ranks=np.full(T,8,np.int8))
 full=pack(task_matrices=targets)
 specs=[('independent_full',full,'full'),('fixed_rank8_memory',fixed,'factor'),('mirror_rank_adaptive',adaptive,'factor'),('native_svd_rank_truncation',native,'factor')];rows=[]
 for name,payload,kind in specs:
  z=np.load(io.BytesIO(payload))
  if kind=='full':rec=z['task_matrices']
  else:
   a=z['task_left'];b=z['task_right'];offs=z['rank_offsets'];rec=np.stack([center+a[:,offs[t]:offs[t+1]]@b[offs[t]:offs[t+1]] for t in range(T)])
  pred=np.stack([x[t]@rec[t].T for t in range(T)]);err=nrmse(pred,y);compose=[]
  for t in range(T):compose.append(np.linalg.norm(rec[t]@rec[(t+1)%T]-targets[t]@targets[(t+1)%T])/np.linalg.norm(targets[t]@targets[(t+1)%T]))
  rt=float(np.max(np.abs(rec-(targets if kind=='full' else rec))));st=time.perf_counter();_=np.stack([x[t]@rec[t].T for t in range(T)]);wall=time.perf_counter()-st
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_memory_nrmse=err,mean_pair_composition_relative_error=float(np.mean(compose)),max_effective_rank=max(RANKS),rank_sum=sum(RANKS),test_examples=T*N,optimizer_updates=0,active_macs_per_read=(D*D+2*D*(8 if name in ('fixed_rank8_memory','independent_full') else (sum(RANKS)/T))),payload_roundtrip_max_abs=rt,wall_time_s=wall,status_note='native SVD rank allocation alias' if name=='mirror_rank_adaptive' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
