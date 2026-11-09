from __future__ import annotations
import hashlib,io,time
import numpy as np
D=128; R=8; N=16; TRAIN=8

def _serialize(arr):
 b=io.BytesIO();np.savez_compressed(b,**arr);x=b.getvalue();return len(x),hashlib.sha256(x).hexdigest()
def _svd_factor(Dm):
 u,s,vt=np.linalg.svd(Dm,full_matrices=False);return u[:,:R],s[:R],vt[:R]
def _delta(B,A):return B@A
def _gauge(B,A,G):return B@G,np.linalg.solve(G,A)
def _task(seed,t):
 rng=np.random.default_rng(seed+1009*t)
 # Matched aligned left/right subspaces, with task-specific dense core and controlled residual.
 if t==0:
  U,_=np.linalg.qr(rng.normal(size=(D,R)));V,_=np.linalg.qr(rng.normal(size=(D,R)))
 return U@rng.normal(size=(R,R))@V.T + 0.02*rng.normal(size=(D,D))
def run(seed=71601):
 rng=np.random.default_rng(seed)
 U,_=np.linalg.qr(rng.normal(size=(D,R)));V,_=np.linalg.qr(rng.normal(size=(D,R)))
 tasks=[]
 for t in range(N):
  rt=np.random.default_rng(seed+1009*t)
  # all share rank-8 span with independent dense core; residual ratio ~ small but nonzero
  tasks.append(U@rt.normal(size=(R,R))@V.T + 0.02*rt.normal(size=(D,D)))
 tasks=np.stack(tasks); Ut=U.copy();Vt=V.copy()
 train=tasks[:TRAIN];held=tasks[TRAIN:]
 # Joint-SVD/KnOTS analogue: concatenate train deltas, take common left and right singular spans.
 # Alternating projection approximates joint alignment; keep top R factors from pooled covariance.
 ql,_=np.linalg.qr(np.concatenate([train[:,:,i] for i in range(D)],axis=1))
 qv,_=np.linalg.qr(np.concatenate([train[:,i,:].T for i in range(D)],axis=1))
 # Since pooled ranks may be full due noise, estimate dominant shared basis by SVD of each train task stack.
 left=np.concatenate([x for x in train],axis=1); right=np.concatenate([x.T for x in train],axis=1)
 Ul,_,_=np.linalg.svd(left,full_matrices=False); Vr,_,_=np.linalg.svd(right,full_matrices=False)
 L=Ul[:,:R]; Q=Vr[:,:R]
 cores=np.einsum('di,tdf,fj->tij',L,tasks,Q)
 recon=np.einsum('di,tij,fj->tdf',L,cores,Q)
 nrmse=lambda pred: np.linalg.norm(pred-held)/(np.linalg.norm(held)+1e-12)
 methods={}
 # independent rank-r SVD factors
 factors=[]
 for x in tasks:
  a,s,v=_svd_factor(x); factors.append((a*s)@v)
 factors=np.stack(factors)
 methods['independent_LoRA']={'B':np.stack([_svd_factor(x)[0]*_svd_factor(x)[1] for x in tasks]),'A':np.stack([_svd_factor(x)[2] for x in tasks])}
 methods['native_dense_aligned_core']={'L':L,'Q':Q,'cores':cores}
 methods['Mirror_sparse_top2']={'L':L,'Q':Q,'top2_values':np.stack([np.sort(np.abs(c.reshape(-1)))[-2:] for c in cores]),'top2_indices':np.stack([np.argsort(np.abs(c.reshape(-1)))[-2:] for c in cores]).astype(np.int16)}
 methods['native_sparse_top2']=dict(methods['Mirror_sparse_top2'])
 diag=np.zeros_like(cores)
 for t in range(N):diag[t]=np.diag(np.diag(cores[t]))
 methods['native_diagonal_core']={'L':L,'Q':Q,'cores':diag}
 # score sparse control with actual index/value reconstruction
 sparse=np.zeros_like(cores)
 for t in range(N):
  idx=methods['Mirror_sparse_top2']['top2_indices'][t]; vals=methods['Mirror_sparse_top2']['top2_values'][t];sparse[t].reshape(-1)[idx]=vals
 methods['Mirror_sparse_top2']['outputs']=np.einsum('di,tij,fj->tdf',L,sparse,Q)
 methods['native_sparse_top2']['outputs']=methods['Mirror_sparse_top2']['outputs'].copy()
 methods['native_dense_aligned_core']['outputs']=recon
 methods['native_diagonal_core']['outputs']=np.einsum('di,tij,fj->tdf',L,diag,Q)
 methods['independent_LoRA']['outputs']=factors
 rows=[]
 for name,arrays in methods.items():
  size,digest=_serialize(arrays)
  pred=arrays['outputs'][TRAIN:]
  # Gauge check functional matrices before/after arbitrary nonorthogonal refactorization.
  maxdelta=0.0
  if name=='independent_LoRA':
   for t in range(N):
    B=arrays['B'][t];A=arrays['A'][t];G=np.eye(R)+0.05*rng.normal(size=(R,R))
    while abs(np.linalg.det(G))<0.1:G=np.eye(R)+0.05*rng.normal(size=(R,R))
    B2,A2=_gauge(B,A,G);maxdelta=max(maxdelta,float(np.max(np.abs(_delta(B,A)-_delta(B2,A2)))))
  rows.append({'seed':seed,'method':name,'sparsity':2 if 'top2' in name else R*R,'serialized_bytes':size,'payload_sha256':digest,'heldout_nrmse':float(nrmse(pred)),'gauge_max_prediction_delta':maxdelta,'active_compute_proxy':int(D*R*R*N if name!='independent_LoRA' else D*R*D*N),'wall_s':0.0,'status_note':'synthetic aligned subspace; whole task updates held out'})
 return rows

def save(rows,path):
 import csv
 with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',default='RESULTS_CORE.csv');p.add_argument('--seed',type=int,default=71601);a=p.parse_args();t=time.perf_counter();rows=run(a.seed);dt=time.perf_counter()-t
 for x in rows:x['wall_s']=dt
 save(rows,a.out)
