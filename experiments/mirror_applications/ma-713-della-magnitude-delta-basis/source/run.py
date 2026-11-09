#!/usr/bin/env python3
"""Synthetic magnitude-retained delta compression with native PCA controls."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D,N,R,P=2048,16,8,128
SEEDS={'dev-71301':71301,'audit-71311':71311,'audit-71312':71312,'audit-71313':71313}
RHOS=(0.0,0.05,0.2);KEEP=.1;TRIM=.2
def pack(**a):
 b=io.BytesIO();np.savez(b,**a);return b.getvalue()
def sparse_payload(deltas,masks):
 offsets=[0];idx=[];vals=[]
 for x,m in zip(deltas,masks):
  i=np.flatnonzero(m);idx.extend(i.tolist());vals.extend(x[i].tolist());offsets.append(len(idx))
 return pack(offsets=np.asarray(offsets,np.int32),indices=np.asarray(idx,np.uint16),values=np.asarray(vals,np.float32),metadata=np.asarray([N,D],np.int32))
def decode_sparse(p):
 x=np.zeros((N,D),np.float32)
 for n in range(N):
  a,b=int(p['offsets'][n]),int(p['offsets'][n+1]);x[n,p['indices'][a:b].astype(int)]=p['values'][a:b]
 return x
def nrmse(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
def ties_merge(d):
 z=d.copy();k=int(D*(1-TRIM))
 for i in range(N):
  t=np.partition(np.abs(z[i]),-k)[-k];z[i,np.abs(z[i])<t]=0
 sign=np.sign(z.sum(0));m=(z*sign[None,:])>0;cnt=m.sum(0)
 return np.divide((z*m).sum(0),np.maximum(cnt,1))
def run_one(label,seed,outdir,rhos=RHOS):
 rng=np.random.default_rng(seed);outdir.mkdir(parents=True,exist_ok=True);rows=[];basis,_=np.linalg.qr(rng.normal(size=(D,R)))
 for rho in rhos:
  codes=rng.normal(size=(N,R));delta=codes@basis.T
  if rho:
   priv=rng.normal(size=(N,D));priv*=rho*np.linalg.norm(delta,axis=1,keepdims=True)/np.maximum(np.linalg.norm(priv,axis=1,keepdims=True),1e-30);delta+=priv
  probes=[rng.normal(size=(P,D)).astype(np.float32) for _ in range(N)];target=[x@d for x,d in zip(probes,delta)]
  k=int(D*KEEP);mag_mask=np.zeros((N,D),bool);rand_mask=np.zeros((N,D),bool)
  for n in range(N):
   mag_mask[n,np.argpartition(np.abs(delta[n]),-k)[-k:]]=True
   rand_mask[n,rng.choice(D,size=k,replace=False)]=True
  mag=delta*mag_mask;rand=delta*rand_mask/KEEP
  t=time.perf_counter();mean=mag.mean(0);u,s,vt=np.linalg.svd(mag-mean,full_matrices=False);pca_wall=time.perf_counter()-t
  t=time.perf_counter();ties=ties_merge(delta);ties_wall=time.perf_counter()-t
  models={'independent_dense':(pack(deltas=delta.astype(np.float32),metadata=np.asarray([N,D],np.int32)),delta,'dense'),'magnitude_topk':(sparse_payload(mag,mag_mask),mag,'sparse_mag'),'random_DARE':(sparse_payload(rand,rand_mask),rand,'sparse_random'),'TIES_trim20':(pack(merged=ties.astype(np.float32),metadata=np.asarray([D,TRIM],np.float32)),np.repeat(ties[None,:],N,axis=0),'merge')}
  for rank in (1,2,4,8):
   coeff=(mag-mean)@vt[:rank].T;recon=mean+coeff@vt[:rank]
   data=pack(mean=mean.astype(np.float32),basis=vt[:rank].astype(np.float32),codes=coeff.astype(np.float32),metadata=np.asarray([N,D,rank],np.int32))
   models[f'Magnitude_PCA_r{rank}']=(data,recon,'pca');models[f'native_PCA_r{rank}']=(data,recon.copy(),'pca')
  for method,(payload,_,kind) in models.items():
   (outdir/f'{label}_rho{rho:g}.{method}.npz').write_bytes(payload)
   with np.load(io.BytesIO(payload),allow_pickle=False) as q:
    if kind=='dense':replay=q['deltas']
    elif kind in ('sparse_mag','sparse_random'):replay=decode_sparse(q)
    elif kind=='pca':replay=q['mean']+q['codes']@q['basis']
    else:replay=np.repeat(q['merged'][None,:],N,axis=0)
   errs=[nrmse(x@v,y) for x,v,y in zip(probes,replay,target)];rel=nrmse(replay,mag);nnz=int(mag_mask.sum())
   ops=nnz//N if kind in ('sparse_mag','sparse_random') else (int(method.rsplit('r',1)[1])*D if kind=='pca' else D)
   rows.append(dict(condition=label,seed=seed,private_fraction=rho,method=method,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),mean_task_output_nrmse=float(np.mean(errs)),max_task_output_nrmse=float(np.max(errs)),nrmse_vs_magnitude_sparse=rel,magnitude_nonzeros=nnz,examples=N*P,optimizer_updates=0,fit_wall_s=(pca_wall if kind=='pca' else (ties_wall if kind=='merge' else 0.0)),active_compute_proxy=ops,status_note='synthetic task deltas; magnitude top-k control; sparse indices charged'))
 return rows
def main():
 rows=[]
 for label,seed in SEEDS.items():rows.extend(run_one(label,seed,ROOT/'source'/'payloads'))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('wrote',len(rows),'model-level rows')
if __name__=='__main__':main()
