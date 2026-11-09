#!/usr/bin/env python3
"""Synthetic task delta compression with DARE, PCA/native controls and merge points."""
import csv,hashlib,io
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D,N,R,PROBES=2048,16,8,128
SEEDS={'dev-71101':71101,'audit-71111':71111,'audit-71112':71112,'audit-71113':71113}
RHOS=(0.0,0.05,0.2);KEEP=.1;TIES_TRIM=.2
def pack(**a):
 b=io.BytesIO();np.savez(b,**a);return b.getvalue()
def dare(delta,rng):
 mask=rng.random(delta.shape)<KEEP
 return delta*mask/KEEP,mask
def sparse_payload(masked,mask):
 offsets=[0];indices=[];values=[]
 for row,m in zip(masked,mask):
  ix=np.flatnonzero(m);indices.extend(ix.tolist());values.extend(row[ix].tolist());offsets.append(len(indices))
 return pack(offsets=np.asarray(offsets,np.int32),indices=np.asarray(indices,np.uint16),values=np.asarray(values,np.float32),metadata=np.asarray([N,D],np.int32))
def decode_sparse(p):
 out=np.zeros((N,D),np.float32)
 for i in range(N):
  a,b=int(p['offsets'][i]),int(p['offsets'][i+1]);out[i,p['indices'][a:b].astype(int)]=p['values'][a:b]
 return out
def nrmse(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
def make_ties(deltas):
 trimmed=deltas.copy();k=int(D*(1-TIES_TRIM))
 for i in range(N):
  thresh=np.partition(np.abs(trimmed[i]),-k)[-k];trimmed[i,np.abs(trimmed[i])<thresh]=0
 signs=np.sign(trimmed.sum(axis=0));keep=(trimmed*signs[None,:])>0
 num=keep.sum(axis=0);return np.divide((trimmed*keep).sum(axis=0),np.maximum(num,1))
def run_one(label,seed,outdir,rhos=RHOS):
 rng=np.random.default_rng(seed);outdir.mkdir(parents=True,exist_ok=True);rows=[]
 basis,_=np.linalg.qr(rng.normal(size=(D,R)))
 for rho in rhos:
  codes=rng.normal(size=(N,R));shared=codes@basis.T;deltas=shared.copy()
  if rho:
   private=rng.normal(size=(N,D));private*=rho*np.linalg.norm(shared,axis=1,keepdims=True)/np.maximum(np.linalg.norm(private,axis=1,keepdims=True),1e-30);deltas+=private
  probes=[rng.normal(size=(PROBES,D)).astype(np.float32) for _ in range(N)]
  targets=[x@d for x,d in zip(probes,deltas)]
  t0=__import__('time').perf_counter();dared=np.zeros_like(deltas);masks=np.zeros_like(deltas,dtype=bool)
  for n in range(N):dared[n],masks[n]=dare(deltas[n],rng)
  dare_wall=__import__('time').perf_counter()-t0;dared_payload=sparse_payload(dared,masks)
  t0=__import__('time').perf_counter();mean=dared.mean(axis=0);u,s,vt=np.linalg.svd(dared-mean,full_matrices=False);pca_wall=__import__('time').perf_counter()-t0
  models={'independent_dense':(pack(deltas=deltas.astype(np.float32),metadata=np.asarray([N,D],np.int32)),deltas),'DARE_sparse':(dared_payload,dared)}
  models['raw_shared_basis']=(pack(basis=basis.astype(np.float32),codes=codes.astype(np.float32),metadata=np.asarray([N,D,R],np.int32)),shared)
  for rank in (1,2,4,8):
   pcodes=(dared-mean)@vt[:rank].T;recon=mean+pcodes@vt[:rank]
   pca=pack(mean=mean.astype(np.float32),basis=vt[:rank].astype(np.float32),codes=pcodes.astype(np.float32),metadata=np.asarray([N,D,rank],np.int32))
   models[f'DARE_PCA_r{rank}']=(pca,recon);models[f'native_PCA_r{rank}']=(pca,recon.copy())
  t0=__import__('time').perf_counter();task_arithmetic=deltas.mean(axis=0);merge_wall=__import__('time').perf_counter()-t0
  t0=__import__('time').perf_counter();ties=make_ties(deltas);ties_wall=__import__('time').perf_counter()-t0
  models['task_arithmetic_mean']=(pack(merged=task_arithmetic.astype(np.float32),metadata=np.asarray([D],np.int32)),np.repeat(task_arithmetic[None,:],N,axis=0))
  models['TIES_trim20']=(pack(merged=ties.astype(np.float32),metadata=np.asarray([D,TIES_TRIM],np.float32)),np.repeat(ties[None,:],N,axis=0))
  nnz=int(masks.sum())
  for method,(payload,recon) in models.items():
   (outdir/f'{label}_rho{rho:g}.{method}.npz').write_bytes(payload)
   with np.load(io.BytesIO(payload),allow_pickle=False) as q:
    if method=='independent_dense': replay=q['deltas']
    elif method=='DARE_sparse': replay=decode_sparse(q)
    elif method=='raw_shared_basis': replay=q['codes']@q['basis'].T
    elif method.startswith('DARE_PCA') or method.startswith('native_PCA'): replay=q['mean']+q['codes']@q['basis']
    else: replay=np.repeat(q['merged'][None,:],N,axis=0)
   dared_replay=decode_sparse(np.load(io.BytesIO(dared_payload),allow_pickle=False))
   task_errs=[nrmse(x@r,y) for x,r,y in zip(probes,replay,targets)]
   dare_rel=nrmse(replay,dared_replay)
   compute=(nnz if method=='DARE_sparse' else (int(method.rsplit('r',1)[1])*D if method.startswith('DARE_PCA') or method.startswith('native_PCA') else (R*D if method=='raw_shared_basis' else (N*D if method=='independent_dense' else D))))
   interference=float(np.mean(task_errs)) if method in ('task_arithmetic_mean','TIES_trim20') else 0.0
   rows.append(dict(condition=label,seed=seed,private_delta_fraction=rho,method=method,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),mean_task_output_nrmse=float(np.mean(task_errs)),max_task_output_nrmse=float(np.max(task_errs)),nrmse_vs_DARE=dare_rel,dare_nonzeros=nnz,examples=N*PROBES,optimizer_updates=0,fit_wall_s=(dare_wall if method=='DARE_sparse' else (pca_wall if method.startswith('DARE_PCA') or method.startswith('native_PCA') else (merge_wall if method=='task_arithmetic_mean' else (ties_wall if method=='TIES_trim20' else 0.0)))),active_compute_proxy=compute,merge_interference_nrmse=interference,status_note='synthetic shared-basis task deltas; DARE masks serialized as uint16 indices/fp32 values'))
 return rows
def main():
 rows=[]
 for label,seed in SEEDS.items():rows.extend(run_one(label,seed,ROOT/'source'/'payloads'))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('wrote',len(rows),'model-level rows')
if __name__=='__main__':main()
