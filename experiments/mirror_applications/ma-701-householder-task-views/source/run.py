#!/usr/bin/env python3
"""Householder task-view storage/compute screen with orthogonal controls."""
import csv,io,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D,N,E=64,16,128
SEEDS={'dev-70101':70101,'audit-70111':70111,'audit-70112':70112,'audit-70113':70113};RS=(1,2,4,8);BLOCK=8
def npz(**a):
 b=io.BytesIO();np.savez(b,**a);return b.getvalue()
def hh_apply(x,vs):
 y=x.copy()
 for u in vs:y=y-2*(y@u)[:,None]*u[None,:]
 return y
def dense_q(vs):
 return hh_apply(np.eye(D,dtype=np.float32),vs)
def block_project(Q):
 out=np.zeros_like(Q);packed=[]
 for a in range(0,D,BLOCK):
  block=Q[a:a+BLOCK,a:a+BLOCK];u,_,vt=np.linalg.svd(block);orth=(u@vt).astype(np.float32)
  out[a:a+BLOCK,a:a+BLOCK]=orth;packed.append(orth)
 return out,np.stack(packed)
def run_one(label,seed,outdir):
 rng=np.random.default_rng(seed);outdir.mkdir(parents=True,exist_ok=True)
 W=rng.normal(size=(D,D)).astype(np.float32)/np.sqrt(D);X=rng.normal(size=(E,D)).astype(np.float32)
 rows=[];start=time.perf_counter()
 for r in RS:
  vecs=rng.normal(size=(N,r,D)).astype(np.float32);vecs/=np.linalg.norm(vecs,axis=-1,keepdims=True)
  Qs=[];Wviews=[];blockQs=[];blockPacked=[];err=[];blockerr=[];nativeerr=[]
  for m in range(N):
   Q=dense_q(vecs[m]);Qs.append(Q)
   # W_m.T = W.T Q for row-vector application x W.T Q.
   Wm=(W.T@Q).T.astype(np.float32);Wviews.append(Wm)
   qb,qpack=block_project(Q);blockQs.append(qb);blockPacked.append(qpack)
   yh=hh_apply(X@W.T,vecs[m]);yi=X@Wm.T;yb=X@W.T@qb
   err.append(float(np.linalg.norm(yh-yi)/max(np.linalg.norm(yh),1e-30)))
   blockerr.append(float(np.linalg.norm(yh-yb)/max(np.linalg.norm(yh),1e-30)))
   yn=hh_apply(X@W.T,vecs[m]);nativeerr.append(float(np.linalg.norm(yh-yn)/max(np.linalg.norm(yh),1e-30)))
  Wviews=np.stack(Wviews);Qs=np.stack(Qs);blockQs=np.stack(blockQs);blockPacked=np.stack(blockPacked)
  payload=npz(W=W,reflection_vectors=vecs,metadata=np.array([D,N,r],np.int32))
  native=npz(W=W,reflection_vectors=vecs,metadata=np.array([D,N,r],np.int32))
  independent=npz(W_views=Wviews,metadata=np.array([D,N],np.int32))
  dense=npz(W=W,oft_orthogonal_views=Qs,metadata=np.array([D,N],np.int32))
  block=npz(W=W,block_orthogonal_blocks=blockPacked,metadata=np.array([D,N,BLOCK],np.int32))
  for method,data in [('householder',payload),('independent',independent),('dense_oft',dense),('block_oft',block)]:
   (outdir/f'{label}_r{r}.{method}.npz').write_bytes(data)
  mirror_mac=E*(D*D+2*r*D);ind_mac=E*D*D;dense_mac=E*2*D*D;block_mac=E*(D*D+D*BLOCK)
  rows.append({'condition':label,'seed':seed,'reflection_count':r,'method':'shared_W_householder_views',
   'serialized_bytes':len(payload),'independent_bytes':len(independent),'native_householder_bytes':len(native),
   'dense_oft_bytes':len(dense),'block_oft_bytes':len(block),
   'payload_sha256':hashlib.sha256(payload).hexdigest(),'native_sha256':hashlib.sha256(native).hexdigest(),
   'train_tokens_or_examples':0,'optimizer_updates':0,'active_compute_proxy':mirror_mac,'independent_compute_proxy':ind_mac,
   'dense_oft_compute_proxy':dense_mac,'block_oft_compute_proxy':block_mac,
   'max_householder_output_nrmse':max(err),'native_householder_alias_nrmse':max(nativeerr),
   'block_oft_output_nrmse':max(blockerr),'wall_time_s':time.perf_counter()-start,
   'status_note':'synthetic linear maps; serialized NPZ bytes; block-OFT is oracle block projection'})
 return rows
def main():
 rows=[];out=ROOT/'source'/'payloads'
 for k,s in SEEDS.items():rows.extend(run_one(k,s,out))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
