#!/usr/bin/env python3
"""Prefix-active specialists over one canonical cache; exact algebra and costs."""
import csv,io,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];T,D,N=128,16,8
SEEDS={'dev-69701':69701,'audit-69711':69711,'audit-69712':69712,'audit-69713':69713}
DWELLS=(1,16,128,512)
def plane_rot(angles):
 A=np.zeros((D,D),np.float32)
 for j,a in enumerate(angles):
  c,s=np.float32(np.cos(a)),np.float32(np.sin(a));A[2*j:2*j+2,2*j:2*j+2]=[[c,-s],[s,c]]
 return A
def npz(**a):
 b=io.BytesIO();np.savez(b,**a);return b.getvalue()
def softmax(x):
 z=x-x.max(axis=-1,keepdims=True);e=np.exp(z,dtype=np.float32);return e/e.sum(axis=-1,keepdims=True)
def run_one(label,seed,outdir):
 rng=np.random.default_rng(seed);outdir.mkdir(parents=True,exist_ok=True)
 q=rng.normal(size=(T,D)).astype(np.float32);k=rng.normal(size=(T,D)).astype(np.float32);v=rng.normal(size=(T,D)).astype(np.float32)
 mask=np.where(np.tril(np.ones((T,T),np.float32))>0,0.,-1e9).astype(np.float32)
 aa=rng.uniform(-.8,.8,size=(N,D//2)).astype(np.float32);ab=rng.uniform(-.8,.8,size=(N,D//2)).astype(np.float32)
 basep=softmax(q@k.T/np.sqrt(D)+mask);baseout=basep@v
 ks=[];vs=[];ys=[];lazyerr=[];nativeerr=[];directerr=[]
 for m in range(N):
  A=plane_rot(aa[m]);B=plane_rot(ab[m])
  km=k@A.T;vm=v@B.T
  p=softmax(q@km.T/np.sqrt(D)+mask);y=p@vm
  # Lazy exact right-transform algebra over the same physical K/V cache.
  qview=q@A
  pl=softmax(qview@k.T/np.sqrt(D)+mask);yl=(pl@v)@B.T
  # Direct native rotation coordinate control.
  An=plane_rot(aa[m]);Bn=plane_rot(ab[m]);yn=(softmax((q@An)@k.T/np.sqrt(D)+mask)@v)@Bn.T
  ks.append(km);vs.append(vm);ys.append(y)
  den=max(float(np.linalg.norm(y)),1e-30);lazyerr.append(float(np.linalg.norm(y-yl)/den));nativeerr.append(float(np.linalg.norm(yl-yn)/den));directerr.append(float(np.linalg.norm(y-baseout)/den))
 start=time.perf_counter()
 common={'positions':np.arange(T,dtype=np.int32),'metadata':np.array([T,D,N],np.int32),'causal':np.array([1],np.uint8)}
 shared=npz(K=k,V=v,view_key_angles=aa,view_value_angles=ab,**common)
 independent=npz(K_views=np.stack(ks),V_views=np.stack(vs),**common)
 # aLoRA prefix control leaves prefix outputs equal to base; it cannot realize prefix-active specialist behavior.
 alora_err=max(directerr)
 (outdir/f'{label}.shared.npz').write_bytes(shared);(outdir/f'{label}.independent.npz').write_bytes(independent)
 with np.load(io.BytesIO(shared),allow_pickle=False) as z:
  assert np.array_equal(z['K'],k) and np.array_equal(z['view_key_angles'],aa)
 # Rotation proxy: 4 MAC-equivalents/coordinate for query+output per view; cache map costs same per prefix token.
 lazy_per_decode=4*D*N;materialize_once=4*T*D*N
 rows=[]
 for dwell in DWELLS:
  rows.append({'condition':label,'seed':seed,'dwell_tokens':dwell,'method':'lazy_shared_vs_materialized_and_direct_reuse',
   'serialized_bytes':len(shared),'independent_bytes':len(independent),'shared_sha256':hashlib.sha256(shared).hexdigest(),
   'independent_sha256':hashlib.sha256(independent).hexdigest(),'train_tokens_or_examples':0,'optimizer_updates':0,
   'active_compute_proxy':lazy_per_decode*dwell,'materialize_once_compute_proxy':materialize_once,
   'transform_only_break_even_tokens':T,'wall_time_s':time.perf_counter()-start,
   'max_lazy_rel_error':max(lazyerr),'native_rotation_rel_error':max(nativeerr),'direct_cache_reuse_max_rel_error':max(directerr),
   'alora_prefix_error_vs_active_specialist':alora_err,
   'status_note':'synthetic; NPZ actual bytes; causal mask reconstructible from causal flag; common attention FLOPs excluded'})
 return rows
def main():
 out=ROOT/'source'/'payloads';rows=[]
 for k,s in SEEDS.items():rows.extend(run_one(k,s,out))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
