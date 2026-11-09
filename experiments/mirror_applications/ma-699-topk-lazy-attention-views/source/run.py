#!/usr/bin/env python3
"""Top-k attention views over a canonical cache; synthetic exactness/cost screen."""
import csv,io,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];T,D,N=64,16,8
SEEDS={'dev-69901':69901,'audit-69911':69911,'audit-69912':69912,'audit-69913':69913};KS=(1,2,4,8)
def rot(ang):
 a=np.zeros((D,D),np.float32)
 for j,t in enumerate(ang):
  c,s=np.float32(np.cos(t)),np.float32(np.sin(t));a[2*j:2*j+2,2*j:2*j+2]=[[c,-s],[s,c]]
 return a
def npz(**x):
 b=io.BytesIO();np.savez(b,**x);return b.getvalue()
def sm(x):
 z=x-x.max(axis=-1,keepdims=True);e=np.exp(z,dtype=np.float32);return e/e.sum(axis=-1,keepdims=True)
def outputs(q,k,v,A,B,mask):
 km=k@A.T;vm=v@B.T
 p=sm(q@km.T/np.sqrt(D)+mask)
 return p@vm
def run_one(label,seed,outdir):
 rng=np.random.default_rng(seed);outdir.mkdir(parents=True,exist_ok=True)
 q=rng.normal(size=(T,D)).astype(np.float32);k=rng.normal(size=(T,D)).astype(np.float32);v=rng.normal(size=(T,D)).astype(np.float32)
 mask=np.where(np.tril(np.ones((T,T),np.float32))>0,0.,-1e9).astype(np.float32)
 ak=rng.uniform(-.7,.7,size=(N,D//2)).astype(np.float32);av=rng.uniform(-.7,.7,size=(N,D//2)).astype(np.float32)
 As=[rot(a) for a in ak];Bs=[rot(a) for a in av]
 material=[outputs(q,k,v,As[m],Bs[m],mask) for m in range(N)]
 native=[outputs(q,k,v,rot(ak[m]),rot(av[m]),mask) for m in range(N)]
 # Ring schedule provides deterministic routing; pay selected logical IDs explicitly.
 rows=[];start=time.perf_counter()
 Kviews=np.stack([k@a.T for a in As]);Vviews=np.stack([v@b.T for b in Bs])
 for topk in KS:
  ids=np.stack([(np.arange(T)+i)%N for i in range(topk)],axis=1).astype(np.uint8)
  matout=np.zeros((T,D),np.float32);lazyout=np.zeros_like(matout);natout=np.zeros_like(matout)
  for row in range(T):
   for idx in ids[row]:
    m=int(idx);matout[row]+=material[m][row]/topk
    qview=q[row:row+1]@As[m]
    probs=sm(qview@k.T/np.sqrt(D)+mask[row:row+1])
    lazyout[row]+=(probs@v@Bs[m].T)[0]/topk
    qn=q[row:row+1]@rot(ak[m]);pn=sm(qn@k.T/np.sqrt(D)+mask[row:row+1])
    natout[row]+=(pn@v@rot(av[m]).T)[0]/topk
  den=max(float(np.linalg.norm(matout)),1e-30)
  err=float(np.linalg.norm(matout-lazyout)/den);nerr=float(np.linalg.norm(lazyout-natout)/max(float(np.linalg.norm(lazyout)),1e-30))
  # Pairwise diversity of constituent view outputs on queries, not independent capacity.
  ds=[]
  for i in range(N):
   for j in range(i+1,N):ds.append(float(np.linalg.norm(material[i]-material[j])/max(np.linalg.norm(material[i]),1e-30)))
  # All view coordinates and routing indices are paid on both alternatives.
  common={'selected_ids':ids,'metadata':np.array([T,D,N,topk],np.int32),'causal':np.array([1],np.uint8)}
  shared=npz(K=k,V=v,key_angles=ak,value_angles=av,**common)
  independent=npz(K_views=Kviews,V_views=Vviews,key_angles=ak,value_angles=av,**common)
  (outdir/f'{label}_k{topk}.shared.npz').write_bytes(shared);(outdir/f'{label}_k{topk}.independent.npz').write_bytes(independent)
  independent_mac=2*topk*T*T*D
  transform_mac=4*topk*T*D
  rows.append({'condition':label,'seed':seed,'top_k':topk,'method':'topk_lazy_vs_independent_materialized',
   'serialized_bytes':len(shared),'independent_bytes':len(independent),'shared_sha256':hashlib.sha256(shared).hexdigest(),
   'independent_sha256':hashlib.sha256(independent).hexdigest(),'train_tokens_or_examples':0,'optimizer_updates':0,
   'active_compute_proxy':independent_mac+transform_mac,'independent_compute_proxy':independent_mac,
   'compute_multiple_of_single_attention':topk*(1+2/T),'max_topk_output_rel_error':err,'native_phase_rel_error':nerr,
   'mean_pairwise_view_output_distance':float(np.mean(ds)),'wall_time_s':time.perf_counter()-start,
   'status_note':'synthetic FP32; selected IDs paid; common attention MAC plus lazy query/output transform MAC'})
 return rows
def main():
 rows=[];out=ROOT/'source'/'payloads'
 for label,s in SEEDS.items():rows.extend(run_one(label,s,out))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
