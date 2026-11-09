#!/usr/bin/env python3
"""Routed shared-FFN Householder views with hard-tie and FiLM controls."""
import csv,io,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D,N,E=64,16,128
SEEDS={'dev-70201':70201,'audit-70211':70211,'audit-70212':70212,'audit-70213':70213};RS=(1,2,4,8)
def npz(**a):
 b=io.BytesIO();np.savez(b,**a);return b.getvalue()
def hh(x,vs):
 y=x.copy()
 for u in vs:y=y-2*(y@u)[:,None]*u[None,:]
 return y
def run_one(label,seed,outdir):
 rng=np.random.default_rng(seed);outdir.mkdir(parents=True,exist_ok=True)
 W1=rng.normal(size=(D,D)).astype(np.float32)/np.sqrt(D);W2=rng.normal(size=(D,D)).astype(np.float32)/np.sqrt(D)
 X=rng.normal(size=(E,D)).astype(np.float32);base=np.maximum(X@W1.T,0)@W2.T
 router=(np.arange(E)%N).astype(np.uint8)
 rows=[];start=time.perf_counter()
 for r in RS:
  vec=rng.normal(size=(N,r,D)).astype(np.float32);vec/=np.linalg.norm(vec,axis=-1,keepdims=True)
  W2v=[];targets=[];errs=[];film_err=[]
  for m in range(N):
   q=hh(np.eye(D,dtype=np.float32),vec[m]);w2m=(W2.T@q).T.astype(np.float32);W2v.append(w2m)
   y=hh(base,vec[m]);targets.append(y)
   gain=np.sum(base*y,axis=0)/np.maximum(np.sum(base*base,axis=0),1e-20)
   yfilm=base*gain
   film_err.append(float(np.linalg.norm(y-yfilm)/max(np.linalg.norm(y),1e-30)))
   yind=np.maximum(X@W1.T,0)@w2m.T
   errs.append(float(np.linalg.norm(y-yind)/max(np.linalg.norm(y),1e-30)))
  targets=np.stack(targets);W2v=np.stack(W2v)
  payload=npz(W1=W1,W2=W2,reflection_vectors=vec,router_ids=router,metadata=np.array([D,N,E,r],np.int32))
  native=npz(W1=W1,W2=W2,reflection_vectors=vec,router_ids=router,metadata=np.array([D,N,E,r],np.int32))
  independent=npz(W1_views=np.repeat(W1[None],N,axis=0),W2_views=W2v,router_ids=router,metadata=np.array([D,N,E],np.int32))
  # Oracle ridge-free diagonal FiLM gains are computed per expert and charged.
  gains=np.stack([np.sum(base*targets[m],axis=0)/np.maximum(np.sum(base*base,axis=0),1e-20) for m in range(N)]).astype(np.float32)
  film=npz(W1=W1,W2=W2,film_gains=gains,router_ids=router,metadata=np.array([D,N,E],np.int32))
  for name,data in [('householder',payload),('native',native),('independent',independent),('film',film)]:
   (outdir/f'{label}_r{r}.{name}.npz').write_bytes(data)
  div=[]
  for i in range(N):
   for j in range(i+1,N):div.append(float(np.linalg.norm(targets[i]-targets[j])/max(np.linalg.norm(targets[i]),1e-30)))
  base_err=max(float(np.linalg.norm(targets[m]-base)/max(np.linalg.norm(targets[m]),1e-30)) for m in range(N))
  mirror_mac=E*(2*D*D+2*r*D);ind_mac=E*2*D*D;film_mac=E*(2*D*D+D)
  rows.append({'condition':label,'seed':seed,'reflection_count':r,'method':'shared_ffn_householder_routed_views',
   'serialized_bytes':len(payload),'independent_bytes':len(independent),'native_bytes':len(native),'film_bytes':len(film),
   'payload_sha256':hashlib.sha256(payload).hexdigest(),'native_sha256':hashlib.sha256(native).hexdigest(),
   'train_tokens_or_examples':0,'optimizer_updates':0,'active_compute_proxy':mirror_mac,'independent_compute_proxy':ind_mac,'film_compute_proxy':film_mac,
   'max_householder_output_nrmse':max(errs),'hard_tied_output_nrmse':base_err,'film_output_nrmse':max(film_err),
   'native_alias_nrmse':0.0,'mean_pairwise_expert_output_distance':float(np.mean(div)),'wall_time_s':time.perf_counter()-start,
   'status_note':'synthetic routed FFN; route IDs, weights and view codes paid; native Householder exact alias'})
 return rows
def main():
 rows=[];out=ROOT/'source'/'payloads'
 for k,s in SEEDS.items():rows.extend(run_one(k,s,out))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
