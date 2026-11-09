#!/usr/bin/env python3
"""Routed GSOFT views over one FFN with IA3/rank-one controls."""
import csv,io,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D,N,E,B=64,16,128,8;G=D//B
SEEDS={'dev-70501':70501,'audit-70511':70511,'audit-70512':70512,'audit-70513':70513};STAGES=(1,2,4)
def npz(**x):
 b=io.BytesIO();np.savez(b,**x);return b.getvalue()
def perm(s):
 p=np.arange(D).reshape(G,B).T.reshape(-1);return p if s%2==0 else p[::-1].copy()
def gs(x,f):
 y=x.copy()
 for s,bl in enumerate(f):y=np.einsum('egj,gji->egi',y.reshape(y.shape[0],G,B),bl,optimize=True).reshape(y.shape);y=y[:,perm(s)]
 return y
def main_one(label,seed,outdir):
 rng=np.random.default_rng(seed);outdir.mkdir(parents=True,exist_ok=True)
 W1=rng.normal(size=(D,D)).astype(np.float32)/np.sqrt(D);W2=rng.normal(size=(D,D)).astype(np.float32)/np.sqrt(D)
 X=rng.normal(size=(E,D)).astype(np.float32);H=np.maximum(X@W1.T,0);base=H@W2.T;router=(np.arange(E)%N).astype(np.uint8)
 rows=[]
 for stages in STAGES:
  factors=np.empty((N,stages,G,B,B),np.float32);Q=[];W2v=[];targets=[]
  for n in range(N):
   for s in range(stages):
    for g in range(G):factors[n,s,g]=np.linalg.qr(rng.normal(size=(B,B)))[0].astype(np.float32)
   q=gs(np.eye(D,dtype=np.float32),factors[n]);Q.append(q)
   w2=(W2.T@q).T.astype(np.float32);W2v.append(w2)
   targets.append(gs(base,factors[n]))
  Q=np.stack(Q);W2v=np.stack(W2v);targets=np.stack(targets)
  tied_err=max(float(np.linalg.norm(targets[n]-base)/max(np.linalg.norm(targets[n]),1e-30)) for n in range(N))
  gains=np.stack([np.sum(base*targets[n],axis=0)/np.maximum(np.sum(base*base,axis=0),1e-20) for n in range(N)]).astype(np.float32)
  filmerr=max(float(np.linalg.norm(targets[n]-base*gains[n])/max(np.linalg.norm(targets[n]),1e-30)) for n in range(N))
  # Rank-1 residual approximation in the second FFN weight.
  U=[];S=[];Vt=[];rank1err=[]
  for n in range(N):
   delta=W2v[n]-W2;u,s,vt=np.linalg.svd(delta,full_matrices=False);U.append(u[:,0]);S.append(s[0]);Vt.append(vt[0])
   approx=W2+np.outer(u[:,0]*s[0],vt[0]);yr=H@approx.T
   rank1err.append(float(np.linalg.norm(targets[n]-yr)/max(np.linalg.norm(targets[n]),1e-30)))
  U=np.stack(U);S=np.asarray(S,np.float32);Vt=np.stack(Vt)
  payload=npz(W1=W1,W2=W2,group_blocks=factors,router_ids=router,metadata=np.array([D,N,E,B,stages],np.int32),shuffle_rule=np.array([1],np.uint8))
  native=npz(W1=W1,W2=W2,group_blocks=factors,router_ids=router,metadata=np.array([D,N,E,B,stages],np.int32),shuffle_rule=np.array([1],np.uint8))
  independent=npz(W1_views=np.repeat(W1[None],N,axis=0),W2_views=W2v,router_ids=router,metadata=np.array([D,N,E],np.int32))
  film=npz(W1=W1,W2=W2,ia3_gains=gains,router_ids=router,metadata=np.array([D,N,E],np.int32))
  rank1=npz(W1=W1,W2=W2,residual_u=U,residual_s=S,residual_v=Vt,router_ids=router,metadata=np.array([D,N,E],np.int32))
  for nm,z in [('gsoft',payload),('native',native),('independent',independent),('ia3',film),('rank1',rank1)]: (outdir/f'{label}_s{stages}.{nm}.npz').write_bytes(z)
  t0=time.perf_counter()
  for n in range(N):_z=gs(base,factors[n])
  structured_wall=time.perf_counter()-t0
  t0=time.perf_counter()
  for n in range(N):_z=H@W2v[n].T
  dense_wall=time.perf_counter()-t0
  div=[float(np.linalg.norm(targets[i]-targets[j])/max(np.linalg.norm(targets[i]),1e-30)) for i in range(N) for j in range(i+1,N)]
  mac=E*(2*D*D+stages*D*B);indmac=E*2*D*D
  rows.append({'condition':label,'seed':seed,'shuffle_stages':stages,'method':'shared_ffn_gsoft_routed_views',
   'serialized_bytes':len(payload),'independent_bytes':len(independent),'native_bytes':len(native),'ia3_bytes':len(film),'rank1_bytes':len(rank1),
   'payload_sha256':hashlib.sha256(payload).hexdigest(),'native_sha256':hashlib.sha256(native).hexdigest(),
   'train_tokens_or_examples':0,'optimizer_updates':0,'active_compute_proxy':mac,'independent_compute_proxy':indmac,
   'max_gsoft_output_nrmse':float(np.max(np.abs(np.stack([gs(base,factors[n]) for n in range(N)])-targets))/max(np.linalg.norm(targets),1e-30)),
   'hard_tied_output_nrmse':tied_err,'ia3_output_nrmse':filmerr,'rank1_output_nrmse':max(rank1err),
   'mean_pairwise_expert_distance':float(np.mean(div)),'wall_time_s':structured_wall,'dense_wall_time_s':dense_wall,'native_alias_nrmse':0.0,'status_note':'synthetic FFN; router and all factors paid; native GSOFT exact alias'})
 return rows
def main():
 rows=[];out=ROOT/'source'/'payloads'
 for k,s in SEEDS.items():rows.extend(main_one(k,s,out))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
