#!/usr/bin/env python3
"""Group-shuffle structured transforms over shared weights."""
import csv,io,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D,N,E,B=64,16,128,8;G=D//B
SEEDS={'dev-70401':70401,'audit-70411':70411,'audit-70412':70412,'audit-70413':70413};STAGES=(1,2,4)
def npz(**a):
 z=io.BytesIO();np.savez(z,**a);return z.getvalue()
def shuffle_perm(stage):
 base=np.arange(D).reshape(G,B).T.reshape(-1)
 return base if stage%2==0 else base[::-1].copy()
def apply_gs(x,factors):
 y=x.copy()
 for s,blocks in enumerate(factors):
  v=y.reshape(y.shape[0],G,B)
  y=np.einsum('egj,gji->egi',v,blocks,optimize=True).reshape(y.shape)
  y=y[:,shuffle_perm(s)]
 return y
def run_one(label,seed,outdir):
 rng=np.random.default_rng(seed);outdir.mkdir(parents=True,exist_ok=True)
 W=rng.normal(size=(D,D)).astype(np.float32)/np.sqrt(D);X=rng.normal(size=(E,D)).astype(np.float32)
 rows=[]
 for stages in STAGES:
  factors=np.empty((N,stages,G,B,B),np.float32);Qviews=[];Y=[];errs=[];native_err=[]
  for n in range(N):
   for s in range(stages):
    for g in range(G):factors[n,s,g]=np.linalg.qr(rng.normal(size=(B,B)))[0].astype(np.float32)
   Q=apply_gs(np.eye(D,dtype=np.float32),factors[n]);Qviews.append(Q)
   y=apply_gs(X@W.T,factors[n]);Y.append(y)
   yd=(X@W.T)@Q
   errs.append(float(np.linalg.norm(y-yd)/max(np.linalg.norm(y),1e-30)))
   yn=apply_gs(X@W.T,factors[n]);native_err.append(float(np.linalg.norm(y-yn)/max(np.linalg.norm(y),1e-30)))
  Qviews=np.stack(Qviews);Y=np.stack(Y)
  denseW=np.stack([(W.T@q).T for q in Qviews]).astype(np.float32)
  payload=npz(W=W,group_blocks=factors,metadata=np.array([D,N,E,B,stages],np.int32),shuffle_rule=np.array([1],np.uint8))
  native=npz(W=W,group_blocks=factors,metadata=np.array([D,N,E,B,stages],np.int32),shuffle_rule=np.array([1],np.uint8))
  independent=npz(W_views=denseW,metadata=np.array([D,N,E],np.int32))
  oft=npz(W=W,orthogonal_views=Qviews,metadata=np.array([D,N,E],np.int32))
  for name,data in [('gsoft',payload),('native',native),('independent',independent),('dense_oft',oft)]:
   (outdir/f'{label}_s{stages}.{name}.npz').write_bytes(data)
  t0=time.perf_counter()
  for n in range(N):_y=apply_gs(X@W.T,factors[n])
  structured_wall=time.perf_counter()-t0
  t0=time.perf_counter()
  for n in range(N):_y=X@denseW[n].T
  dense_wall=time.perf_counter()-t0
  structured_mac=E*(D*D+stages*D*B);dense_mac=E*D*D
  rows.append({'condition':label,'seed':seed,'shuffle_stages':stages,'method':'shared_W_group_shuffle_views',
   'serialized_bytes':len(payload),'independent_bytes':len(independent),'native_gsoft_bytes':len(native),'dense_oft_bytes':len(oft),
   'payload_sha256':hashlib.sha256(payload).hexdigest(),'native_sha256':hashlib.sha256(native).hexdigest(),
   'train_tokens_or_examples':0,'optimizer_updates':0,'active_compute_proxy':structured_mac,'independent_compute_proxy':dense_mac,
   'max_output_nrmse':max(errs),'native_alias_nrmse':max(native_err),'structured_wall_s':structured_wall,'dense_wall_s':dense_wall,
   'status_note':'synthetic generated GSOFT maps; packed block payload; wall times are NumPy-only'})
 return rows
def main():
 rows=[];out=ROOT/'source'/'payloads'
 for k,s in SEEDS.items():rows.extend(run_one(k,s,out))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
