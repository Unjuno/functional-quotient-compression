#!/usr/bin/env python3
"""Synthetic YOCO shared-cache layer-view attention frontier."""
import csv,io,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
T,D,R,L,B=96,16,4,4,2
SEEDS={'dev-69501':69501,'audit-69511':69511,'audit-69512':69512,'audit-69513':69513}
RHOS=(0.,.1,.25);RANKS=(0,1,2,4)
def npz(**a):
 b=io.BytesIO();np.savez(b,**a);return b.getvalue()
def softmax(x):
 z=x-x.max(axis=-1,keepdims=True);e=np.exp(z,dtype=np.float32);return e/e.sum(axis=-1,keepdims=True)
def attn(q,c,m):
 uk,uv=m
 kl=c@uk;vl=c@uv
 p=softmax(q@kl.T/np.sqrt(D))
 return p@vl
def lazy_attn(q,c,m):
 uk,uv=m
 qlat=q@uk.T
 p=softmax(qlat@c.T/np.sqrt(D))
 return (p@c)@uv
def run_one(label,seed,outdir):
 rng=np.random.default_rng(seed);outdir.mkdir(parents=True,exist_ok=True)
 raw=rng.normal(size=(2*R*D,B)).astype(np.float32);basis=np.linalg.qr(raw)[0].T.astype(np.float32)
 coef=rng.normal(size=(L,B)).astype(np.float32);shared=coef@basis
 res=rng.normal(size=(L,2*R*D)).astype(np.float32);res-=(res@basis.T)@basis
 res/=np.sqrt(np.mean(res**2,axis=1,keepdims=True));scale=np.sqrt(np.mean(shared**2,axis=1,keepdims=True))
 c=rng.normal(size=(T,R)).astype(np.float32);queries=rng.normal(size=(L,T,D)).astype(np.float32)
 rows=[];start=time.perf_counter()
 for rho in RHOS:
  full=(shared+np.float32(rho)*scale*res).astype(np.float32);maps=full.reshape(L,2,R,D)
  kviews=np.stack([c@m[0] for m in maps]);vviews=np.stack([c@m[1] for m in maps])
  independent=npz(K_views=kviews,V_views=vviews,metadata=np.array([T,D,L],np.int32))
  tied=np.mean(maps,axis=0)
  tied_err=[]
  for l in range(L):
   y=attn(queries[l],c,maps[l]);yh=attn(queries[l],c,tied)
   tied_err.append(float(np.linalg.norm(y-yh)/max(np.linalg.norm(y),1e-30)))
  residual=full-shared;u,s,vt=np.linalg.svd(residual,full_matrices=False)
  for rank in RANKS:
   q=min(rank,len(s));approx=(u[:,:q]*s[:q])@vt[:q] if q else np.zeros_like(residual)
   candidate=(shared+approx).reshape(L,2,R,D).astype(np.float32)
   errs=[];native_err=[]
   for l in range(L):
    y=attn(queries[l],c,maps[l]);yh=lazy_attn(queries[l],c,candidate[l])
    yn=lazy_attn(queries[l],c,candidate[l])
    errs.append(float(np.linalg.norm(y-yh)/max(np.linalg.norm(y),1e-30)))
    native_err.append(float(np.linalg.norm(yh-yn)/max(np.linalg.norm(yh),1e-30)))
   payload=npz(C_global=c,shared_basis=basis,layer_coefficients=coef,private_u=u[:,:q],private_s=s[:q],private_vt=vt[:q],metadata=np.array([T,D,R,L,B,q],np.int32))
   native=npz(C_global=c,shared_basis=basis,layer_coefficients=coef,private_u=u[:,:q],private_s=s[:q],private_vt=vt[:q],metadata=np.array([T,D,R,L,B,q],np.int32))
   tag=f'{label}_rho{rho:.2f}_q{q}';(outdir/f'{tag}.mirror.npz').write_bytes(payload)
   if q==0:(outdir/f'{label}_rho{rho:.2f}.independent.npz').write_bytes(independent)
   # Per full sequence: lazy MLA-style query/output projections + latent attention; independent operates at width D.
   candidate_mac=L*(2*T*D*R+2*T*T*R)+L*B*(2*R*D)+q*(L+2*R*D)
   independent_mac=2*L*T*T*D
   rows.append({'condition':label,'seed':seed,'rho':rho,'private_rank':q,'method':'global_yoco_cache_layer_view_latent_read',
      'serialized_bytes':len(payload),'independent_bytes':len(independent),'native_control_bytes':len(native),
      'payload_sha256':hashlib.sha256(payload).hexdigest(),'independent_sha256':hashlib.sha256(independent).hexdigest(),
      'native_control_sha256':hashlib.sha256(native).hexdigest(),'train_tokens_or_examples':0,'optimizer_updates':0,
      'active_compute_proxy':candidate_mac,'independent_compute_proxy':independent_mac,
      'max_layer_output_nrmse':max(errs),'hard_tied_read_max_nrmse':max(tied_err),'native_basis_alias_nrmse':max(native_err),
      'wall_time_s':0.0,'status_note':'synthetic FP32; actual NPZ; latent query/output absorption; common data generation excluded'})
 elapsed=time.perf_counter()-start
 for row in rows:row['wall_time_s']=elapsed/len(rows)
 return rows
def main():
 rows=[];out=ROOT/'source'/'payloads'
 for k,s in SEEDS.items():rows.extend(run_one(k,s,out))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
