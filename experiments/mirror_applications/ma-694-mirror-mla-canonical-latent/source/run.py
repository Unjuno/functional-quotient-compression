#!/usr/bin/env python3
"""Synthetic MLA latent-cache shared-view/private-residual frontier."""
import csv,io,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
T,D,R,NV,B=96,16,4,8,2
SEEDS={'dev-69401':69401,'audit-69411':69411,'audit-69412':69412,'audit-69413':69413}
RHOS=(0.0,0.1,0.25); RANKS=(0,1,2,4,8)

def npz(**a):
 b=io.BytesIO();np.savez(b,**a);return b.getvalue()
def softmax(x):
 z=x-x.max(axis=-1,keepdims=True);e=np.exp(z,dtype=np.float32);return e/e.sum(axis=-1,keepdims=True)
def output(q,c,m):
 k=c@m[0];v=c@m[1]
 p=softmax((q@k.T)/np.sqrt(D))
 return p@v

def run_one(label,seed,outdir):
 rng=np.random.default_rng(seed)
 outdir.mkdir(parents=True,exist_ok=True)
 # Shared orthonormal row basis in the flattened joint K/V projection space.
 raw=rng.normal(size=(2*R*D,B)).astype(np.float32)
 basis=np.linalg.qr(raw)[0].T.astype(np.float32)
 coef=rng.normal(size=(NV,B)).astype(np.float32)
 shared=(coef@basis).astype(np.float32)
 residual=rng.normal(size=(NV,2*R*D)).astype(np.float32)
 residual-= (residual@basis.T)@basis
 residual/=np.sqrt(np.mean(residual**2,axis=1,keepdims=True))
 shared_scale=np.sqrt(np.mean(shared**2,axis=1,keepdims=True))
 q=rng.normal(size=(T,D)).astype(np.float32); c=rng.normal(size=(T,R)).astype(np.float32)
 rows=[];t0=time.perf_counter()
 for rho in RHOS:
  full=(shared+np.float32(rho)*shared_scale*residual).astype(np.float32)
  mats=full.reshape(NV,2,R,D)
  # SVD private residual codes; truncate at preregistered q.
  res=(full-shared).astype(np.float32)
  u,s,vt=np.linalg.svd(res,full_matrices=False)
  independent=npz(C=c,projections=mats,metadata=np.array([T,D,R,NV],dtype=np.int32))
  tied_map=np.mean(mats,axis=0)
  tied=npz(C=c,projection=tied_map,metadata=np.array([T,D,R],dtype=np.int32))
  tied_errs=[float(np.linalg.norm(output(q,c,mats[m])-output(q,c,tied_map))/max(np.linalg.norm(output(q,c,mats[m])),1e-30)) for m in range(NV)]
  for rank in RANKS:
   rr=min(rank,len(s))
   recon_res=(u[:,:rr]*s[:rr])@vt[:rr] if rr else np.zeros_like(res)
   rec=(shared+recon_res).reshape(NV,2,R,D).astype(np.float32)
   # Direct native shared-basis coefficient control reconstructs the same operators.
   native_rec=(coef@basis+recon_res).reshape(NV,2,R,D).astype(np.float32)
   native_errs=[]
   errs=[]
   for m in range(NV):
    y=output(q,c,mats[m]); yh=output(q,c,rec[m])
    errs.append(float(np.linalg.norm(y-yh)/max(np.linalg.norm(y),1e-30)))
    native_errs.append(float(np.linalg.norm(yh-output(q,c,native_rec[m]))/max(np.linalg.norm(yh),1e-30)))
   # Pay C, basis, view codes and all private SVD factors; native basis control uses same canonical payload bytes.
   payload=npz(C=c,shared_basis=basis,view_coefficients=coef,private_u=u[:,:rr],private_s=s[:rr],private_vt=vt[:rr],
               metadata=np.array([T,D,R,NV,B,rr],dtype=np.int32))
   native_payload=npz(C=c,shared_basis=basis,view_coefficients=coef,private_u=u[:,:rr],private_s=s[:rr],private_vt=vt[:rr],
                      metadata=np.array([T,D,R,NV,B,rr],dtype=np.int32))
   tag=f'{label}_rho{rho:.2f}_q{rr}'
   (outdir/f'{tag}.mirror.npz').write_bytes(payload)
   (outdir/f'{label}_rho{rho:.2f}.independent.npz').write_bytes(independent)
   # Full independent payload bytes share the same canonical latent cache.
   indep_bytes=len(independent)
   # Decode basis+residual map cost plus the same per-token latent projections.
   decode_mac=NV*(2*B*(2*R*D)+rr*(NV+2*R*D))
   project_mac=NV*T*2*R*D
   independent_mac=NV*T*2*R*D
   rows.append({'condition':label,'seed':seed,'rho':rho,'private_rank':rr,'method':'shared_mla_latent_basis_private_svd',
      'serialized_bytes':len(payload),'independent_bytes':indep_bytes,'native_control_bytes':len(native_payload),'tied_mla_bytes':len(tied),
      'payload_sha256':hashlib.sha256(payload).hexdigest(),'native_control_sha256':hashlib.sha256(native_payload).hexdigest(),
      'independent_sha256':hashlib.sha256(independent).hexdigest(),'train_tokens_or_examples':0,'optimizer_updates':0,
      'active_compute_proxy':decode_mac+project_mac,'independent_compute_proxy':independent_mac,
      'max_view_output_nrmse':max(errs),'native_basis_alias_nrmse':max(native_errs),'hard_tied_mla_max_nrmse':max(tied_errs),'wall_time_s':0.0,
      'status_note':'synthetic FP32; full NPZ includes C; native basis uses identical canonical representation'} )
 wall=time.perf_counter()-t0
 for row in rows: row['wall_time_s']=wall/len(rows)
 return rows

def main():
 out=ROOT/'source'/'payloads';rows=[]
 for label,seed in SEEDS.items(): rows.extend(run_one(label,seed,out))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
