"""Synthetic merged-expert shared/private residual storage frontier."""
import csv,hashlib,io
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=78501;N=16;G=4;D=12;R=2

def pack(**a):
 b=io.BytesIO();np.savez_compressed(b,**a);return b.getvalue()
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b**2)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);groups=np.repeat(np.arange(G),N//G).astype(np.int16)
 rows=[]; held=np.arange(N)%4==3; train=~held
 for sigma in (0.,.05,.20):
  centers0=rng.normal(0,.22,(G,D,D)).astype(np.float32)
  basis0=rng.normal(0,.08,(R,D,D)).astype(np.float32)
  coeff0=rng.normal(0,.25,(N,R)).astype(np.float32)
  private=rng.normal(0,sigma,(N,D,D)).astype(np.float32)
  weights=centers0[groups]+np.einsum('nr,rij->nij',coeff0,basis0)+private
  centers=np.stack([weights[groups==g].mean(axis=0) for g in range(G)]).astype(np.float32)
  residual=weights-centers[groups]
  _,_,vh=np.linalg.svd(residual[train].reshape(train.sum(),-1),full_matrices=False)
  basis=vh[:R].reshape(R,D,D).astype(np.float32)
  c=np.einsum('nij,rij->nr',residual,basis).astype(np.float32)
  rem=residual-np.einsum('nr,rij->nij',c,basis)
  payload=pack(centers=centers,basis=basis,coefficients=c,groups=groups,expert_ids=np.arange(N,dtype=np.int16))
  recon=centers[groups]+np.einsum('nr,rij->nij',c,basis)
  rows.append({'seed':SEED,'heterogeneity_sigma':sigma,'method':'mirror_shared_rank2','private_rank':0,'serialized_bytes':len(payload),'payload_sha256':hashlib.sha256(payload).hexdigest(),'heldout_operator_nrmse':nrmse(recon[held],weights[held]),'inference_macs_per_token_expert':D*D,'reconstruction_macs_total':G*R*D*D,'optimizer_updates':0,'status_note':'shared basis is native low-rank factorization'})
  native=pack(centers=centers,basis=basis,coefficients=c,groups=groups,expert_ids=np.arange(N,dtype=np.int16))
  assert payload==native
  rows.append({'seed':SEED,'heterogeneity_sigma':sigma,'method':'native_rank2_decompression','private_rank':0,'serialized_bytes':len(native),'payload_sha256':hashlib.sha256(native).hexdigest(),'heldout_operator_nrmse':nrmse(recon[held],weights[held]),'inference_macs_per_token_expert':D*D,'reconstruction_macs_total':G*R*D*D,'optimizer_updates':0,'status_note':'exact alias of shared-basis View'})
  for q in (2,4,8,12):
   us=[];ss=[];vs=[]
   for e in range(N):
    u,s,v=np.linalg.svd(rem[e],full_matrices=False);us.append(u[:,:q]);ss.append(s[:q]);vs.append(v[:q])
   U=np.asarray(us,dtype=np.float32);S=np.asarray(ss,dtype=np.float32);V=np.asarray(vs,dtype=np.float32)
   priv=np.einsum('ndq,nq,nqk->ndk',U,S,V)
   pred=recon+priv
   pay=pack(centers=centers,basis=basis,coefficients=c,private_u=U,private_s=S,private_v=V,groups=groups,expert_ids=np.arange(N,dtype=np.int16))
   rows.append({'seed':SEED,'heterogeneity_sigma':sigma,'method':'shared_plus_private_lowrank','private_rank':q,'serialized_bytes':len(pay),'payload_sha256':hashlib.sha256(pay).hexdigest(),'heldout_operator_nrmse':nrmse(pred[held],weights[held]),'inference_macs_per_token_expert':D*D,'reconstruction_macs_total':G*R*D*D+N*2*D*q,'optimizer_updates':0,'status_note':'native per-expert SVD residual factors'})
  independent=pack(weights=weights,groups=groups,expert_ids=np.arange(N,dtype=np.int16))
  rows.append({'seed':SEED,'heterogeneity_sigma':sigma,'method':'independent_full_bank','private_rank':D,'serialized_bytes':len(independent),'payload_sha256':hashlib.sha256(independent).hexdigest(),'heldout_operator_nrmse':0.0,'inference_macs_per_token_expert':D*D,'reconstruction_macs_total':0,'optimizer_updates':0,'status_note':'independent upper control'})
  merged=pack(centers=centers,groups=groups,expert_ids=np.arange(N,dtype=np.int16))
  rows.append({'seed':SEED,'heterogeneity_sigma':sigma,'method':'merge_only','private_rank':0,'serialized_bytes':len(merged),'payload_sha256':hashlib.sha256(merged).hexdigest(),'heldout_operator_nrmse':nrmse(centers[groups][held],weights[held]),'inference_macs_per_token_expert':D*D,'reconstruction_macs_total':0,'optimizer_updates':0,'status_note':'merged physical centers only'})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
