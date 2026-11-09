"""Synthetic pruning-plus-shared-residual expert recovery screen."""
import csv,hashlib,io
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=79101;N=16;D=12;R=1

def pack(**x):
 b=io.BytesIO();np.savez_compressed(b,**x);return b.getvalue()
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b**2)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);ret=np.arange(0,N,2);pr=np.arange(1,N,2);partner=ret.copy();held=pr[4:];trainpr=pr[:4];basis0=rng.normal(0,.08,(D,D)).astype(np.float32);basis0/=np.linalg.norm(basis0);rows=[]
 for sigma in (0.,.05):
  W=np.zeros((N,D,D),np.float32);W[ret]=rng.normal(0,.2,(len(ret),D,D)).astype(np.float32);coef=rng.normal(0,.4,len(pr)).astype(np.float32)
  W[pr]=W[partner]+coef[:,None,None]*basis0+rng.normal(0,sigma,(len(pr),D,D)).astype(np.float32)
  res=W[pr]-W[partner];_,_,vh=np.linalg.svd(res[:4].reshape(4,-1),full_matrices=False);basis=vh[:R].reshape(R,D,D).astype(np.float32);codes=np.einsum('nij,rij->nr',res,basis).astype(np.float32);rec=W[partner]+np.einsum('nr,rij->nij',codes,basis)
  route=np.column_stack([pr,partner]).astype(np.int16);basepay=pack(retained_weights=W[ret],retained_ids=ret,pruned_ids=pr,partner_map=partner,route_map=route)
  rows.append({'seed':SEED,'heterogeneity_sigma':sigma,'method':'prune_and_route_to_partner','private_rank':0,'serialized_bytes':len(basepay),'payload_sha256':hashlib.sha256(basepay).hexdigest(),'heldout_pruned_nrmse':nrmse(W[partner][4:],W[held]),'all_pruned_nrmse':nrmse(W[partner],W[pr]),'inference_macs_per_expert':D*D,'reconstruction_macs_total':0,'optimizer_updates':0,'status_note':'retained bank only'})
  shared=pack(retained_weights=W[ret],basis=basis,codes=codes,retained_ids=ret,pruned_ids=pr,partner_map=partner,route_map=route)
  rows.append({'seed':SEED,'heterogeneity_sigma':sigma,'method':'mirror_pruned_expert_views','private_rank':0,'serialized_bytes':len(shared),'payload_sha256':hashlib.sha256(shared).hexdigest(),'heldout_pruned_nrmse':nrmse(rec[4:],W[held]),'all_pruned_nrmse':nrmse(rec,W[pr]),'inference_macs_per_expert':D*D,'reconstruction_macs_total':len(pr)*R*D*D,'optimizer_updates':0,'status_note':'shared basis is native PCA'})
  native=pack(retained_weights=W[ret],basis=basis,codes=codes,retained_ids=ret,pruned_ids=pr,partner_map=partner,route_map=route);assert native==shared
  rows.append({'seed':SEED,'heterogeneity_sigma':sigma,'method':'native_pca_residual_codes','private_rank':0,'serialized_bytes':len(native),'payload_sha256':hashlib.sha256(native).hexdigest(),'heldout_pruned_nrmse':nrmse(rec[4:],W[held]),'all_pruned_nrmse':nrmse(rec,W[pr]),'inference_macs_per_expert':D*D,'reconstruction_macs_total':len(pr)*R*D*D,'optimizer_updates':0,'status_note':'exact native PCA alias'})
  us=[];ss=[];vs=[]
  for dlt in res:
   u,s,v=np.linalg.svd(dlt-np.einsum('r,rij->ij',np.einsum('rij,ij->r',basis,dlt),basis),full_matrices=False);us.append(u[:,:4]);ss.append(s[:4]);vs.append(v[:4])
  U=np.asarray(us,np.float32);S=np.asarray(ss,np.float32);V=np.asarray(vs,np.float32);priv=np.einsum('ndq,nq,nqk->ndk',U,S,V);prrec=rec+priv
  pp=pack(retained_weights=W[ret],basis=basis,codes=codes,private_u=U,private_s=S,private_v=V,retained_ids=ret,pruned_ids=pr,partner_map=partner,route_map=route)
  rows.append({'seed':SEED,'heterogeneity_sigma':sigma,'method':'shared_plus_private_rank4','private_rank':4,'serialized_bytes':len(pp),'payload_sha256':hashlib.sha256(pp).hexdigest(),'heldout_pruned_nrmse':nrmse(prrec[4:],W[held]),'all_pruned_nrmse':nrmse(prrec,W[pr]),'inference_macs_per_expert':D*D,'reconstruction_macs_total':len(pr)*(R+8)*D*D,'optimizer_updates':0,'status_note':'native private SVD residual'})
  full=pack(weights=W,expert_ids=np.arange(N,dtype=np.int16));rows.append({'seed':SEED,'heterogeneity_sigma':sigma,'method':'independent_full_bank','private_rank':D,'serialized_bytes':len(full),'payload_sha256':hashlib.sha256(full).hexdigest(),'heldout_pruned_nrmse':0.,'all_pruned_nrmse':0.,'inference_macs_per_expert':D*D,'reconstruction_macs_total':0,'optimizer_updates':0,'status_note':'full upper control'})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
