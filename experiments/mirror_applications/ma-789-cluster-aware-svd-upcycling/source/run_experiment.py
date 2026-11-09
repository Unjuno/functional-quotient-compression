"""Synthetic cluster-aware expert SVD vs a shared global View basis."""
import csv,hashlib,io
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=78901;C=3;M=4;N=C*M;D=12;R=3

def pack(**x):
 b=io.BytesIO();np.savez_compressed(b,**x);return b.getvalue()
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b**2)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);groups=np.repeat(np.arange(C),M).astype(np.int16);held=np.tile([False,False,False,True],C);train=~held
 rows=[]
 for sigma in (0.,.05):
  parent=rng.normal(0,.2,(D,D)).astype(np.float32);B0=rng.normal(0,.07,(R,D,D)).astype(np.float32);cc=rng.normal(0,.3,(N,R)).astype(np.float32);cluster_bias=rng.normal(0,.04,(C,D,D)).astype(np.float32);noise=rng.normal(0,sigma,(N,D,D)).astype(np.float32)
  W=parent+cluster_bias[groups]+np.einsum('nr,rij->nij',cc,B0)+noise
  # Synthetic semantic activation clusters and nearest-centroid router.
  centers_x=np.asarray([[-1.,0.],[0.,1.],[1.,-1.]],np.float32);xtrain=np.concatenate([centers_x[g]+rng.normal(0,.18,(64,2)) for g in range(C)])
  learned_centers=np.stack([xtrain[i*64:(i+1)*64].mean(0) for i in range(C)])
  xtest=np.concatenate([centers_x[g]+rng.normal(0,.18,(128,2)) for g in range(C)]);labels=np.repeat(np.arange(C),128);pred=np.argmin(((xtest[:,None]-learned_centers[None])**2).sum(-1),axis=1);acc=float(np.mean(pred==labels))
  # Remove cluster means, fit one shared global basis only on training expert roles.
  cluster_means=np.stack([W[(groups==g)&train].mean(0) for g in range(C)]).astype(np.float32);res=W-cluster_means[groups]
  _,_,vh=np.linalg.svd(res[train].reshape(train.sum(),-1),full_matrices=False);B=vh[:R].reshape(R,D,D).astype(np.float32);codes=np.einsum('nij,rij->nr',res,B).astype(np.float32);recon=cluster_means[groups]+np.einsum('nr,rij->nij',codes,B)
  router=pack(centroids=learned_centers,cluster_ids=np.arange(C,dtype=np.int16))
  glob=pack(cluster_means=cluster_means,basis=B,codes=codes,groups=groups,router=router,expert_ids=np.arange(N,dtype=np.int16))
  rows.append({'seed':SEED,'private_noise_sigma':sigma,'method':'mirror_global_shared_basis','serialized_bytes':len(glob),'payload_sha256':hashlib.sha256(glob).hexdigest(),'heldout_operator_nrmse':nrmse(recon[held],W[held]),'router_accuracy':acc,'inference_macs_per_expert_token':D*D,'reconstruction_macs_total':C*R*D*D,'optimizer_updates':0,'status_note':'global basis equals native PCA'})
  native=pack(cluster_means=cluster_means,basis=B,codes=codes,groups=groups,router=router,expert_ids=np.arange(N,dtype=np.int16));assert native==glob
  rows.append({'seed':SEED,'private_noise_sigma':sigma,'method':'native_global_pca','serialized_bytes':len(native),'payload_sha256':hashlib.sha256(native).hexdigest(),'heldout_operator_nrmse':nrmse(recon[held],W[held]),'router_accuracy':acc,'inference_macs_per_expert_token':D*D,'reconstruction_macs_total':C*R*D*D,'optimizer_updates':0,'status_note':'exact native PCA alias'})
  local_b=[];local_c=[];local_rec=np.empty_like(W)
  for g in range(C):
   rr=res[(groups==g)&train];_,_,vhg=np.linalg.svd(rr.reshape(len(rr),-1),full_matrices=False);bg=vhg[:R].reshape(R,D,D).astype(np.float32);cg=np.einsum('nij,rij->nr',res[groups==g],bg).astype(np.float32);local_b.append(bg);local_c.append(cg);local_rec[groups==g]=cluster_means[g]+np.einsum('nr,rij->nij',cg,bg)
  cluster_payload=pack(cluster_means=cluster_means,cluster_bases=np.asarray(local_b),cluster_codes=np.asarray(local_c),groups=groups,router=router,expert_ids=np.arange(N,dtype=np.int16))
  rows.append({'seed':SEED,'private_noise_sigma':sigma,'method':'cluster_specific_svd','serialized_bytes':len(cluster_payload),'payload_sha256':hashlib.sha256(cluster_payload).hexdigest(),'heldout_operator_nrmse':nrmse(local_rec[held],W[held]),'router_accuracy':acc,'inference_macs_per_expert_token':D*D,'reconstruction_macs_total':C*R*D*D,'optimizer_updates':0,'status_note':'per-cluster native truncated SVD'})
  full=pack(weights=W,groups=groups,router=router,expert_ids=np.arange(N,dtype=np.int16))
  rows.append({'seed':SEED,'private_noise_sigma':sigma,'method':'independent_full_experts','serialized_bytes':len(full),'payload_sha256':hashlib.sha256(full).hexdigest(),'heldout_operator_nrmse':0.,'router_accuracy':acc,'inference_macs_per_expert_token':D*D,'reconstruction_macs_total':0,'optimizer_updates':0,'status_note':'independent upper control'})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
