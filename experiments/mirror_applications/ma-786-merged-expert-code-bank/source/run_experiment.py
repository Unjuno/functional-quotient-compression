"""Synthetic routing-weighted merged expert residual code sweep."""
import csv,hashlib,io
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=78601;G=8;M=4;N=G*M;D=16;R=2

def pack(**x):
 b=io.BytesIO();np.savez_compressed(b,**x);return b.getvalue()
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b**2)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);groups=np.repeat(np.arange(G),M).astype(np.int16);freq=np.tile(np.array([8,4,2,1],np.float32),G)
 held=np.tile(np.array([False,False,False,True]),G);train=~held;rows=[]
 for sigma in (0.,.10):
  base=rng.normal(0,.18,(G,D,D)).astype(np.float32);basis0=rng.normal(0,.06,(R,D,D)).astype(np.float32);codes0=rng.normal(0,.22,(N,R)).astype(np.float32);priv=rng.normal(0,sigma,(N,D,D)).astype(np.float32)
  w=base[groups]+np.einsum('nr,rij->nij',codes0,basis0)+priv
  centers=np.stack([(w[groups==g]*freq[groups==g,None,None]).sum(0)/freq[groups==g].sum() for g in range(G)]).astype(np.float32)
  res=w-centers[groups];_,_,vh=np.linalg.svd(res[train].reshape(train.sum(),-1),full_matrices=False);basis=vh[:R].reshape(R,D,D).astype(np.float32)
  coeff=np.einsum('nij,rij->nr',res,basis).astype(np.float32);rec=centers[groups]+np.einsum('nr,rij->nij',coeff,basis);rem=w-rec
  common={'centers':centers,'basis':basis,'member_codes':coeff,'frequencies':freq,'groups':groups,'member_ids':np.arange(N,dtype=np.int16)}
  view=pack(**common);rows.append({'seed':SEED,'heterogeneity_sigma':sigma,'method':'mirror_member_code_bank','private_rank':0,'serialized_bytes':len(view),'payload_sha256':hashlib.sha256(view).hexdigest(),'heldout_member_nrmse':nrmse(rec[held],w[held]),'frequency_weighted_nrmse':nrmse(rec*np.sqrt(freq[:,None,None]),w*np.sqrt(freq[:,None,None])),'inference_macs_per_member_token':D*D,'reconstruction_macs_total':G*R*D*D,'optimizer_updates':0,'status_note':'shared basis is native PCA/SVD'})
  native=pack(**common);assert native==view
  rows.append({'seed':SEED,'heterogeneity_sigma':sigma,'method':'native_pca_code_bank','private_rank':0,'serialized_bytes':len(native),'payload_sha256':hashlib.sha256(native).hexdigest(),'heldout_member_nrmse':nrmse(rec[held],w[held]),'frequency_weighted_nrmse':nrmse(rec*np.sqrt(freq[:,None,None]),w*np.sqrt(freq[:,None,None])),'inference_macs_per_member_token':D*D,'reconstruction_macs_total':G*R*D*D,'optimizer_updates':0,'status_note':'exact native PCA alias'})
  for q in (4,8,16):
   us=[];ss=[];vs=[]
   for x in rem:
    u,s,v=np.linalg.svd(x,full_matrices=False);us.append(u[:,:q]);ss.append(s[:q]);vs.append(v[:q])
   U=np.asarray(us,np.float32);S=np.asarray(ss,np.float32);V=np.asarray(vs,np.float32);pr=np.einsum('ndq,nq,nqk->ndk',U,S,V);out=rec+pr
   pay=pack(**common,private_u=U,private_s=S,private_v=V)
   rows.append({'seed':SEED,'heterogeneity_sigma':sigma,'method':'shared_plus_private_member_residual','private_rank':q,'serialized_bytes':len(pay),'payload_sha256':hashlib.sha256(pay).hexdigest(),'heldout_member_nrmse':nrmse(out[held],w[held]),'frequency_weighted_nrmse':nrmse(out*np.sqrt(freq[:,None,None]),w*np.sqrt(freq[:,None,None])),'inference_macs_per_member_token':D*D,'reconstruction_macs_total':G*R*D*D+N*2*D*q,'optimizer_updates':0,'status_note':'per-member SVD fallback'})
  full=pack(weights=w,frequencies=freq,groups=groups,member_ids=np.arange(N,dtype=np.int16))
  rows.append({'seed':SEED,'heterogeneity_sigma':sigma,'method':'independent_members','private_rank':D,'serialized_bytes':len(full),'payload_sha256':hashlib.sha256(full).hexdigest(),'heldout_member_nrmse':0.,'frequency_weighted_nrmse':0.,'inference_macs_per_member_token':D*D,'reconstruction_macs_total':0,'optimizer_updates':0,'status_note':'independent upper control'})
  merged=pack(centers=centers,frequencies=freq,groups=groups,member_ids=np.arange(N,dtype=np.int16))
  rows.append({'seed':SEED,'heterogeneity_sigma':sigma,'method':'routing_weighted_merge_only','private_rank':0,'serialized_bytes':len(merged),'payload_sha256':hashlib.sha256(merged).hexdigest(),'heldout_member_nrmse':nrmse(centers[groups][held],w[held]),'frequency_weighted_nrmse':nrmse(centers[groups]*np.sqrt(freq[:,None,None]),w*np.sqrt(freq[:,None,None])),'inference_macs_per_member_token':D*D,'reconstruction_macs_total':0,'optimizer_updates':0,'status_note':'weighted cluster centers only'})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');wr.writeheader();wr.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
