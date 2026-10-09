#!/usr/bin/env python3
import argparse,csv,io,json,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SEEDS=(58501,58502);RHOS=(0,.1,.3,.6);FRACS=(0,.05,.1,.25,.5,1.0);ROLES=12;S=4;D=4;RANK=2;Q=64
def pack(a):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_STORED) as z:
  for k in sorted(a):
   q=io.BytesIO();np.save(q,a[k],allow_pickle=False);z.writestr(zipfile.ZipInfo(k+'.npy',(1980,1,1,0,0,0)),q.getvalue())
 return b.getvalue()
def attn(q,k,v):
 scores=q@k.T/np.sqrt(D);scores-=scores.max(axis=1,keepdims=True);p=np.exp(scores);p/=p.sum(axis=1,keepdims=True);return p@v
def run(seed):
 rng=np.random.default_rng(seed);rows=[]
 for rho in RHOS:
  kb=rng.normal(size=(S,D))*.2;vb=rng.normal(size=(S,D))*.2;kf=rng.normal(size=(RANK,S,D))*.15;vf=rng.normal(size=(RANK,S,D))*.15;kc=rng.normal(size=(ROLES,RANK));vc=rng.normal(size=(ROLES,RANK))
  K=np.stack([kb+np.einsum('r,rsd->sd',kc[i],kf) for i in range(ROLES)]);V=np.stack([vb+np.einsum('r,rsd->sd',vc[i],vf) for i in range(ROLES)])
  mask=rng.random((2,ROLES,S,D))<rho;K+=rng.normal(0,.35,K.shape)*mask[0];V+=rng.normal(0,.35,V.shape)*mask[1]
  # Fit rank-2 shared role latent by separate SVD of flattened K and V role tables.
  def fitbank(X):
   mean=X.mean(0);u,s,v=np.linalg.svd((X-mean).reshape(ROLES,-1),full_matrices=False);basis=(s[:RANK,None]*v[:RANK]).reshape(RANK,S,D);code=u[:,:RANK];rec=mean[None]+np.einsum('nr,rsd->nsd',code,basis);return mean,basis,code,rec
  km,kb2,kc2,Krec=fitbank(K);vm,vb2,vc2,Vrec=fitbank(V);resK=K-Krec;resV=V-Vrec
  queries=rng.normal(size=(ROLES,Q,D));ref=np.stack([attn(queries[i],K[i],V[i]) for i in range(ROLES)])
  independent={'K':K.astype(np.float32),'V':V.astype(np.float32),'meta':np.frombuffer(b'independent',dtype=np.uint8)};indbytes=len(pack(independent))
  for method in ('native_shared','mirror_rank2','direct_coeff','private_sparse','independent'):
   fracs=FRACS if method=='private_sparse' else (0,)
   for frac in fracs:
    arrays={'km':km.astype(np.float32),'vm':vm.astype(np.float32),'kb':kb2.astype(np.float32),'vb':vb2.astype(np.float32),'kc':kc2.astype(np.float32),'vc':vc2.astype(np.float32),'meta':np.frombuffer(method.encode(),dtype=np.uint8)}
    kr=Krec.copy();vr=Vrec.copy()
    if method=='private_sparse' and frac>0:
     flat=np.concatenate([resK.ravel(),resV.ravel()]);n=max(1,int(frac*flat.size));idx=np.argpartition(np.abs(flat),-n)[-n:];vals=flat[idx];arrays['private_indices']=idx.astype(np.uint16);arrays['private_values']=vals.astype(np.float32);allrec=np.concatenate([kr.ravel(),vr.ravel()]);allrec[idx]+=vals;kr=allrec[:K.size].reshape(K.shape);vr=allrec[K.size:].reshape(V.shape)
    if method=='independent':kr,vr=K,V;payload=pack(independent)
    elif method=='native_shared':kr=np.repeat(km[None],ROLES,axis=0);vr=np.repeat(vm[None],ROLES,axis=0);payload=pack({'km':km.astype(np.float32),'vm':vm.astype(np.float32),'meta':np.frombuffer(b'native-shared',dtype=np.uint8)})
    else:payload=pack(arrays)
    pred=np.stack([attn(queries[i],kr[i],vr[i]) for i in range(ROLES)]);nmse=float(np.mean((pred-ref)**2)/max(np.mean(ref**2),1e-12));rows.append({'world':seed,'heterogeneity':rho,'method':method,'private_fraction':frac,'serialized_bytes':len(payload),'train_examples':0,'optimizer_updates':0,'attention_mac_proxy':ROLES*Q*S*D*2,'reconstruction_mac_proxy':ROLES*RANK*S*D*2,'wall_time_s':'0','attention_output_nmse':f'{nmse:.9g}','status_note':'oracle rank-2 role basis and magnitude top-k residual'})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'artifacts'/'dev_results.csv'));a=ap.parse_args();rows=[]
 for s in SEEDS:rows+=run(s)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 print(json.dumps({'rows':len(rows)},indent=2))
if __name__=='__main__':main()
