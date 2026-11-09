from __future__ import annotations
import hashlib,io,time
import numpy as np
D,T,TRAIN,R=512,16,8,8
KS=(0,8,16,32,64,128,256);RHOS=(0.,.05,.2,.5)
def pack(a):
 b=io.BytesIO();np.savez_compressed(b,**a);v=b.getvalue();return len(v),hashlib.sha256(v).hexdigest()
def run(seed=71901):
 rows=[];rng=np.random.default_rng(seed);shared,_=np.linalg.qr(rng.normal(size=(D,R)));coef=rng.normal(size=(T,R));base=(shared@coef.T).T
 priv=rng.normal(size=(T,D));signs=np.where(np.arange(T)[:,None]%2==0,1.,-1.);priv*=signs
 requests=np.array([[8+i,8+(i+1)%8] for i in range(4)],dtype=np.int16)
 for rho in RHOS:
  deltas=base+rho*priv;_,_,vt=np.linalg.svd(deltas[:TRAIN],full_matrices=False);B=vt[:R].T
  coeff=deltas@B;shared_recon=coeff@B.T;res=deltas-shared_recon
  target_pairs=np.stack([deltas[a]+deltas[b] for a,b in requests])
  for method in ('dense','shared_only'):
   rec=deltas.copy() if method=='dense' else shared_recon.copy()
   arrays={'deltas':deltas.astype(np.float32),'request_ids':requests,'individual_outputs':rec.astype(np.float32),'pair_outputs':np.stack([rec[a]+rec[b] for a,b in requests]).astype(np.float32)} if method=='dense' else {'basis':B.astype(np.float32),'codes':coeff.astype(np.float32),'request_ids':requests,'individual_outputs':rec.astype(np.float32),'pair_outputs':np.stack([rec[a]+rec[b] for a,b in requests]).astype(np.float32)}
   size,digest=pack(arrays)
   rows.append({'seed':seed,'heterogeneity':rho,'method':method,'private_entries_per_task':0,'private_fraction':0.,'serialized_bytes':size,'payload_sha256':digest,'heldout_nrmse':float(np.linalg.norm(rec[TRAIN:]-deltas[TRAIN:])/(np.linalg.norm(deltas[TRAIN:])+1e-12)),'pair_nrmse':float(np.linalg.norm(np.stack([rec[a]+rec[b] for a,b in requests])-target_pairs)/(np.linalg.norm(target_pairs)+1e-12)),'active_compute_proxy':int(D*T*(1 if method=='dense' else R)),'wall_s':0.,'status_note':'held-out tasks 8-15; full bundle including reconstructed outputs'})
  for k in KS:
   for mode in ('magnitude_private','random_private'):
    ids=np.full((T,k),-1,np.int16);vals=np.zeros((T,k),np.float32);private=np.zeros_like(res)
    for t in range(T):
     ix=(np.argpartition(np.abs(res[t]),-k)[-k:] if mode=='magnitude_private' else rng.choice(D,k,False)) if k else np.array([],int)
     ids[t,:len(ix)]=ix;vals[t,:len(ix)]=res[t,ix];private[t,ix]=res[t,ix]
    rec=shared_recon+private;pair=np.stack([rec[a]+rec[b] for a,b in requests]);arrays={'basis':B.astype(np.float32),'codes':coeff.astype(np.float32),'private_indices':ids,'private_values':vals,'request_ids':requests,'individual_outputs':rec.astype(np.float32),'pair_outputs':pair.astype(np.float32)}
    size,digest=pack(arrays)
    rows.append({'seed':seed,'heterogeneity':rho,'method':mode,'private_entries_per_task':k,'private_fraction':float(k/D),'serialized_bytes':size,'payload_sha256':digest,'heldout_nrmse':float(np.linalg.norm(rec[TRAIN:]-deltas[TRAIN:])/(np.linalg.norm(deltas[TRAIN:])+1e-12)),'pair_nrmse':float(np.linalg.norm(pair-target_pairs)/(np.linalg.norm(target_pairs)+1e-12)),'active_compute_proxy':int(D*T*R+T*k+4*k),'wall_s':0.,'status_note':'native residual coordinates; magnitude/random selection'})
 return rows
def save(rows,path):
 import csv
 with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=71901);p.add_argument('--out',default='RESULTS_CORE.csv');a=p.parse_args();t=time.perf_counter();rows=run(a.seed);dt=time.perf_counter()-t
 for x in rows:x['wall_s']=dt
 save(rows,a.out)
