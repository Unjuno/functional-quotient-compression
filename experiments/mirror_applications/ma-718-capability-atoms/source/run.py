from __future__ import annotations
import hashlib,io,time
import numpy as np
D,T,TRAIN,LAT=512,24,16,6

def pack(a):
 b=io.BytesIO();np.savez_compressed(b,**a);v=b.getvalue();return len(v),hashlib.sha256(v).hexdigest()
def run(seed=71801):
 rng=np.random.default_rng(seed);atoms,_=np.linalg.qr(rng.normal(size=(D,LAT)))
 coeff=rng.normal(size=(T,LAT));deltas=(atoms@coeff.T).T
 noise=rng.normal(size=deltas.shape);noise-=((noise@atoms)@atoms.T);deltas+=.02*noise/np.sqrt(D)
 train=deltas[:TRAIN];test=deltas[TRAIN:]
 _,_,vt=np.linalg.svd(train,full_matrices=False);U=vt.T
 requests=np.array([[16+i,16+(i+1)%8] for i in range(4)],dtype=np.int16)
 targets=np.stack([deltas[a]+deltas[b] for a,b in requests])
 methods=[]
 methods.append(('independent_dense',0,{'source_deltas':deltas.astype(np.float32),'request_ids':requests,'outputs':targets.astype(np.float32)},deltas[TRAIN:],targets))
 for name,mode in [('DARE_random_top10','random'),('DELLA_magnitude_top10','magnitude')]:
  k=int(D*.1);idx=np.zeros((T,k),dtype=np.int16);val=np.zeros((T,k),dtype=np.float32);rec=np.zeros_like(deltas)
  for t,x in enumerate(deltas):
   ids=rng.choice(D,k,replace=False) if mode=='random' else np.argpartition(np.abs(x),-k)[-k:]
   idx[t]=ids;val[t]=x[ids];rec[t,ids]=x[ids]
  out=np.stack([rec[a]+rec[b] for a,b in requests])
  methods.append((name,k,{'indices':idx,'values':val,'request_ids':requests,'outputs':out.astype(np.float32)},rec[TRAIN:],out))
 for rank in (2,4,6,8,12,16):
  B=U[:,:rank];codes=deltas@B;recon=codes@B.T;out=np.stack([recon[a]+recon[b] for a,b in requests])
  payload={'basis':B.astype(np.float32),'codes':codes.astype(np.float32),'request_ids':requests,'outputs':out.astype(np.float32)}
  methods.append(('native_PCA',rank,payload,recon[TRAIN:],out))
  methods.append(('Mirror_atoms',rank,payload,recon[TRAIN:],out))
 rows=[]
 for name,rank,payload,rec_test,out in methods:
  size,digest=pack(payload);hnr=float(np.linalg.norm(rec_test-test)/(np.linalg.norm(test)+1e-12));cnr=float(np.linalg.norm(out-targets)/(np.linalg.norm(targets)+1e-12))
  compute=int(D*rank*T if name in ('native_PCA','Mirror_atoms') else D*T)
  rows.append({'seed':seed,'method':name,'rank':rank,'serialized_bytes':size,'payload_sha256':digest,'heldout_nrmse':hnr,'composition_nrmse':cnr,'active_compute_proxy':compute,'wall_s':0.0,'status_note':'dev source bank; first 16 fit PCA, last 8 held out; whole-bank payload'})
 return rows
def save(rows,path):
 import csv
 with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=71801);p.add_argument('--out',default='RESULTS_CORE.csv');a=p.parse_args();t=time.perf_counter();rows=run(a.seed);dt=time.perf_counter()-t
 for x in rows:x['wall_s']=dt
 save(rows,a.out)
