#!/usr/bin/env python3
"""Replay MA-486 serialized sparse and dense logical functions."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from run import D,K,N,TRAIN,S,Q,METHODS,world

def verify(root):
 errors=[];checks=[];mx=0.
 for seed in (48601,48602):
  d=root/'runs'/f'dev_{seed}';doc=json.loads((d/'metrics.json').read_text());w=world(seed)
  for method in METHODS:
   m=doc['methods'][method];p=d/f'{method}_payload.npz';raw=p.read_bytes()
   if len(raw)!=m['serialized_payload_bytes'] or hashlib.sha256(raw).hexdigest()!=m['payload_sha256']:errors.append(f'{seed}/{method} bytes/hash mismatch')
   with np.load(p,allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files if not k.startswith('__')}
   if method=='independent':mat=torch.tensor(a['function_matrices'])
   else:
    atoms=torch.tensor(a['shared_atoms']).reshape(K,-1)
    if method=='dense_float':coef=torch.tensor(a['dense_codes'])
    elif method=='dense_int8':coef=torch.tensor(a['codes_int8']).float()*torch.tensor(a['scale'])
    elif method.endswith('float'):
     idx=torch.tensor(a['sparse_indices']).long();val=torch.tensor(a['sparse_values']);coef=torch.zeros(N,K);coef.scatter_add_(1,idx,val)
    else:
     idx=torch.tensor(a['sparse_indices']).long();val=torch.tensor(a['sparse_values_int8']).float()*torch.tensor(a['scale']);coef=torch.zeros(N,K);coef.scatter_add_(1,idx,val)
    mat=(coef@atoms).reshape(N,D,D)
   pred=torch.stack([w['queries'][i]@mat[i].T for i in range(N)]);rmse=float(torch.sqrt(torch.mean((pred[TRAIN:]-w['outputs'][TRAIN:])**2)));diff=abs(rmse-m['heldout_function_rmse']);mx=max(mx,diff)
   if diff>1e-7:errors.append(f'{seed}/{method} metric replay mismatch')
  for a,b in [('mirror_sparse_float','native_omp_float'),('mirror_sparse_int8','native_omp_int8')]:
   x=(d/f'{a}_payload.npz').read_bytes();y=(d/f'{b}_payload.npz').read_bytes()
   if x!=y:errors.append(f'{seed} native OMP mismatch: {a}')
   else:checks.append({'seed':seed,'pair':[a,b],'sha256':hashlib.sha256(x).hexdigest()})
 rep={'experiment_id':'MA-486','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':mx,'fresh_accessed':False,'native_omp_exact_alias':not errors,'checks':checks,'errors':errors};(root/'verification_report.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n');return rep

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=verify(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
