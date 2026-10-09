#!/usr/bin/env python3
"""Replay MA-488 shared/private function bank payloads."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from run import D,R,N,TRAIN,Q,RATES,METHODS,world

def verify(root):
 errors=[];checks=[];mx=0.
 for seed in (48801,48802):
  d=root/'runs'/f'dev_{seed}';doc=json.loads((d/'metrics.json').read_text())
  for p in RATES:
   w=world(seed,p)
   for method in METHODS:
    m=doc['heterogeneity'][str(p)][method];path=d/f'h{p}_{method}_payload.npz';raw=path.read_bytes()
    if len(raw)!=m['inference_payload_bytes'] or hashlib.sha256(raw).hexdigest()!=m['payload_sha256']:errors.append(f'{seed}/{p}/{method}: bytes/hash mismatch')
    with np.load(path,allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files if not k.startswith('__')}
    if method=='independent_float32':mat=torch.tensor(a['function_matrices'])
    elif method=='full_int8':mat=torch.tensor(a['matrices_int8']).float()*torch.tensor(a['scale'])
    else:
     atoms=torch.tensor(a['shared_atoms']).reshape(R,-1);codes=torch.tensor(a['shared_codes']);flat=codes@atoms
     if method in ('mirror_shared_private','native_shared_private'):
      idx=torch.tensor(a['private_indices']).long();flat[idx]+=torch.tensor(a['private_residuals'])
     mat=flat.reshape(N,D,D)
    pred=torch.stack([w['queries'][i]@mat[i].T for i in range(N)]);err=(pred[TRAIN:]-w['outputs'][TRAIN:]);rmse=float(torch.sqrt(torch.mean(err**2)));diff=abs(rmse-m['overall_heldout_rmse']);mx=max(mx,diff)
    if diff>1e-7:errors.append(f'{seed}/{p}/{method}: metric replay mismatch')
    mask=torch.zeros(N,dtype=torch.bool);mask[w['private_rows']]=True;ph=mask[TRAIN:]
    perr=float(torch.sqrt(torch.mean(err[ph]**2))) if bool(ph.any()) else 0.;pdiff=abs(perr-m['private_heldout_rmse']);mx=max(mx,pdiff)
    if pdiff>1e-7:errors.append(f'{seed}/{p}/{method}: private RMSE replay mismatch')
   a=(d/f'h{p}_mirror_shared_private_payload.npz').read_bytes();b=(d/f'h{p}_native_shared_private_payload.npz').read_bytes()
   if a!=b:errors.append(f'{seed}/{p}: native shared/private mismatch')
   else:checks.append({'seed':seed,'heterogeneity':p,'sha256':hashlib.sha256(a).hexdigest()})
 rep={'experiment_id':'MA-488','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':mx,'fresh_accessed':False,'native_shared_private_exact_alias':not errors,'checks':checks,'errors':errors};(root/'verification_report.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n');return rep

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=verify(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
