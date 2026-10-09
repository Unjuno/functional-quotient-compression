#!/usr/bin/env python3
"""Replay MA-482 function bank predictions from serialized states."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from run import D,N,TRAIN,world,METHODS,normalize

def verify(root):
 errors=[];checks=[];mx=0.
 for seed in (48201,48202):
  d=root/'runs'/f'dev_{seed}';doc=json.loads((d/'metrics.json').read_text());w=world(seed)
  for method in METHODS:
   m=doc['methods'][method];path=d/f'{method}_payload.npz';raw=path.read_bytes()
   if len(raw)!=m['inference_payload_bytes'] or hashlib.sha256(raw).hexdigest()!=m['payload_sha256']:errors.append(f'{seed}/{method}: bytes/hash mismatch')
   with np.load(path,allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files if not k.startswith('__')}
   base=normalize(method)
   if base=='independent':mat=torch.tensor(a['function_matrices'])
   else:
    atoms=torch.tensor(a['shared_atoms']).reshape(-1,D*D)
    if base=='continuous':codes=torch.tensor(a['function_codes'])
    elif base=='int8':codes=torch.tensor(a['codes_int8']).float()*torch.tensor(a['scales'])
    elif base.startswith('vq'):codes=torch.tensor(a['centers'])[0][torch.tensor(a['indices'])[:,0]]
    else:codes=torch.stack([torch.tensor(a['stage_centers'])[s][torch.tensor(a['stage_indices'])[:,s]] for s in range(a['stage_indices'].shape[1])]).sum(0)
    mat=(codes@atoms).reshape(N,D,D)
   pred=torch.stack([w['queries'][e]@mat[e].T for e in range(N)]);rmse=float(torch.sqrt(torch.mean((pred[TRAIN:]-w['outputs'][TRAIN:])**2)));diff=abs(rmse-m['heldout_function_rmse']);mx=max(mx,diff)
   if diff>1e-7:errors.append(f'{seed}/{method}: replay mismatch')
  for k in (8,16):
   for depth in (2,3,4):
    a=(d/f'mirror_rvq{k}x{depth}_payload.npz').read_bytes();b=(d/f'native_rvq{k}x{depth}_payload.npz').read_bytes()
    if a!=b:errors.append(f'{seed}: native residual VQ{k}x{depth} mismatch')
    else:checks.append({'seed':seed,'config':f'RVQ{k}x{depth}','sha256':hashlib.sha256(a).hexdigest()})
 report={'experiment_id':'MA-482','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':mx,'fresh_accessed':False,'native_rvq_exact_alias':not errors,'checks':checks,'errors':errors};(root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=verify(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
