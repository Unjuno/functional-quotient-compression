#!/usr/bin/env python3
"""Replay MA-481 logical function outputs from serialized bank payloads."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from run import D,N,TRAIN,world

def replay(root):
 errors=[];checks=[];mx=0.
 for seed in (48101,48102):
  d=root/f'runs/dev_{seed}';doc=json.loads((d/'metrics.json').read_text());w=world(seed)
  for method,m in doc['methods'].items():
   path=d/f'{method}_payload.npz';raw=path.read_bytes()
   with np.load(path,allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files if not k.startswith('__')}
   if len(raw)!=m['inference_payload_bytes'] or hashlib.sha256(raw).hexdigest()!=m['payload_sha256']:errors.append(f'{seed}/{method}: bytes/hash mismatch')
   if method=='independent':mat=torch.tensor(a['function_matrices'])
   else:
    atoms=torch.tensor(a['shared_atoms']).reshape(-1,D*D)
    if method=='continuous':codes=torch.tensor(a['function_codes'])
    elif method=='int8':codes=torch.tensor(a['codes_int8']).float()*torch.tensor(a['code_scales'])
    else:codes=torch.tensor(a['vq_centers'])[torch.tensor(a['function_indices']).long()]
    mat=(codes@atoms).reshape(N,D,D)
   pred=torch.stack([w['queries'][e]@mat[e].T for e in range(N)]);rmse=float(torch.sqrt(torch.mean((pred[TRAIN:]-w['outputs'][TRAIN:])**2)))
   diff=abs(rmse-m['heldout_function_rmse']);mx=max(mx,diff)
   if diff>1e-7:errors.append(f'{seed}/{method}: function RMSE replay mismatch')
  for k in (16,64,128):
   a=(d/f'mirror_vq{k}_payload.npz').read_bytes();b=(d/f'native_vq{k}_payload.npz').read_bytes()
   if a!=b:errors.append(f'{seed}: native VQ{k} mismatch')
   else:checks.append({'seed':seed,'vq_size':k,'native_vq_sha256':hashlib.sha256(a).hexdigest(),'replay':True})
 report={'experiment_id':'MA-481','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':mx,'fresh_accessed':False,'checks':checks,'errors':errors};(root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=replay(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
