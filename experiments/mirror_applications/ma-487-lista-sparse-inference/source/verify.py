#!/usr/bin/env python3
"""Replay MA-487 predictor outputs from serialized dictionary/LISTA payloads."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from run import D,K,N,TRAIN,Q,METHODS,world,encode

def verify(root):
 errors=[];mx=0.
 for seed in (48701,48702):
  d=root/'runs'/f'dev_{seed}';doc=json.loads((d/'metrics.json').read_text());w=world(seed)
  for method in METHODS:
   m=doc['methods'][method];path=d/f'{method}_payload.npz';raw=path.read_bytes()
   if len(raw)!=m['inference_payload_bytes'] or hashlib.sha256(raw).hexdigest()!=m['payload_sha256']:errors.append(f'{seed}/{method}: bytes/hash mismatch')
   with np.load(path,allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files if not k.startswith('__')}
   if method=='independent':mat=torch.tensor(a['function_matrices'])
   else:
    state={'depth':int(a['steps'].shape[0]) if 'steps' in a else 0}
    if 'steps' in a:state.update({'steps':torch.tensor(a['steps']),'thresholds':torch.tensor(a['thresholds'])})
    w2=dict(w);w2['atoms']=torch.tensor(a['shared_atoms'])
    codes,_=encode(method,w2,state if method.startswith('lista') else None);mat=(codes@w2['atoms'].reshape(K,-1)).reshape(N,D,D)
   pred=torch.stack([w['queries'][i]@mat[i].T for i in range(N)]);rmse=float(torch.sqrt(torch.mean((pred[TRAIN:]-w['outputs'][TRAIN:])**2)));diff=abs(rmse-m['heldout_function_rmse']);mx=max(mx,diff)
   if diff>1e-7:errors.append(f'{seed}/{method}: metric replay mismatch {diff}')
 report={'experiment_id':'MA-487','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':mx,'fresh_accessed':False,'errors':errors};(root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=verify(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
