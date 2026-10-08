#!/usr/bin/env python3
"""Replay MA-498 route and function metrics from serialized codebooks."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np,torch
from run import D,E,N,SIGMAS,METHODS,world,decode

def verify(root):
 errors=[];mx=0.
 for seed in (49801,49802):
  d=root/'runs'/f'dev_{seed}';doc=json.loads((d/'metrics.json').read_text());w=world(seed)
  for method in METHODS:
   m=doc['methods'][method];p=d/f'{method}_payload.npz';raw=p.read_bytes()
   if len(raw)!=m['inference_payload_bytes'] or hashlib.sha256(raw).hexdigest()!=m['payload_sha256']:errors.append(f'{seed}/{method}: bytes/hash mismatch')
   with np.load(p,allow_pickle=False) as z:
    mats=torch.tensor(np.array(z['expert_matrices']));codes=torch.tensor(np.array(z['address_codebook'])) if 'address_codebook' in z else None
   if codes is not None:
    dd=torch.cdist(codes,codes);dd.fill_diagonal_(float('inf'));mind=float(dd.min())
   else:mind=1.
   if abs(mind-m['minimum_pairwise_code_distance'])>1e-7:errors.append(f'{seed}/{method}: min distance replay mismatch')
   for sigma in SIGMAS:
    gen=torch.Generator().manual_seed(seed+round(sigma*10000)+3498);noise=torch.randn(N,E,generator=gen)*sigma
    if method=='raw_id':base=((w['requests'][:,None]>>torch.arange(3))&1).float()*2-1;z=torch.zeros(N,E);z[:,:3]=base/math.sqrt(3)
    else:z=codes[w['requests']]
    idx=decode(method,z+noise,codes);pred=torch.stack([w['queries'][i]@mats[idx[i]].T for i in range(N)]);target=torch.stack([w['queries'][i]@mats[w['requests'][i]].T for i in range(N)])
    rmse=float(torch.sqrt(torch.mean((pred-target)**2)));wrong=float((idx!=w['requests']).float().mean());obs=m['noise_results'][str(sigma)];diff=max(abs(rmse-obs['output_rmse']),abs(wrong-obs['wrong_route_rate']));mx=max(mx,diff)
    if diff>1e-7:errors.append(f'{seed}/{method}/{sigma}: metric replay mismatch')
  if (d/'mirror_distance_reg8_payload.npz').read_bytes()!=(d/'native_metric8_payload.npz').read_bytes():errors.append(f'{seed}: native metric-learning alias mismatch')
 report={'experiment_id':'MA-498','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':mx,'fresh_accessed':False,'errors':errors};(root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=verify(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
