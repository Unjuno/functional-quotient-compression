#!/usr/bin/env python3
"""Replay MA-494 addresses from serialized expert bank and codebook."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from run import D,E,N,NOISE,METHODS,world,corrupt,decode

def verify(root):
 errors=[];mx=0.
 for seed in (49401,49402):
  d=root/'runs'/f'dev_{seed}';doc=json.loads((d/'metrics.json').read_text());w=world(seed)
  for method in METHODS:
   m=doc['methods'][method];p=d/f'{method}_payload.npz';raw=p.read_bytes()
   if len(raw)!=m['inference_payload_bytes'] or hashlib.sha256(raw).hexdigest()!=m['payload_sha256']:errors.append(f'{seed}/{method}: bytes/hash mismatch')
   with np.load(p,allow_pickle=False) as z:mat=torch.tensor(np.array(z['expert_matrices']));bits=np.unpackbits(np.array(z['packed_codebook']),bitorder='little')[:E*m['address_bits']].reshape(E,m['address_bits']);codes=torch.tensor(bits.astype(np.uint8))
   for rate in NOISE:
    target=torch.stack([w['queries'][i]@mat[w['requests'][i]].T for i in range(N)]);rec,_=corrupt(codes[w['requests']],rate,seed);idx=decode(method,rec,codes);pred=torch.stack([w['queries'][i]@mat[idx[i]].T for i in range(N)])
    rmse=float(torch.sqrt(torch.mean((pred-target)**2)));wrong=float((idx!=w['requests']).float().mean());coverage=int(torch.unique(idx).numel())/E;obs=m['noise_results'][str(rate)]
    diff=max(abs(rmse-obs['output_rmse']),abs(wrong-obs['wrong_route_rate']),abs(coverage-obs['decoded_expert_coverage']));mx=max(mx,diff)
    if diff>1e-7:errors.append(f'{seed}/{method}/{rate}: metric replay mismatch')
 report={'experiment_id':'MA-494','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':mx,'fresh_accessed':False,'errors':errors};(root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=verify(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
