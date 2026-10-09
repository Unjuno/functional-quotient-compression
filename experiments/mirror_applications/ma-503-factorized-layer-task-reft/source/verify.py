#!/usr/bin/env python3
"""Replay MA-503 crossed-function metrics from stored interventions."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
import run

def verify(root):
 errors=[];mx=0.
 compare=('heldout_pair_relative_output_rmse','visible_pair_relative_output_rmse','active_compute_proxy_per_example','unique_heldout_functions','max_code_causal_output_change')
 for seed in (50301,50302):
  sd=root/'runs'/f'dev_{seed}';doc=json.loads((sd/'metrics.json').read_text())
  for rho in run.RHOS:
   w=run.world(seed,rho);rd=sd/f'rho_{rho:g}'
   for method in run.METHODS:
    obs=doc['rhos'][str(rho)][method];p=rd/f'{method}_payload.npz';raw=p.read_bytes()
    if len(raw)!=obs['inference_payload_bytes'] or hashlib.sha256(raw).hexdigest()!=obs['payload_sha256']:errors.append(f'{seed}/{rho}/{method}: byte/hash mismatch')
    with np.load(p,allow_pickle=False) as z:obj={k:torch.from_numpy(np.array(z[k])) for k in z.files if k!='schema_json'}
    met=run.score(method,obj,w)
    for key in compare:
     before=obs[key];after=met[key]
     if before is None and after is None:continue
     diff=abs(float(before)-float(after));mx=max(mx,diff)
     if diff>1e-6:errors.append(f'{seed}/{rho}/{method}/{key}: replay mismatch {diff}')
    if method=='full_matrix_oracle' and met['heldout_pair_relative_output_rmse']>1e-5:errors.append(f'{seed}/{rho}: oracle upper invalid')
   if (rd/'factor_mirror_payload.npz').read_bytes()!=(rd/'native_bilinear_payload.npz').read_bytes():errors.append(f'{seed}/{rho}: native bilinear alias mismatch')
 report={'experiment_id':'MA-503','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':mx,'full_matrix_upper_checked':not errors,'native_bilinear_alias_checked':not errors,'heldout_pairs_checked':8,'fresh_accessed':False,'errors':errors}
 (root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=verify(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
