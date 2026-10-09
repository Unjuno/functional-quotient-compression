#!/usr/bin/env python3
"""Replay MA-508 probe, serialization and native PCA alias metrics."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
import run

def verify(root):
 errors=[];mx=0.
 keys=('heldout_probe_relative_rmse','heldout_off_target_probe_rmse','unique_heldout_views','max_output_change_when_code_zeroed','active_compute_proxy_per_example')
 for seed in (50801,50802):
  sd=root/'runs'/f'dev_{seed}';doc=json.loads((sd/'metrics.json').read_text())
  for rho in run.RHOS:
   w=run.world(seed,rho);rd=sd/f'rho_{rho:g}'
   for rank in run.RANKS:
    methods=doc['ranks'][str(rho)][str(rank)]
    for method in run.METHODS:
     obs=methods[method];p=rd/f'rank_{rank}_{method}_payload.npz';raw=p.read_bytes()
     if len(raw)!=obs['inference_payload_bytes'] or hashlib.sha256(raw).hexdigest()!=obs['payload_sha256']:errors.append(f'{seed}/{rho}/{rank}/{method}: byte/hash mismatch')
     with np.load(p,allow_pickle=False) as z:obj={k:torch.from_numpy(np.array(z[k])) for k in z.files if k!='schema_json'}
     met=run.score(method,obj,w)
     for k in keys:
      d=abs(float(met[k])-float(obs[k]));mx=max(mx,d)
      if d>1e-6:errors.append(f'{seed}/{rho}/{rank}/{method}/{k}: replay mismatch')
     if method=='explicit_vector' and met['heldout_probe_relative_rmse']>1e-5:errors.append(f'{seed}/{rho}/{rank}: explicit upper invalid')
    if (rd/f'rank_{rank}_shared_mirror_payload.npz').read_bytes()!=(rd/f'rank_{rank}_native_pca_payload.npz').read_bytes():errors.append(f'{seed}/{rho}/{rank}: native PCA alias mismatch')
 report={'experiment_id':'MA-508','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':mx,'explicit_vector_upper_checked':not errors,'native_pca_alias_checked':not errors,'all_ranks_and_rhos_checked':True,'fresh_accessed':False,'errors':errors}
 (root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=verify(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
