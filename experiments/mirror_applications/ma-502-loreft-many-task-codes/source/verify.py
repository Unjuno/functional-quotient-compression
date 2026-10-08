#!/usr/bin/env python3
"""Replay MA-502 serialized functions, task counts and actual payload sizes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
import run

def verify(root):
 errors=[];mx=0.
 for seed in (50201,50202):
  sd=root/'runs'/f'dev_{seed}';doc=json.loads((sd/'metrics.json').read_text())
  for tasks in run.TASK_COUNTS:
   w=run.world(seed,tasks);td=sd/f'tasks_{tasks}'
   for method in run.METHODS:
    obs=doc['sweep'][str(tasks)][method];p=td/f'{method}_payload.npz';raw=p.read_bytes()
    if len(raw)!=obs['inference_payload_bytes'] or hashlib.sha256(raw).hexdigest()!=obs['payload_sha256']:errors.append(f'{seed}/{tasks}/{method}: bytes/hash mismatch')
    with np.load(p,allow_pickle=False) as z:obj={k:torch.from_numpy(np.array(z[k])) for k in z.files if k!='schema_json'}
    met=run.score(method,obj,w)
    for key in ('relative_output_rmse','unique_functions','max_output_change_when_codes_zeroed'):
     d=abs(float(met[key])-float(obs[key]));mx=max(mx,d)
     if d>1e-6:errors.append(f'{seed}/{tasks}/{method}/{key}: replay mismatch')
    if method=='full_matrix' and met['relative_output_rmse']>1e-5:errors.append(f'{seed}/{tasks}: full matrix upper invalid')
    if method in ('shared_mirror','native_shared_code') and obs['unique_task_codes']!=tasks:errors.append(f'{seed}/{tasks}: task code collision')
   if (td/'shared_mirror_payload.npz').read_bytes()!=(td/'native_shared_code_payload.npz').read_bytes():errors.append(f'{seed}/{tasks}: native shared-code alias mismatch')
 report={'experiment_id':'MA-502','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':mx,'full_matrix_upper_checked':not errors,'native_shared_code_alias_checked':not errors,'task_counts_checked':list(run.TASK_COUNTS),'fresh_accessed':False,'errors':errors}
 (root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=verify(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
