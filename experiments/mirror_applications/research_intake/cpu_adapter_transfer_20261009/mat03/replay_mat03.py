#!/usr/bin/env python3
"""Retest selected frozen MAT03/MAT03R seed/method cells, excluding jittery CPU wall time."""
from pathlib import Path
import csv,sys,hashlib,json
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'source'))
import run_mat03
import numpy as np
checks=[]
for stage,path,seed,methods in [
 ('MAT03',R/'results/fresh_raw.csv',301,['native_multihead','mirror_hadamard4','native_hadamard_linear4']),
 ('MAT03R',R/'results/replica/fresh_raw.csv',401,['native_multihead','mirror_hadamard4','native_hadamard_linear4'])
]:
 with path.open(newline='') as fp:reference={(int(row['seed']),row['method']):row for row in csv.DictReader(fp)}
 for method in methods:
  obs=run_mat03.train_one(seed,method)
  ref=reference[seed,method]
  fields=['fresh_mean_nll','fresh_worst_task_nll','fresh_accuracy','serialized_npz_bytes','member_scalars','trunk_invocations_per_forward','out_prob_std']
  maxdev=max(abs(float(obs[k])-float(ref[k])) for k in fields)
  ok=maxdev<=1e-8 and obs['dataset_sha256']==ref['dataset_sha256']
  checks.append(dict(stage=stage,seed=seed,method=method,fields=fields,max_abs_dev=maxdev,pass_same=float(ok),latency_replay=False))
  assert ok,(stage,method,maxdev)
print(json.dumps({'checked_rows':len(checks),'all_pass':all(x['pass_same'] for x in checks),'rows':checks},indent=2))
(R/'results/REPLAY_VERIFICATION.json').write_text(json.dumps({'all_pass':True,'checked_rows':len(checks),'rows':checks},indent=2)+'\n')
