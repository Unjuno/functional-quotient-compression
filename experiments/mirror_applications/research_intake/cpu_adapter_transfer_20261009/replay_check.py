#!/usr/bin/env python3
"""Independent in-process re-training of one frozen fresh world per MAT pilot."""
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'source'))
import run_mat01,run_mat02

def load(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))

out={}
for name,seed,runner in [('mat01',101,run_mat01.run_one),('mat02',201,run_mat02.run_seed)]:
 records=load(BASE/'results'/name/'fresh_raw.csv')
 saved={(r['seed'],r['role'],r['method']):r for r in records}
 rows=runner(seed,'fresh')
 tol=0
 mismatch=[]
 keys=['nll','accuracy','serialized_npz_bytes','support_size','audit_size']
 for r in rows:
  old=saved[(str(r['seed']),r['role'],r['method'])]
  for key in keys:
   diff=abs(float(r[key])-float(old[key]));tol=max(tol,diff)
   if diff>1e-6:mismatch.append((r['role'],r['method'],key,diff))
  for k in ('base_dataset_sha256','dataset_sha256','source_svd_sha256','source_family_sha256'):
   if k in old and old[k]!=str(r[k]):mismatch.append((r['role'],r['method'],k,'hash mismatch'))
 out[name]={'replayed_seed':seed,'tested_rows':len(rows),'fields_checked':keys,'max_abs_difference':tol,'mismatches':mismatch,'passed':not mismatch,'timing_reproduced':False}
print(json.dumps(out,indent=2))
(BASE/'results'/'REPLAY_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n')
if any(not o['passed'] for o in out.values()):raise SystemExit(1)
