#!/usr/bin/env python3
"""Read-only integrity verifier, no model retraining and no files overwritten."""
from pathlib import Path
import csv,hashlib,json,sys,math
base=Path(__file__).resolve().parent
v=json.loads((base/'results/VERIFICATION.json').read_text())
expected={'development':('results/dev_raw.csv',27,[31,32,33]),
          'initial_fresh':('results/fresh_raw.csv',45,[301,302,303,304,305]),
          'postdiscovery_replication':('results/replica/fresh_raw.csv',45,[401,402,403,404,405])}
err=[]
for name,(rel,num,seeds) in expected.items():
    with (base/rel).open(newline='',encoding='utf8') as fp:rows=list(csv.DictReader(fp))
    if len(rows)!=num or sorted(set(int(x['seed']) for x in rows))!=seeds:err.append(f'{name} wrong rows/seeds')
    if len(set((r['method'],r['seed']) for r in rows))!=num:err.append(f'{name} duplicate')
    if any(int(row['trunk_invocations_per_forward'])!=1 or row['input_same_across_roles']!='True' for row in rows):err.append(f'{name} multiple trunks/input roles')
    if any(not math.isfinite(float(x['fresh_mean_nll'])) for x in rows):err.append(f'{name} loss NaN')
for rel,sha in {**v['source_sha256'],**v['result_sha256']}.items():
    if hashlib.sha256((base/rel).read_bytes()).hexdigest()!=sha:err.append(f'sha mismatch {rel}')
if not v['replay'] or v['hypothesis']['replica_margin_gate_pass']:
    err.append('replay or negative hypothesis verdict mismatch')
print(json.dumps({'MAT03_dev':27,'MAT03_fresh':45,'MAT03R_fresh':45,'original_tests':10,'replica_margin_gate':'FAIL','errors':err}))
sys.exit(bool(err))
