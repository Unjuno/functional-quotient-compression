#!/usr/bin/env python3
"""LME01 numerical replay, compares 2 complete untouched fresh worlds."""
import csv
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
from run_lme01 import train_one,METHODS
base=Path(__file__).resolve().parent
raw=list(csv.DictReader((base/'results/fresh_raw.csv').open(encoding='utf-8')))
ref={(int(r['seed']),r['method']):r for r in raw}
fields=('test_nll','test_worst_task_nll','test_accuracy','output_prob_std','train_nll','serializer_bytes','role_trainable_scalars','per_task_nll','per_task_accuracy','original_train_id_sha256','audit_id_sha256')
errors=[]
for seed in (701,705):
    for m in METHODS:
        r=train_one(seed,m,timing=False)
        o=ref[(seed,m)]
        for f in fields:
            a=str(r[f]);b=str(o[f])
            if a!=b:errors.append({'seed':seed,'method':m,'field':f,'observed':a,'ref':b})
    print(f'Replayed seed {seed}: {len(METHODS)} models',flush=True)
result={'schema_version':1,'seeds':[701,705],'methods':list(METHODS),'model_training_replays':2*len(METHODS),
'fields_compared':list(fields),'cells_compared':2*len(METHODS)*len(fields),'errors':errors,
'match':len(errors)==0,'hardware':'CPU PyTorch2.10+cpu float32 OPENBLAS_NUM_THREADS=1',
'limitation':'This tests within-environment deterministic replay, not independent hardware reproduction; P95 timings excluded.'}
p=(base/'results/REPLAY_VERIFICATION.json')
p.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print('Replay verdict:',result['match'],'comparison cells:',result['cells_compared'],'errors:',len(errors),flush=True)
if errors:sys.exit(1)
