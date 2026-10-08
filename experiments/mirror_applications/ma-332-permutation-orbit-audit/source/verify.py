import hashlib
import json
from pathlib import Path

import torch

import run

ROOT=Path(__file__).resolve().parents[1]
checked=0;max_diff=0.0
for split in ('development','fresh'):
    folder=ROOT/'artifacts'/split
    for rp in sorted(folder.glob('*.json')):
        d=json.loads(rp.read_text());seed=d['seed'];model,x,_=run.world(seed);ref=run.eval_model(run.arrays_for(model),x)
        for row in d['summaries']:
            if row['method']=='incoming_only_negative_control':
                bad=run.transformed(model, list(range(run.HIDDEN)))
                # metrics record is the deliberate incoming-only transform from runner.
                continue
            if row['method']=='shared_plus_paid_permutations':
                path=folder/f"{split}_{seed}_shared_plus_permutations.zip"
                raw=path.read_bytes();assert len(raw)==row['serialized_bytes'] and hashlib.sha256(raw).hexdigest()==row['payload_sha256']
                a=run.load(path);base=[torch.from_numpy(a[f'w{i}']).float() for i in range(4)]
                errs=[]
                for p in a['permutations']:
                    pred=run.forward(x,*run.transformed(base,p));errs.append(float(((pred-ref)**2).mean()))
                diff=float(max(errs)**0.5)
                assert abs(diff-row['max_abs_output_diff'])<1e-12
            elif row['method']=='independent_8_checkpoints':
                hashes=[];sizes=[];errs=[]
                for i in range(8):
                    path=folder/f"{split}_{seed}_independent_{i}.zip";raw=path.read_bytes();a=run.load(path)
                    sizes.append(len(raw));hashes.append(hashlib.sha256(raw).hexdigest())
                    pred=run.eval_model(a,x);errs.append(float(((pred-ref)**2).mean()))
                assert sum(sizes)==row['serialized_bytes'] and hashes==row['payload_sha256']
                diff=float(max(errs)**0.5);assert abs(diff-row['max_abs_output_diff'])<1e-12
            else:continue
            max_diff=max(max_diff,abs(diff-row['max_abs_output_diff']));checked+=1
print(json.dumps({'serialized_rows_checked':checked,'max_metric_replay_difference':max_diff}))
