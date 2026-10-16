"""Long-budget new-world serialized-model audit with NumPy float64 functional path."""
from __future__ import annotations
import json,hashlib,sys,csv
from pathlib import Path
import numpy as np,torch
from model import generate_world,infer_serialized
from audit import independent_reference
ROOT=Path(__file__).resolve().parents[1]


def main():
    for k,v in json.loads((ROOT/'PRE_LONG_LOCK.json').read_text())['sha256_files'].items():
        assert hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v,(k,'long source mismatch')
    p=ROOT/'results/long';runs=sorted((p/'runs').glob('*.json'))
    assert len(runs)==108,len(runs)
    max_diff=0.;max_error=0.;rows=[]
    for fname in runs:
        d=json.loads(fname.read_text());w=generate_world(d['world'],d['family'].replace('_full','').replace('_holdout',''))
        blob=(p/d['model_path']).read_bytes()
        assert len(blob)==d['serialized_bytes'] and hashlib.sha256(blob).hexdigest()==d['sha256']
        x,y=w['test'];replayed=infer_serialized(blob,x).numpy()
        numeric=independent_reference(blob,x.numpy())
        diff=float(np.abs(replayed-numeric).max());max_diff=max(max_diff,diff)
        assert diff<5e-5,(d['name'],diff)
        loss=float(np.mean((replayed-y.numpy())**2)/(w['scale']**2));delta=abs(loss-d['test_nmse'])
        assert delta < 2e-7 + 1e-6*abs(loss),(d['name'],delta)
        max_error=max(max_error,delta)
        assert d['updates']==6000 and d['examples_presented']==384000
        assert len(d['checkpoints'])==5, (d['name'],len(d['checkpoints']))
        rows.append({k:d[k] for k in ['name','family','world','seed','method','test_nmse','val_nmse','test_heldout_pairs_nmse','worst_pair_nmse','serialized_bytes','active_parameters','elapsed_s','sha256']})
    with (ROOT/'results/LONG_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    record={'stage':'LONG','models_verified':len(rows),'pre_long_source_hashes_match':True,'all_serialized_bytes_sha256_match':True,'max_abs_float64_np_vs_torch_prediction_delta':max_diff,'max_abs_metric_replay_error':max_error,'old_primary_worlds_unused':True,'new_worlds':[73211,73212,73213]}
    (ROOT/'results/LONG_AUDIT.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
