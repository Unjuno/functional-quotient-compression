import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

import run

ROOT=Path(__file__).resolve().parents[1]
DEV=ROOT/'artifacts'/'development'
checked=0;max_diff=0.0
for result_path in sorted(DEV.glob('*.json')):
    rec=json.loads(result_path.read_text());seed=rec['seed'];base,models,_,x=run.world(seed)
    fit=run.aligned_deltas(base,models)
    for row in rec['summaries']:
        path=DEV/f"development_{seed}_{row['method']}.zip";raw=path.read_bytes()
        assert len(raw)==row['serialized_bytes']
        assert hashlib.sha256(raw).hexdigest()==row['payload_sha256']
        arrays=run.load_arrays(path)
        metrics=run.metrics(row['method'],arrays,base,models,x)
        d=abs(metrics['mean_nmse']-row['validation_mean_nmse']);max_diff=max(max_diff,d)
        assert d<=1e-15
        assert np.allclose(metrics['task_nmse'],row['validation_task_nmse'],rtol=0,atol=1e-15)
        checked+=1
print(json.dumps({'payloads_and_metric_rows_checked':checked,'max_abs_nMSE_difference':max_diff}))
