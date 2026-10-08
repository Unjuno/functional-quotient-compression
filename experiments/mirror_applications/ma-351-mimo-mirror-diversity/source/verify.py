#!/usr/bin/env python3
import csv, hashlib, importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma351_run',ROOT/'source'/'run.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
rows=list(csv.DictReader((ROOT/'artifacts'/'development.csv').open()))
assert len(rows)==10
for row in rows:
    seed=int(row['seed']); method=row['method']; _,train_data,test_data=mod.data(seed)
    state=mod.train(seed,method,train_data)
    payload,loaded,meta=mod.pack(state,{'method':method,'seed':seed,'updates':mod.UPDATES,'members':mod.MEMBERS,'format':'MA351-inference-v1'})
    assert len(payload)==int(row['payload_bytes'])
    assert hashlib.sha256(payload).hexdigest()==row['sha256']
    logits=mod.forward(test_data[0],loaded,method)
    assert float(np.max(np.abs(logits-mod.forward(test_data[0],state,method))))==0
    q=mod.metrics(logits,test_data[1])
    assert abs(q[0]-float(row['ensemble_nll']))<1e-12
    assert abs(q[3]-float(row['correctness_corr']))<1e-12
assert {int(r['seed']) for r in rows}=={35101,35102}
print('development replay exact: 10 payload/hash/metric rows; fresh seeds sealed')
