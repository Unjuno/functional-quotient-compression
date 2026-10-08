#!/usr/bin/env python3
import csv,hashlib,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('ma353_run',ROOT/'source'/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=list(csv.DictReader((ROOT/'artifacts'/'development.csv').open()));assert len(rows)==10
for r in rows:
 seed=int(r['seed']);method=r['method'];w,j,d,x,xood,y,ptrue,pood,_=m.world(seed);arrays,meta=m.posterior(method,w,j,d);payload,state,loaded=m.pack(arrays,meta)
 assert len(payload)==int(r['payload_bytes']) and hashlib.sha256(payload).hexdigest()==r['sha256']
 q=m.metrics(m.predict(method,state,x),y,ptrue)
 assert abs(q[0]-float(r['predictive_nll']))<1e-12 and abs(q[3]-float(r['calibration_mse']))<1e-12
 assert abs(m.metrics(m.predict(method,state,xood),(pood>=.5).astype('f4'),pood)[0]-float(r['ood_nll']))<1e-12
assert {int(r['seed']) for r in rows}=={35301,35302}
print('development replay exact: 10 payload/hash/metric rows; fresh seeds sealed')
