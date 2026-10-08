#!/usr/bin/env python3
import csv,hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('ma356_run',ROOT/'source'/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=list(csv.DictReader((ROOT/'artifacts'/'development.csv').open()));assert len(rows)==8
for r in rows:
 seed=int(r['seed']);name=r['method'];w=m.world(seed);a,meta=m.states(w)[name];p,s,_=m.pack(a,meta)
 assert len(p)==int(r['payload_bytes']) and hashlib.sha256(p).hexdigest()==r['sha256']
 q=m.score(name,s,w);assert abs(q[0]-float(r['heldout_nMSE']))<1e-15 and q[1]==int(r['distinct_functions']) and q[2]==int(r['lookup_MAC_proxy'])
assert {int(r['seed']) for r in rows}=={35601,35602}
print('development replay exact: 8 payload/hash/metric rows; fresh sealed')
