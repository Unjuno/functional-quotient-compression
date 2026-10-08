#!/usr/bin/env python3
import csv,hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sp=importlib.util.spec_from_file_location('ma359_run',ROOT/'source'/'run.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
rows=list(csv.DictReader((ROOT/'artifacts'/'development.csv').open()));assert len(rows)==10
for a in rows:
 w=m.world(int(a['seed']));s,meta=m.state(a['method'],w);p,l,_=m.pack(s,meta);r=m.run(int(a['seed']),a['method'])
 assert len(p)==int(a['payload_bytes']) and hashlib.sha256(p).hexdigest()==a['sha256']
 for k in ('heldout_nMSE','train_nMSE','distinct_functions','transform_MAC_proxy'):assert str(r[k])==str(a[k])
assert {int(a['seed']) for a in rows}=={35901,35902}
print('development replay exact: 10 rows; fresh sealed')
