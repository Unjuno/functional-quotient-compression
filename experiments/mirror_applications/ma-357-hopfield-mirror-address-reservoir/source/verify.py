#!/usr/bin/env python3
import csv,hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sp=importlib.util.spec_from_file_location('ma357_run',ROOT/'source'/'run.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
rows=list(csv.DictReader((ROOT/'artifacts'/'development.csv').open()));assert len(rows)==27
for a in rows:
 r=m.run(int(a['seed']),int(a['K']),a['method'])
 for k in ['payload_bytes','sha256','query_accuracy','decoded_code_nMSE','lookup_MAC_proxy']:
  assert str(r[k])==str(a[k]),(a['seed'],a['K'],a['method'],k,r[k],a[k])
assert {int(a['seed']) for a in rows}=={35701,35702,35703}
print('development replay exact: 27 payload/hash/metric rows; fresh sealed')
