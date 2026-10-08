#!/usr/bin/env python3
import csv,hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sp=importlib.util.spec_from_file_location('ma361run',ROOT/'source'/'run.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
rows=list(csv.DictReader((ROOT/'artifacts'/'development.csv').open()));assert len(rows)==8
for a in rows:
 r=m.run(int(a['seed']),a['method'])
 for k in ('payload_bytes','sha256','test_NLL','token_accuracy','active_MAC_proxy'):assert str(r[k])==str(a[k]),(a['seed'],a['method'],k,r[k],a[k])
assert {int(a['seed']) for a in rows}=={36101,36102}
print('development replay exact: 8 rows; fresh sealed')
