#!/usr/bin/env python3
import csv,hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sp=importlib.util.spec_from_file_location('ma364run',ROOT/'source'/'run.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
rows=list(csv.DictReader((ROOT/'artifacts'/'development.csv').open()));assert len(rows)==8
for a in rows:
 r=m.run(int(a['seed']),a['method'])
 for k in ('payload_bytes','sha256','mixed_NLL','mixed_accuracy','average_MAC_proxy'):assert str(r[k])==str(a[k])
assert {int(a['seed']) for a in rows}=={36401,36402}
print('development replay exact: 8 rows; fresh sealed')
