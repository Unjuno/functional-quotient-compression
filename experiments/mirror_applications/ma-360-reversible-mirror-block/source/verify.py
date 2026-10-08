#!/usr/bin/env python3
import csv,hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sp=importlib.util.spec_from_file_location('ma360run',ROOT/'source'/'run.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
rows=list(csv.DictReader((ROOT/'artifacts'/'development.csv').open()));assert len(rows)==10
for r in rows:
 expected=m.train(int(r['seed']),r['method'])
 for k in ('payload_bytes','payload_sha256','task_mse','saved_activation_bytes_one_batch','train_MAC_proxy'):assert str(expected[k])==str(r[k]),(r['seed'],r['method'],k,expected[k],r[k])
assert {int(r['seed']) for r in rows}=={36001,36002}
print('development replay exact: 10 rows; fresh sealed')
