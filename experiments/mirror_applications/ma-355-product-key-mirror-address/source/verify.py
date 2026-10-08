#!/usr/bin/env python3
import csv,hashlib,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('ma355_run',ROOT/'source'/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=list(csv.DictReader((ROOT/'artifacts'/'development.csv').open()));assert len(rows)==10
for r in rows:
 seed=int(r['seed']);method=r['method'];w=m.bank(seed);a,meta=m.methods(w)[method];payload,s,_=m.pack(a,meta)
 assert len(payload)==int(r['payload_bytes']) and hashlib.sha256(payload).hexdigest()==r['sha256']
 nerr,nuniq,rm=m.score(seed,method,s,w)
 assert nerr==float(r['heldout_nMSE']) and nuniq==int(r['distinct_functions']) and rm==int(r['routing_MAC_proxy'])
assert {int(r['seed']) for r in rows}=={35501,35502}
print('amended development replay exact: 10 rows; invalid run excluded; fresh sealed')
