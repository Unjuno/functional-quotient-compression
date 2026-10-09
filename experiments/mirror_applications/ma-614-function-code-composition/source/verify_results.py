#!/usr/bin/env python3
import csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];d=json.loads((root/'runs/dev_metrics.json').read_text());n=0
for w in d['worlds']:
 for r in w['records']:
  p=root/'runs'/r['file'];b=p.read_bytes();assert len(b)==r['bytes'];assert hashlib.sha256(b).hexdigest()==r['sha256'];n+=1
rows=list(csv.DictReader((root/'RESULTS_CORE.csv').open()));assert len(rows)==n;print(f'verified {n}/{n} serialized payload size/hash records')
