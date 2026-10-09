#!/usr/bin/env python3
import csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];d=json.loads((root/'runs/dev_metrics.json').read_text());n=0
for seed,rows in d['worlds'].items():
 for name,r in rows.items():
  b=(root/r['file']).read_bytes();assert len(b)==r['bytes'];assert hashlib.sha256(b).hexdigest()==r['sha256'];n+=1
with (root/'RESULTS_CORE.csv').open(newline='') as f:assert len(list(csv.DictReader(f)))==n
print(f'verified {n}/{n} serialized payload size/hash records')
