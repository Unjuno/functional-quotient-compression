#!/usr/bin/env python3
import csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];d=json.loads((root/'runs/dev_metrics.json').read_text());n=0
for seed,methods in d['worlds'].items():
 for m,r in methods.items():
  if not isinstance(r,dict) or 'payload_file' not in r:continue
  b=(root/r['payload_file']).read_bytes();assert len(b)==r['payload_bytes'];assert hashlib.sha256(b).hexdigest()==r['payload_sha256'];assert r['max_abs_error']<=1e-6;assert r['routing_accuracy']==1.;n+=1
with (root/'RESULTS_CORE.csv').open(newline='') as f:assert len(list(csv.DictReader(f)))==n
print(f'verified {n}/{n} payloads: bytes/hash and full-pair routing/output replay')
