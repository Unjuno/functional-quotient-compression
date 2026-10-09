#!/usr/bin/env python3
import csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];d=json.loads((root/'runs/dev_metrics.json').read_text());n=0
for seed,r in d['worlds'].items():
 for m,p in r['payload_files'].items():
  b=(root/p).read_bytes();assert len(b)==r['payload_bytes'][m];assert hashlib.sha256(b).hexdigest()==r['payload_sha256'][m];assert r['payload_replay_max_abs'][m]==0.;n+=1
with (root/'RESULTS_CORE.csv').open(newline='') as f:assert len(list(csv.DictReader(f)))==n
print(f'verified {n}/{n} payload size/hash and exact code replay')
