#!/usr/bin/env python3
import csv,hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parents[1];d=json.loads((r/'runs/dev_metrics.json').read_text());n=0
for world,methods in d['worlds'].items():
 for m,x in methods.items():
  b=(r/x['payload_file']).read_bytes();assert len(b)==x['payload_bytes'];assert hashlib.sha256(b).hexdigest()==x['payload_sha256'];assert x['replay_max_abs_error']==0.;n+=1
with (r/'RESULTS_CORE.csv').open(newline='') as f:assert len(list(csv.DictReader(f)))==n
print(f'verified {n}/{n} payloads: bytes, SHA-256 and exact replay')
