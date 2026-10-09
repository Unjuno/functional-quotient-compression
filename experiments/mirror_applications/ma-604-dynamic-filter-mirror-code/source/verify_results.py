#!/usr/bin/env python3
import csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]; d=json.loads((root/'runs/dev_metrics.json').read_text()); n=0
for world,methods in d['worlds'].items():
 for name,r in methods.items():
  b=(root/r['payload_file']).read_bytes(); assert len(b)==r['payload_bytes']; assert hashlib.sha256(b).hexdigest()==r['payload_sha256']; assert r['replay_max_abs_error']==0.0; n+=1
with (root/'RESULTS_CORE.csv').open(newline='') as f: assert len(list(csv.DictReader(f)))==n
print(f'verified {n}/{n} payloads: actual bytes, SHA-256 and exact replay')
