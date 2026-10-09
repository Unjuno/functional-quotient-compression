#!/usr/bin/env python3
import csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];data=json.loads((root/'runs/dev_metrics.json').read_text());n=0
for world,methods in data['worlds'].items():
 for mode,row in methods.items():
  blob=(root/row['payload_file']).read_bytes();assert len(blob)==row['payload_bytes'];assert hashlib.sha256(blob).hexdigest()==row['payload_sha256'];assert row['replay_max_abs_error']==0.;n+=1
with (root/'RESULTS_CORE.csv').open(newline='') as f:assert len(list(csv.DictReader(f)))==n
print(f'verified {n}/{n} payload sizes, SHA-256 hashes and exact replay')
