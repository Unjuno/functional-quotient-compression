#!/usr/bin/env python3
import csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];data=json.loads((root/'runs/dev_metrics.json').read_text());count=0
for world,modes in data['worlds'].items():
 for mode,rows in modes.items():
  for name,row in rows.items():
   if 'file' not in row:continue
   b=(root/row['file']).read_bytes();assert len(b)==row['bytes'],(world,mode,name,'size');assert hashlib.sha256(b).hexdigest()==row['sha256'],(world,mode,name,'hash');assert row['replay_max_abs_error']<1e-6,(world,mode,name,'replay');count+=1
with (root/'RESULTS_CORE.csv').open(newline='') as f:table=list(csv.DictReader(f))
assert len(table)==20,len(table)
print(f'verified {count}/{count} saved payloads; size/hash match and exact reload replay')
