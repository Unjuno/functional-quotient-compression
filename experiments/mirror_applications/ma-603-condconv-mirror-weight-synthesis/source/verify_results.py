#!/usr/bin/env python3
import csv, hashlib, json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
metrics=json.loads((root/'runs/dev_metrics.json').read_text())
methods=('static','condconv','mirror','mlp_gate','oracle')
count=0
for world, items in metrics['worlds'].items():
    assert set(items)==set(methods)
    for method, row in items.items():
        p=root/row['payload_file']; blob=p.read_bytes()
        assert len(blob)==row['bytes'], (world,method,'byte length')
        assert hashlib.sha256(blob).hexdigest()==row['payload_sha256'], (world,method,'sha256')
        assert row['replay_max_abs_error']==0.0, (world,method,'replay')
        count+=1
with (root/'RESULTS_CORE.csv').open(newline='') as f:
    rows=list(csv.DictReader(f))
assert len(rows)==count
print(f'verified {count}/{count} serialized payloads: size/hash match and exact replay error 0')
