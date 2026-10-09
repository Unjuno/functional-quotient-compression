import csv
import hashlib
import io
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((R/'DEVELOPMENT_RESULTS_A1.csv').open()))
assert len(rows)==864
for x in rows:
 p=Path('/workspace/fqc-ma369')/x['path']
 blob=p.read_bytes();assert len(blob)==int(x['payload_bytes']);assert hashlib.sha256(blob).hexdigest()==x['sha256']
 state=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
 if isinstance(state,bytes):
  state=torch.load(io.BytesIO(state),map_location='cpu',weights_only=False)
 assert isinstance(state,dict)
print('A1 payload byte/hash roundtrip passed for',len(rows),'records')
