import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];d=json.loads((root/'runs/dev_metrics.json').read_text());n=0
for w in d['worlds']:
 for r in w['records']:
  b=(root/'runs'/r['file']).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];n+=1
print(f'verified {n}/{n} serialized payload records')
