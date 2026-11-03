import hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
def test_freeze_hash_and_fresh_sealed():
 f=json.loads((P/'freeze.json').read_text());assert hashlib.sha256((P/'PROTOCOL.json').read_bytes()).hexdigest()==f['protocol_sha256'];assert not f['fresh_accessed']
def test_native_alias_and_actual_payloads():
 rows=json.loads((P/'source'/'development.json').read_text())
 for seed in [62601,62602]:
  for method in ['mirror_shared_decoder','native_direct_shape_conditioner']:
   a=[r for r in rows if r['seed']==seed and r['method']==method]
   assert max(r['heldout_nrmse'] for r in a)<1e-5
  m=next(r for r in rows if r['seed']==seed and r['method']=='mirror_shared_decoder')
  n=next(r for r in rows if r['seed']==seed and r['method']=='native_direct_shape_conditioner')
  assert m['serialized_bytes']==n['serialized_bytes']
  assert all(abs(a['heldout_nrmse']-b['heldout_nrmse'])<1e-15 for a,b in zip([r for r in rows if r['seed']==seed and r['method']=='mirror_shared_decoder'],[r for r in rows if r['seed']==seed and r['method']=='native_direct_shape_conditioner']))
  assert m['active_compute_proxy']>n['active_compute_proxy']
 for r in rows:
  p=P/'source'/f"{r['seed']}_{r['method']}.npz";assert p.stat().st_size==r['serialized_bytes']
