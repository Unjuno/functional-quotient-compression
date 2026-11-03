import hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
def test_freeze_hash_and_fresh_sealed():
 f=json.loads((P/'freeze.json').read_text()); assert hashlib.sha256((P/'PROTOCOL.json').read_bytes()).hexdigest()==f['protocol_sha256']; assert not f['fresh_accessed']
def test_controls_replay_and_payloads():
 x=json.loads((P/'source'/'development.json').read_text())
 assert len(x)==32
 for r in x:
  assert r['heldout_nrmse']<1e-8
  f=P/'source'/f"{r['seed']}_{r['method']}.npz"
  assert f.stat().st_size==r['serialized_bytes']
 for seed in [62401,62402]:
  rows=[r for r in x if r['seed']==seed]
  d=next(r for r in rows if r['method']=='direct_cavia');m=next(r for r in rows if r['method']=='mirror_inferred_code')
  assert abs(m['heldout_nrmse']-d['heldout_nrmse'])<1e-12 and m['serialized_bytes']>d['serialized_bytes'] and m['active_compute_proxy']>d['active_compute_proxy']
