import hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
def test_freeze_and_sealed_fresh():
 f=json.loads((P/'freeze.json').read_text());assert hashlib.sha256((P/'PROTOCOL.json').read_bytes()).hexdigest()==f['protocol_sha256'];assert not f['fresh_accessed']
def test_native_control_exact_alias_and_storage():
 rows=json.loads((P/'source'/'development.json').read_text())
 for s in [62701,62702]:
  m=[r for r in rows if r['seed']==s and r['method']=='shared_mirror_decoder'];n=[r for r in rows if r['seed']==s and r['method']=='native_periodic_activation']
  assert all(r['heldout_nrmse']<1e-5 for r in m+n)
  assert m[0]['serialized_bytes']==n[0]['serialized_bytes']
  for r in m+n:
   q=P/'source'/f"{s}_{r['method']}.npz"
   assert q.stat().st_size==r['serialized_bytes']
  with np.load(P/'source'/f'{s}_shared_mirror_decoder.npz') as a,np.load(P/'source'/f'{s}_native_periodic_activation.npz') as b:
   assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
  curves=next(r for r in rows if r['seed']==s and r['method']=='independent_sampled_curves')
  assert m[0]['serialized_bytes']<.25*curves['serialized_bytes']
