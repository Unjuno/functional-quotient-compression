import hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
def test_protocol_frozen_and_fresh_sealed():
 f=json.loads((P/'freeze.json').read_text());assert hashlib.sha256((P/'PROTOCOL.json').read_bytes()).hexdigest()==f['protocol_sha256'];assert not f['fresh_accessed']
def test_native_lowrank_control_alias():
 rows=json.loads((P/'source'/'development.json').read_text())
 for s in [63501,63502]:
  m=next(r for r in rows if r['seed']==s and r['method']=='Mirror_code_editor');n=next(r for r in rows if r['seed']==s and r['method']=='native_lowrank_MEND')
  assert m['serialized_bytes']==n['serialized_bytes'] and m['heldout_update_nrmse']==n['heldout_update_nrmse']
  assert m['active_compute_proxy']==n['active_compute_proxy']
  with np.load(P/'source'/f'{s}_Mirror_code_editor.npz') as a,np.load(P/'source'/f'{s}_native_lowrank_MEND.npz') as b:
   assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
  for r in rows:
   if r['seed']==s:assert (P/'source'/f"{s}_{r['method']}.npz").stat().st_size==r['serialized_bytes']
