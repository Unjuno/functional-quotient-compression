import hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
def test_frozen_protocol_fresh_sealed():
 f=json.loads((P/'freeze.json').read_text());assert hashlib.sha256((P/'PROTOCOL.json').read_bytes()).hexdigest()==f['protocol_sha256'];assert not f['fresh_accessed']
def test_native_pca_alias_and_scope():
 rows=json.loads((P/'source'/'development.json').read_text())
 for s in [63601,63602]:
  m=next(r for r in rows if r['seed']==s and r['method']=='mirror_value_code');n=next(r for r in rows if r['seed']==s and r['method']=='native_pca_value_code');e=next(r for r in rows if r['seed']==s and r['method']=='serac_explicit')
  assert m['serialized_bytes']==n['serialized_bytes'] and m['heldout_value_nrmse']==n['heldout_value_nrmse']
  assert m['positive_scope_hit_rate']==n['positive_scope_hit_rate']==e['positive_scope_hit_rate']==1
  assert m['negative_false_positive_rate']==n['negative_false_positive_rate']==e['negative_false_positive_rate']==0
  assert m['serialized_bytes']>.5*e['serialized_bytes']
  with np.load(P/'source'/f'{s}_mirror_value_code.npz') as a,np.load(P/'source'/f'{s}_native_pca_value_code.npz') as b:
   assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
