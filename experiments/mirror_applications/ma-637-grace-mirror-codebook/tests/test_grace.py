import hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
def test_freeze_and_fresh_sealed():
 f=json.loads((P/'freeze.json').read_text());assert hashlib.sha256((P/'PROTOCOL.json').read_bytes()).hexdigest()==f['protocol_sha256'];assert not f['fresh_accessed']
def test_pca_alias_retention_and_scope():
 rows=json.loads((P/'source'/'development.json').read_text())
 for s in [63701,63702]:
  for n in [16,32,64,128]:
   m=next(r for r in rows if r['seed']==s and r['snapshot_edits']==n and r['method']=='mirror_basis_codes');p=next(r for r in rows if r['seed']==s and r['snapshot_edits']==n and r['method']=='native_pca_codes');e=next(r for r in rows if r['seed']==s and r['snapshot_edits']==n and r['method']=='grace_explicit')
   assert m['serialized_bytes']==p['serialized_bytes'] and m['retention_nrmse']==p['retention_nrmse']
   assert m['retention_nrmse']<1e-5 and m['positive_scope_hit_rate']==1 and m['negative_false_positive_rate']==0
   assert m['serialized_bytes']==(P/'source'/f'{s}_n{n}_mirror_basis_codes.npz').stat().st_size
   with np.load(P/'source'/f'{s}_n{n}_mirror_basis_codes.npz') as a,np.load(P/'source'/f'{s}_n{n}_native_pca_codes.npz') as b:
    assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
