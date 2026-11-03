import hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
def test_freeze_hash_and_fresh_sealed():
 f=json.loads((P/'freeze.json').read_text());assert hashlib.sha256((P/'PROTOCOL.json').read_bytes()).hexdigest()==f['protocol_sha256'];assert not f['fresh_accessed']
def test_efficacy_locality_and_native_factor_alias():
 rows=json.loads((P/'source'/'development.json').read_text())
 for s in [63101,63102]:
  for m in ['independent_rank1','shared_lowrank_per_edit','mirror_factor_tables','native_additive_factor_tables']:
   r=[x for x in rows if x['seed']==s and x['method']==m]
   assert max(x['efficacy_nrmse'] for x in r)<1e-5
   assert max(x['locality_leakage'] for x in r)<1e-6
   p=P/'source'/f'{s}_{m}.npz';assert p.stat().st_size==r[0]['serialized_bytes']
  a=next(x for x in rows if x['seed']==s and x['method']=='mirror_factor_tables');b=next(x for x in rows if x['seed']==s and x['method']=='native_additive_factor_tables')
  assert a['serialized_bytes']==b['serialized_bytes'] and a['efficacy_nrmse']==b['efficacy_nrmse']
  with np.load(P/'source'/f'{s}_mirror_factor_tables.npz') as x,np.load(P/'source'/f'{s}_native_additive_factor_tables.npz') as y:
   assert x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files)
  ind=next(x for x in rows if x['seed']==s and x['method']=='independent_rank1');assert a['serialized_bytes']<ind['serialized_bytes']
