import hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
def test_freeze_hash_fresh_sealed():
 f=json.loads((P/'freeze.json').read_text());assert hashlib.sha256((P/'PROTOCOL.json').read_bytes()).hexdigest()==f['protocol_sha256'];assert not f['fresh_accessed']
def test_projection_is_separate_and_native_pca_alias():
 rows=json.loads((P/'source'/'development.json').read_text())
 for s in [63901,63902]:
  raw=next(r for r in rows if r['seed']==s and r['world']=='aligned' and r['stored_rank']==2 and r['method']=='unprojected_full')
  loki=next(r for r in rows if r['seed']==s and r['world']=='aligned' and r['stored_rank']==2 and r['method']=='loki_projected_full')
  assert raw['protected_key_leakage']>.3 and loki['protected_key_leakage']<1e-6
  for r in rows:
   if r['seed']==s:
    p=P/'source'/f"{s}_{r['world']}_q{r['stored_rank']}_{r['method']}.npz";assert p.stat().st_size==r['serialized_bytes']
  for world in ['aligned','private']:
   for q in [0,1,2,4,8,16]:
    m=next(r for r in rows if r['seed']==s and r['world']==world and r['stored_rank']==q and r['method']=='mirror_projected_code')
    n=next(r for r in rows if r['seed']==s and r['world']==world and r['stored_rank']==q and r['method']=='native_projected_pca')
    assert m['serialized_bytes']==n['serialized_bytes'] and m['heldout_efficacy_nrmse']==n['heldout_efficacy_nrmse'] and m['protected_key_leakage']==n['protected_key_leakage']
    with np.load(P/'source'/f'{s}_{world}_q{q}_mirror_projected_code.npz') as a,np.load(P/'source'/f'{s}_{world}_q{q}_native_projected_pca.npz') as b:
     assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
def test_aligned_recovery_and_private_boundary():
 rows=json.loads((P/'source'/'development.json').read_text())
 for s in [63901,63902]:
  a=next(r for r in rows if r['seed']==s and r['world']=='aligned' and r['stored_rank']==2 and r['method']=='mirror_projected_code')
  p=next(r for r in rows if r['seed']==s and r['world']=='private' and r['stored_rank']==16 and r['method']=='mirror_projected_code')
  full=next(r for r in rows if r['seed']==s and r['world']=='private' and r['stored_rank']==16 and r['method']=='loki_projected_full')
  assert a['heldout_efficacy_nrmse']<1e-5 and p['heldout_efficacy_nrmse']>.2 and full['heldout_efficacy_nrmse']<1e-5
