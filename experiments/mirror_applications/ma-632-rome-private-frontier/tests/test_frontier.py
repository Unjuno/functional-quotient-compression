import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]
def test_freeze_and_fresh_sealed():
 import hashlib
 f=json.loads((P/'freeze.json').read_text());assert hashlib.sha256((P/'PROTOCOL.json').read_bytes()).hexdigest()==f['protocol_sha256'];assert not f['fresh_accessed']
def test_quality_frontier_and_native_lowrank_dominance():
 rows=json.loads((P/'source'/'development.json').read_text())
 for s in [63201,63202]:
  for p in [0,2,4,8]:
   for q in [p,16]:
    m=next(r for r in rows if r['seed']==s and r['true_private_rank']==p and r['stored_private_rank']==q and r['method']=='mirror_shared_private')
    n=next(r for r in rows if r['seed']==s and r['true_private_rank']==p and r['stored_private_rank']==q and r['method']=='native_joint_lowrank')
    assert m['efficacy_nrmse']<1e-5 and n['efficacy_nrmse']<1e-5
    assert m['locality_leakage']==n['locality_leakage']==0
    assert n['serialized_bytes']<m['serialized_bytes']
  for r in rows:
   if r['seed']==s:
    path=P/'source'/f"{s}_p{r['true_private_rank']}_q{r['stored_private_rank']}_{r['method']}.npz"
    assert path.stat().st_size==r['serialized_bytes']
