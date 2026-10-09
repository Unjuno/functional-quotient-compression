import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_fresh_payload_integrity():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs.jsonl').read_text().splitlines()]
 assert len(rows)==108 and {r['world'] for r in rows}=={46610,46611,46612}
 for r in rows:
  b=Path(r['path']).read_bytes()
  assert len(b)==r['payload_bytes'] and hashlib.sha256(b).hexdigest()==r['hash']
def test_cp_equivalence_and_component_ablations():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs.jsonl').read_text().splitlines()]
 for w in (46610,46611,46612):
  for s in (0,1,2):
   for n in (1,12,36):
    m=next(r for r in rows if r['world']==w and r['seed']==s and r['method']=='mirror' and r['n']==n)
    c=next(r for r in rows if r['world']==w and r['seed']==s and r['method']=='cp' and r['n']==n)
    assert m['payload_bytes']==c['payload_bytes'] and m['hash']==c['hash'] and m['heldout_nrmse']==c['heldout_nrmse']
  m=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='mirror' and r['n']==36)
  assert all(m[f'ablated_component_nrmse_{i}']>1.1*m['heldout_nrmse'] for i in (1,2,3))
