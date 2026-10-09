import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_fresh_bytes_and_order_audit():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs.jsonl').read_text().splitlines()]
 assert len(rows)==108 and {r['world'] for r in rows}=={45520,45521,45522}
 assert {r['method'] for r in rows}=={'tied','mirror','lowrank','independent'}
 for r in rows:
  b=Path(r['path']).read_bytes()
  assert len(b)==r['payload_bytes'] and hashlib.sha256(b).hexdigest()==r['hash']
  assert r['mean_commutator_fro']>0 and r['order_swapped_nrmse']>0
 n64=[r for r in rows if r['method']=='mirror' and r['n']==64]
 assert sum(r['mean_commutator_fro'] for r in n64)/len(n64)>0.1
 assert sum(r['order_swapped_nrmse'] for r in n64)/len(n64)>0.1

def test_registered_gate_misses():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs.jsonl').read_text().splitlines()]
 for w in (45520,45521,45522):
  m=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='mirror' and r['n']==20)
  i=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='independent' and r['n']==20)
  assert m['payload_bytes']/i['payload_bytes']>0.75
  assert m['nrmse_mean']>1.1*i['nrmse_mean']
