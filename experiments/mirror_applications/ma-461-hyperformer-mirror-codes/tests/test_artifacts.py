import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_fresh_payload_integrity():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs.jsonl').read_text().splitlines()]
 assert len(rows)==108 and {r['world'] for r in rows}=={46110,46111,46112}
 assert {r['method'] for r in rows}=={'hyper','mirror','rank2','independent'}
 for r in rows:
  b=Path(r['path']).read_bytes()
  assert len(b)==r['payload_bytes'] and hashlib.sha256(b).hexdigest()==r['hash']

def test_registered_quality_and_byte_gates_fail():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs.jsonl').read_text().splitlines()]
 for w in (46110,46111,46112):
  m=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='mirror' and r['n']==20)
  h=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='hyper' and r['n']==20)
  r=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='rank2' and r['n']==20)
  assert m['nrmse_mean']>1.1*h['nrmse_mean']
  assert m['payload_bytes']>.70*h['payload_bytes']
  assert r['nrmse_mean']<m['nrmse_mean'] and r['payload_bytes']==m['payload_bytes']
