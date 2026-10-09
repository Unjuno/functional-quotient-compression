import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_payload_replay_and_fresh_coverage():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs.jsonl').read_text().splitlines()]
 assert len(rows)==108 and {r['world'] for r in rows}=={45710,45711,45712}
 assert {r['method'] for r in rows}=={'path','mirror','private'}
 for r in rows:
  b=Path(r['path']).read_bytes()
  assert len(b)==r['payload_bytes'] and hashlib.sha256(b).hexdigest()==r['hash']
  assert r['module_births_cumulative']>=1 and r['query_wall_seconds_mean']>0

def test_mirror_birth_reduction_misses_quality_and_byte_gates():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs.jsonl').read_text().splitlines()]
 for w in (45710,45711,45712):
  m=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='mirror' and r['n']==32)
  p=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='path' and r['n']==32)
  q=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='private' and r['n']==32)
  assert m['module_births_cumulative']<=.5*p['module_births_cumulative']
  assert m['payload_bytes']>.75*p['payload_bytes']
  assert m['nrmse_mean']>1.05*q['nrmse_mean']
