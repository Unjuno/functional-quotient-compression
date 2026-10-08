import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'

def test_fresh_payload_hashes_and_partition():
 rows=[json.loads(x) for x in open(ART/'fresh_runs.jsonl')]
 assert len(rows)==144
 assert {r['world'] for r in rows}=={44610,44611,44612}
 for r in rows:
  data=(ROOT.parents[2]/r['path']).read_bytes()
  assert hashlib.sha256(data).hexdigest()==r['hash']
  assert len(data)==r['payload_bytes']

def test_preregistered_step4_gate_fails():
 rows=[json.loads(x) for x in open(ART/'fresh_runs.jsonl') if json.loads(x)['step']==4]
 for w in (44610,44611,44612):
  a=[r['nrmse_mean'] for r in rows if r['world']==w and r['method']=='adam']
  l=[r['nrmse_mean'] for r in rows if r['world']==w and r['method']=='learned']
  assert len(a)==len(l)==3
  assert sum(l)/3 > .9*(sum(a)/3) if w==44612 else sum(l)/3 <= .9*(sum(a)/3)
