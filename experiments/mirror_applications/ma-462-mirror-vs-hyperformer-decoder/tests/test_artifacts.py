import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_a1_payload_hash_and_byte_replay():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs_A1.jsonl').read_text().splitlines()]
 assert len(rows)==162 and {r['world'] for r in rows}=={46220,46221,46222}
 assert {r['method'] for r in rows}=={'hyper','mirror','affine','rank2','fourier','private'}
 for r in rows:
  b=Path(r['path']).read_bytes()
  assert len(b)==r['payload_bytes'] and hashlib.sha256(b).hexdigest()==r['hash']
def test_initial_payload_hash_and_byte_replay():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs_initial.jsonl').read_text().splitlines()]
 assert len(rows)==135 and {r['world'] for r in rows}=={46210,46211,46212}
 for r in rows:
  b=Path(r['path']).read_bytes()
  assert len(b)==r['payload_bytes'] and hashlib.sha256(b).hexdigest()==r['hash']
def test_fourier_control_falsifies_mirror_specific_gate():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs_A1.jsonl').read_text().splitlines()]
 for w in (46220,46221,46222):
  m=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='mirror' and r['n']==32)
  f=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='fourier' and r['n']==32)
  h=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='hyper' and r['n']==32)
  assert m['nrmse_mean']<h['nrmse_mean']
  assert m['payload_bytes']<=.70*h['payload_bytes']
  assert m['payload_bytes']>.70*f['payload_bytes']
  assert m['nrmse_mean']>1.1*f['nrmse_mean']
