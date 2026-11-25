import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_replay_matches_saved_audit():
 r=json.loads((ROOT/'runs/replay.json').read_text())
 assert r['max_abs_delta']<1e-7
 assert set(r['replayed_audit_mse'])=={'shared','mirror','film','independent'}
def test_actual_payload_byte_hashes():
 m=json.loads((ROOT/'runs/payload_manifest.json').read_text())
 for name,x in m.items():
  p=ROOT/'runs'/name
  b=p.read_bytes()
  assert len(b)==x['bytes']
  assert hashlib.sha256(b).hexdigest()==x['sha256']
def test_preregistered_order_and_result_rows():
 p=json.loads((ROOT/'PROTOCOL.json').read_text())
 assert p['fresh']['locked_before_access'] is True
 res=json.loads((ROOT/'runs/results.json').read_text())
 assert {'shared','mirror','film','independent'}==set(res)
 assert all(x['updates']>0 for x in res.values())
