import csv,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_development_payloads_and_preregistered_gate_failure():
 rows=list(csv.DictReader((ROOT/'RESULTS_CORE.csv').open()))
 assert len(rows)==60 and {r['phase'] for r in rows}=={'development'}
 assert {r['method'] for r in rows}=={'cast_explicit','coeff_fp32','coeff_fp16','mirror_fp16','mirror_private_fp16'}
 for r in rows:
  p=ROOT.parents[2]/r['payload_path']
  assert p.exists() and p.stat().st_size==int(r['serialized_payload_bytes'])
  assert hashlib.sha256(p.read_bytes()).hexdigest()==r['payload_sha256']
  assert float(r['max_replay_abs_error'])==0
