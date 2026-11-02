import csv,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_fresh_payloads():
 rows=list(csv.DictReader((ROOT/'RESULTS_CORE.csv').open()))
 assert len(rows)==45
 assert {r['phase'] for r in rows}=={'fresh'}
 for row in rows:
  p=ROOT.parents[2]/row['payload_path']
  assert p.exists() and p.stat().st_size==int(row['serialized_payload_bytes'])
  assert hashlib.sha256(p.read_bytes()).hexdigest()==row['payload_sha256']
  assert float(row['roundtrip_max_abs_error'])==0
