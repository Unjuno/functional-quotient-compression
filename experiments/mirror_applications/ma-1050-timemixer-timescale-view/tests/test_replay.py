import json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_results_and_serialized_payloads():
 r=json.loads((ROOT/"runs/screen_results.json").read_text())
 assert set(r)=={"shared","mirror","additive"}
 for k,v in r.items():
  assert v["updates"]==40
  assert v["payload_bytes"]==v["state_bytes"]+v["metadata_bytes"]
  assert (ROOT/"runs"/f"{k}.pt").stat().st_size==v["state_bytes"]
  assert v["audit_macro_mse"]>0 and v["dev_macro_mse"]>0
def test_hash_manifest():
 m=json.loads((ROOT/"runs/payload_manifest.json").read_text())
 for v in m.values():
  p=ROOT/"runs"/v["file"]
  assert p.stat().st_size==v["bytes"]
  assert hashlib.sha256(p.read_bytes()).hexdigest()==v["sha256"]
