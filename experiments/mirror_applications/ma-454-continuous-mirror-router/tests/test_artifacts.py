import hashlib, json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def test_fresh_payload_integrity_and_mirror_hyper_equivalence():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs.jsonl').read_text().splitlines()]
 assert len(rows)==135
 assert {r['world'] for r in rows}=={45420,45421,45422}
 for r in rows:
  p=Path(r['path'])
  assert p.is_file()
  b=p.read_bytes()
  assert len(b)==r['payload_bytes']
  assert hashlib.sha256(b).hexdigest()==r['hash']
 for w in (45420,45421,45422):
  for s in (0,1,2):
   for n in (1,20,64):
    m=next(r for r in rows if r['world']==w and r['seed']==s and r['method']=='mirror' and r['n']==n)
    h=next(r for r in rows if r['world']==w and r['seed']==s and r['method']=='hyper' and r['n']==n)
    assert m['payload_bytes']==h['payload_bytes'] and m['hash']==h['hash']
    assert m['nrmse_mean']==h['nrmse_mean']
def test_protocol_is_frozen_before_fresh():
 p=json.loads((ROOT/'PROTOCOL.json').read_text())
 assert p['fresh']['locked_before_access'] is True
 assert p['status']=='COMPLETED_FAIL'
