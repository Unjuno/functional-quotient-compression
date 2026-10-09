import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_fresh_payload_bytes_and_hashes():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs.jsonl').read_text().splitlines()]
 assert len(rows)==108 and {r['world'] for r in rows}=={46910,46911,46912}
 for r in rows:
  b=Path(r['path']).read_bytes()
  assert len(b)==r['payload_bytes'] and hashlib.sha256(b).hexdigest()==r['hash']
def test_mirror_matches_factor_function_but_misses_byte_gate():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs.jsonl').read_text().splitlines()]
 for w in (46910,46911,46912):
  for s in (0,1,2):
   m=next(r for r in rows if r['world']==w and r['seed']==s and r['method']=='mirror' and r['n']==64)
   b=next(r for r in rows if r['world']==w and r['seed']==s and r['method']=='mend' and r['n']==64)
   assert m['edit_error']<1e-5 and m['paraphrase_error']<1e-5
   assert abs(m['locality_drift']-b['locality_drift'])<1e-8
   assert m['payload_bytes']>.80*b['payload_bytes']
