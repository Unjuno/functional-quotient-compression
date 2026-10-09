import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_payloads_are_exact_and_fresh_complete():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs.jsonl').read_text().splitlines()]
 assert len(rows)==135 and {r['world'] for r in rows}=={46320,46321,46322}
 for r in rows:
  b=Path(r['path']).read_bytes()
  assert len(b)==r['payload_bytes'] and hashlib.sha256(b).hexdigest()==r['hash']
def test_mirror_is_identical_to_generic_cp_and_misses_gate():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs.jsonl').read_text().splitlines()]
 for w in (46320,46321,46322):
  for s in (0,1,2):
   for n in (1,32,96):
    m=next(r for r in rows if r['world']==w and r['seed']==s and r['method']=='mirror' and r['n']==n)
    c=next(r for r in rows if r['world']==w and r['seed']==s and r['method']=='cp' and r['n']==n)
    assert m['payload_bytes']==c['payload_bytes'] and m['hash']==c['hash'] and m['heldout_nrmse']==c['heldout_nrmse']
 failures=0
 for w in (46320,46321,46322):
  m=[r for r in rows if r['world']==w and r['method']=='mirror' and r['n']==96]
  h=[r for r in rows if r['world']==w and r['method']=='hyper' and r['n']==96]
  if sum(r['heldout_nrmse'] for r in m)/3 > 1.1*sum(r['heldout_nrmse'] for r in h)/3: failures+=1
  assert sum(r['payload_bytes'] for r in m)/3 > .60*sum(r['payload_bytes'] for r in h)/3
 assert failures>=1
