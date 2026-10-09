import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_fresh_package_integrity():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs.jsonl').read_text().splitlines()]
 assert len(rows)==108 and {r['world'] for r in rows}=={46410,46411,46412}
 for r in rows:
  b=Path(r['path']).read_bytes()
  assert len(b)==r['payload_bytes'] and hashlib.sha256(b).hexdigest()==r['hash']
def test_routed_and_merged_gates():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs.jsonl').read_text().splitlines()]
 for w in (46410,46411,46412):
  mr=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='mirror_routed' and r['n']==4)
  ar=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='adamix_routed' and r['n']==4)
  mm=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='mirror_merged' and r['n']==4)
  am=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='adamix_merged' and r['n']==4)
  assert mr['nrmse_mean']>1.1*ar['nrmse_mean'] and mr['payload_bytes']>.60*ar['payload_bytes']
  assert mm['nrmse_mean']<=1.1*am['nrmse_mean']
