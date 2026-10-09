import hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def test_a1_payload_integrity_and_maps():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs.jsonl').read_text().splitlines()]
 assert len(rows)==108 and {r['world'] for r in rows}=={46820,46821,46822}
 for r in rows:
  path=Path(r['path']);b=path.read_bytes()
  assert len(b)==r['payload_bytes'] and hashlib.sha256(b).hexdigest()==r['hash']
  obj=torch.load(path,weights_only=True)
  if r['method']=='full':assert obj['skill_to_module'].tolist()==[0,1,2,3]
  else:assert obj['skill_to_module'].tolist()==[0,0,1,1]
def test_private_residual_boundary_and_byte_gate():
 rows=[json.loads(x) for x in (ROOT/'artifacts/fresh_runs.jsonl').read_text().splitlines()]
 for w in (46820,46821,46822):
  m=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='mirror' and r['n']==6)
  p=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='mirror_private' and r['n']==6)
  f=next(r for r in rows if r['world']==w and r['seed']==0 and r['method']=='full' and r['n']==6)
  assert m['heldout_pair_nrmse']>1.1*f['heldout_pair_nrmse']
  assert m['payload_bytes']>.75*f['payload_bytes']
  assert p['heldout_pair_nrmse']<.01 and p['payload_bytes']>f['payload_bytes']
