from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import torch
from run import make_world,pack,simulate,rel_rmse,diversity,RK4
ROOT=Path(__file__).resolve().parents[1]
def verify():
 rows=list(csv.DictReader((ROOT/'runs/RESULTS_CORE.csv').open()));assert len(rows)==8;checks=[];torch.set_num_threads(1)
 for seed in (42401,42402):
  field,theta,x0=make_world(seed);reference=simulate(pack(field,theta,'mirror'),x0,'mirror',steps=RK4,integrator='rk4');out={}
  for name in ('mirror','native','independent','shared'):
   raw=(ROOT/'runs'/f'dev{seed}_{name}.npz').read_bytes();assert raw==pack(field,theta,name);pred=simulate(raw,x0,name);out[name]=pred;row=next(r for r in rows if int(r['world_or_seed'])==seed and r['method']==name)
   assert abs(float(row['primary_value'])-rel_rmse(pred,reference))<1e-8
   assert int(row['serialized_bytes'])==len(raw)
   assert row['status_note'].startswith(hashlib.sha256(raw).hexdigest())
   checks.append({'seed':seed,'method':name,'payload_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'trajectory_replay':'PASS'})
  assert (ROOT/'runs'/f'dev{seed}_mirror.npz').read_bytes()==(ROOT/'runs'/f'dev{seed}_native.npz').read_bytes()
  assert torch.max(torch.abs(out['mirror']-out['native']))<=1e-7
 screen=json.loads((ROOT/'runs/screen.json').read_text());assert not screen['fresh_accessed']
 for summary in screen['seed_results']:
  seed=summary['seed'];mirror=(ROOT/'runs'/f'dev{seed}_mirror.npz').read_bytes();field,theta,x0=make_world(seed);out=simulate(mirror,x0,'mirror');assert abs(summary['mode_diversity_rms']-diversity(out))<1e-8
 return {'experiment_id':'MA-424','status':'VERIFIED_DEVELOPMENT_SCREEN','protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),'metric_replay':'PASS','payload_replay':'PASS','native_alias_replay':'PASS','fresh_accessed':False,'development_gate_passed':screen['development_gate_passed'],'initial_invalid_timing_runs_preserved':True,'checks':checks}
if __name__=='__main__':
 result=verify();(ROOT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
