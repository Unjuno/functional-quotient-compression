from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import torch
from run import make_world,pack,simulate,simulate_values,nrmse,diversity,ROLES,NSEQ,LENGTH
ROOT=Path(__file__).resolve().parents[1]
def verify():
 rows=list(csv.DictReader((ROOT/'runs/RESULTS_CORE.csv').open()));assert len(rows)==8;checks=[];div_by_seed={};torch.set_num_threads(1)
 for seed in (43401,43402):
  base,codes,x=make_world(seed);ref=simulate_values(base,codes,x,'mirror');preds={}
  for method in ('mirror','native','independent','shared'):
   raw=(ROOT/'runs'/f'dev{seed}_{method}.npz').read_bytes();assert raw==pack(base,codes,method);pred=simulate(raw,x,method);preds[method]=pred;row=next(r for r in rows if int(r['world_or_seed'])==seed and r['method']==method)
   assert abs(float(row['primary_value'])-nrmse(pred,ref))<1e-6
   assert int(row['serialized_bytes'])==len(raw)
   assert row['status_note'].startswith(hashlib.sha256(raw).hexdigest())
   checks.append({'seed':seed,'method':method,'payload_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'output_replay':'PASS'})
  assert (ROOT/'runs'/f'dev{seed}_mirror.npz').read_bytes()==(ROOT/'runs'/f'dev{seed}_native.npz').read_bytes()
  assert torch.max(torch.abs(preds['mirror']-preds['native']))==0
  div_by_seed[seed]=diversity(preds['mirror'])
 screen=json.loads((ROOT/'runs/screen.json').read_text());assert not screen['fresh_accessed']
 for s in screen['seed_results']:
  assert abs(s['mode_diversity_rms']-div_by_seed[s['seed']])<1e-6
 return {'experiment_id':'MA-434','status':'VERIFIED_DEVELOPMENT_SCREEN','protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),'metric_replay':'PASS','payload_replay':'PASS','native_alias_replay':'PASS','fresh_accessed':False,'development_gate_passed':screen['development_gate_passed'],'checks':checks}
if __name__=='__main__':
 result=verify();(ROOT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
