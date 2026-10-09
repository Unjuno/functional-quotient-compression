from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import torch
from run import make_world,sample_x,target,infer,nrmse
ROOT=Path(__file__).resolve().parents[1]
def verify():
 rows=list(csv.DictReader((ROOT/'runs/RESULTS_CORE.csv').open()));assert len(rows)==10;checks=[];torch.set_num_threads(1)
 for seed in (41901,41902):
  codes,amp,freq,phase,tr,he=make_world(seed);xq=sample_x(seed+3,len(he),512);yq=target(xq,codes,amp,freq,phase,he);outputs={}
  for name in ('mirror','native','concat','private','oracle'):
   raw=(ROOT/'runs'/f'dev{seed}_{name}.npz').read_bytes();kind=name if name in ('mirror','native','concat') else 'table';pred=infer(raw,xq,kind);outputs[name]=pred;row=next(r for r in rows if int(r['world_or_seed'])==seed and r['method']==name)
   assert abs(float(row['primary_value'])-nrmse(pred,yq))<1e-6
   assert int(row['serialized_bytes'])==len(raw)
   assert row['status_note']==hashlib.sha256(raw).hexdigest()
   checks.append({'seed':seed,'method':name,'payload_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'metric_replay':'PASS'})
  assert torch.max(torch.abs(outputs['mirror']-outputs['native']))<=1e-7
 screen=json.loads((ROOT/'runs/screen.json').read_text());assert not screen['fresh_accessed']
 return {'experiment_id':'MA-419','status':'VERIFIED_DEVELOPMENT_SCREEN','protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),'metric_replay':'PASS','payload_replay':'PASS','native_alias_replay':'PASS','fresh_accessed':False,'development_gate_passed':screen['development_gate_passed'],'checks':checks}
if __name__=='__main__':
 result=verify();(ROOT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
