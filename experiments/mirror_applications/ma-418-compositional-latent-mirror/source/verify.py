from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import torch
from run import make_world,make_decoder,sample_inputs,target,infer,nrmse,O,S,D
ROOT=Path(__file__).resolve().parents[1]
def verify():
 rows=list(csv.DictReader((ROOT/'runs/RESULTS_CORE.csv').open()));assert len(rows)==8;checks=[]
 for seed in (41801,41802):
  pairs,ztrue,train_ix,held_ix=make_world(seed);ids=pairs[held_ix];dec=make_decoder(seed);xq=sample_inputs(seed+5,len(held_ix),1024);yq=target(dec,xq,ztrue[held_ix])
  outputs={}
  for name,kind in [('factor','factor'),('native','native'),('deep','deep'),('oracle','deep')]:
   raw=(ROOT/'runs'/f'dev{seed}_{name}.npz').read_bytes();pred=infer(raw,ids,xq,kind);outputs[name]=pred;row=next(r for r in rows if int(r['world_or_seed'])==seed and r['method']==name)
   assert abs(float(row['primary_value'])-nrmse(pred,yq))<1e-6
   assert int(row['serialized_bytes'])==len(raw)
   assert row['status_note']==hashlib.sha256(raw).hexdigest()
   checks.append({'seed':seed,'method':name,'payload_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'metric_replay':'PASS'})
  assert torch.max(torch.abs(outputs['factor']-outputs['native']))<=1e-6
 screen=json.loads((ROOT/'runs/screen.json').read_text());assert not screen['fresh_accessed']
 return {'experiment_id':'MA-418','status':'VERIFIED_DEVELOPMENT_SCREEN','protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),'metric_replay':'PASS','payload_replay':'PASS','fresh_accessed':False,'development_gate_passed':screen['development_gate_passed'],'checks':checks}
if __name__=='__main__':
 r=verify();(ROOT/'VERIFICATION.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
