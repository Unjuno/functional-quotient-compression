from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import torch
from run import METHODS,world,replay,nrmse
ROOT=Path(__file__).resolve().parents[1]
def verify():
 rows=list(csv.DictReader((ROOT/'runs/RESULTS_CORE.csv').open()));assert len(rows)==10
 checks=[]
 for seed in (40301,40302):
  x,h,p,y,xt,ht,pt,yt,w1,w2=world(seed)
  for method in METHODS:
   r=next(r for r in rows if int(r['world_or_seed'])==seed and r['method']==method)
   raw=(ROOT/'runs'/f'dev{seed}_{method}.npz').read_bytes();pred=replay(raw,xt,pt,method);score=nrmse(pred,yt)
   assert abs(score-float(r['primary_value']))<1e-6
   assert len(raw)==int(r['serialized_bytes'])
   assert hashlib.sha256(raw).hexdigest()==r['status_note']
   checks.append({'seed':seed,'method':method,'metric_replay':'PASS','payload_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
 screen=json.loads((ROOT/'runs/screen.json').read_text());assert not screen['fresh_accessed']
 return {'experiment_id':'MA-403','status':'VERIFIED_DEVELOPMENT_SCREEN','protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),'metric_replay':'PASS','payload_replay':'PASS','fresh_accessed':False,'checks':checks,'development_gate_passed':screen['development_gate_passed']}
if __name__=='__main__':
 r=verify();(ROOT/'VERIFICATION.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
