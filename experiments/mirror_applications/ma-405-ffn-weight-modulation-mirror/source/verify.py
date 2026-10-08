from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
from run import METHODS,world,replay,metric
ROOT=Path(__file__).resolve().parents[1]
def verify():
 rows=list(csv.DictReader((ROOT/'runs/RESULTS_CORE.csv').open()));assert len(rows)==50
 checks=[]
 for seed in (40501,40502):
  x,h,y,xt,ytest,w1,w2,w,tm=world(seed)
  for method in METHODS:
   raw=(ROOT/'runs'/f'dev{seed}_{method}.npz').read_bytes();digest=hashlib.sha256(raw).hexdigest()
   selected=[r for r in rows if int(r['world_or_seed'])==seed and r['method']==method]
   scores=metric(replay(raw,xt,method),ytest)
   for r in selected:
    ai=round(float(r['condition'].split('=')[1])*4)
    assert abs(float(r['primary_value'])-scores[ai])<1e-6
    assert int(r['serialized_bytes'])==len(raw) and r['status_note']==digest
   checks.append({'seed':seed,'method':method,'payload_bytes':len(raw),'sha256':digest,'metric_rows_replayed':5})
 screen=json.loads((ROOT/'runs/screen.json').read_text());assert screen['fresh_accessed'] is False
 return {'experiment_id':'MA-405','status':'VERIFIED_DEVELOPMENT_SCREEN','protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),'metric_replay':'PASS','payload_replay':'PASS','fresh_accessed':False,'development_gate_passed':screen['development_gate_passed'],'checks':checks}
if __name__=='__main__':
 r=verify();(ROOT/'VERIFICATION.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
