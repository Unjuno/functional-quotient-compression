#!/usr/bin/env python3
"""Replay saved audit metrics and verify actual serialized payload accounting."""
import hashlib,json
from pathlib import Path
import numpy as np
import fit_controls as fc

root=Path(__file__).resolve().parents[1]
fc.RUNS=root/'runs'
result=json.loads((fc.RUNS/'audit/results.json').read_text())
selected=json.loads((fc.RUNS/'dev/tuning.json').read_text())
meta=json.loads((fc.RUNS/'input_manifest.json').read_text())
assert result['audit_cell']==[1,0] if 'audit_cell' in result else True
checks=[]
for seed in (105702,105703):
 rows=fc.gather('audit',(fc.AUDIT_CELL,),selected['patch_size'],seed)
 assert len(rows)==24
 for method in fc.METHODS:
  code=np.asarray(selected['codes'].get(method,[]))
  score,_=fc.evaluate(rows,method,code,meta)
  saved=next(r for r in result['results'] if r['seed']==seed and r['method']==method)
  for metric,value in score.items():
   assert abs(value-saved['metrics'][metric])<1e-12,(seed,method,metric,value,saved['metrics'][metric])
  checks.append({'seed':seed,'method':method,'metrics':score})
for method,info in result['payloads'].items():
 folder=fc.RUNS/'audit/payloads'/method
 actual=info['common_model_bytes']+(folder/'coefficients.f32le').stat().st_size+(folder/'method.json').stat().st_size
 assert actual==info['complete_payload_bytes']
 assert hashlib.sha256((folder/'coefficients.f32le').read_bytes()).hexdigest()==json.loads((folder/'method.json').read_text())['code_sha256']
print(json.dumps({'status':'PASS','replayed_rows':len(checks),'payload_methods':len(result['payloads'])},indent=2))
