#!/usr/bin/env python3
"""Verify frozen MA-1057 audit by replaying exact saved float32 inference state."""
import hashlib,json
from pathlib import Path
import numpy as np
import fit_controls as fc
ROOT=Path(__file__).resolve().parents[1]; fc.RUNS=ROOT/'runs'
results=json.loads((fc.RUNS/'audit/results.json').read_text())
manifest=json.loads((fc.RUNS/'input_manifest.json').read_text())
selected=json.loads((fc.RUNS/'dev/tuning.json').read_text())
model_bytes=manifest['model']['weights_bytes']+manifest['model']['config_bytes']
for method,info in results['payloads'].items():
 folder=fc.RUNS/'audit/payloads'/method; doc=json.loads((folder/'method.json').read_text())
 coeff=(folder/'coefficients.f32le').read_bytes()
 assert hashlib.sha256(coeff).hexdigest()==doc['code_sha256']
 if method=='seasonal_naive': assert info['common_model_bytes']==0 and not (folder/'normalization.f32le').exists()
 else: assert info['common_model_bytes']==model_bytes
 if (folder/'normalization.f32le').exists():
  norm=(folder/'normalization.f32le').read_bytes(); assert hashlib.sha256(norm).hexdigest()==doc['normalization_sha256']
  vals=np.frombuffer(norm,dtype='<f4').reshape(2,2,8)
  for f in (0,1):
   manifest['data']['training_only_series_scales'][str(f)]['mean']=vals[f,0].astype(float).tolist()
   manifest['data']['training_only_series_scales'][str(f)]['std']=vals[f,1].astype(float).tolist()
 actual=info['common_model_bytes']+(folder/'method.json').stat().st_size+(folder/'coefficients.f32le').stat().st_size+((folder/'normalization.f32le').stat().st_size if (folder/'normalization.f32le').exists() else 0)
 if method=='seasonal_naive': actual=(folder/'method.json').stat().st_size
 assert actual==info['complete_payload_bytes'],(method,actual,info['complete_payload_bytes'])
for seed in (105702,105703):
 rows=fc.gather('audit',(fc.AUDIT_CELL,),selected['patch_size'],seed)
 assert len(rows)==24
 for method in fc.METHODS:
  folder=fc.RUNS/'audit/payloads'/method
  code=np.fromfile(folder/'coefficients.f32le',dtype='<f4') if method not in ('native','seasonal_naive') else np.zeros(0)
  score,_=fc.evaluate(rows,method,code,manifest)
  saved=next(r for r in results['results'] if r['seed']==seed and r['method']==method)
  for metric,value in score.items(): assert abs(value-saved['metrics'][metric])<1e-12,(seed,method,metric)
print(json.dumps({'status':'PASS','metric_rows_replayed':len(results['results']),'payloads_verified':len(results['payloads'])},indent=2))
