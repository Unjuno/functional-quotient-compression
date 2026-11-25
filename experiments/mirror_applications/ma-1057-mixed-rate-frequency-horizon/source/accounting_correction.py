#!/usr/bin/env python3
"""Correct deployed payload accounting and replay the actual f32 serialized state.

No audit observations are used to select or change any experiment setting. This
post-audit auditability amendment only materializes state required to execute the
frozen calibration, removes the unused checkpoint from the seasonal-naive
baseline, and replays stored predictions through those actual bytes.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import fit_controls as fc

ROOT=Path(__file__).resolve().parents[1]; fc.RUNS=ROOT/'runs'
manifest=json.loads((fc.RUNS/'input_manifest.json').read_text())
result=json.loads((fc.RUNS/'audit/results.json').read_text())
selected=json.loads((fc.RUNS/'dev/tuning.json').read_text())
model_bytes=manifest['model']['weights_bytes']+manifest['model']['config_bytes']
# Rewrite immutable decoder metadata and materialize calibration normalization.
for method,info in result['payloads'].items():
 folder=fc.RUNS/'audit/payloads'/method
 coeff=folder/'coefficients.f32le'; code=np.frombuffer(coeff.read_bytes(),dtype='<f4')
 doc={'method':method,'code_shape':list(code.shape),'code_dtype':'float32-little-endian','code_sha256':hashlib.sha256(coeff.read_bytes()).hexdigest(),
      'sensor_ids':list(range(14,22)),'frequency_minutes':[5,15],'horizon_hours':[1,24]}
 if method=='seasonal_naive':
  doc.update({'predictor':'repeat input at 96-step physical-day lag','seasonal_period_steps':96,'source_frequency_minutes':15,'model_payload_bytes':0})
  base=0; state_extra=0
 else:
  doc.update({'model_revision':manifest['model']['revision'],'patch_size':manifest['model']['patch_size'],'samples':manifest['model']['samples']})
  base=model_bytes; state_extra=0
  if method!='native':
   stats=manifest['data']['training_only_series_scales']; vals=[]
   for f in (0,1): vals.extend(stats[str(f)]['mean']); vals.extend(stats[str(f)]['std'])
   norm=np.asarray(vals,dtype='<f4'); nb=folder/'normalization.f32le'; nb.write_bytes(norm.tobytes()); state_extra=nb.stat().st_size
   doc.update({'normalization_file':'normalization.f32le','normalization_dtype':'float32-little-endian','normalization_shape':[2,2,8],
               'normalization_order':'frequency-major [means[8],stds[8]]','normalization_sha256':hashlib.sha256(nb.read_bytes()).hexdigest()})
  else: (folder/'normalization.f32le').unlink(missing_ok=True)
 methodfile=folder/'method.json';methodfile.write_text(json.dumps(doc,sort_keys=True,separators=(',',':'))+'\n')
 state=(0 if method=='seasonal_naive' else coeff.stat().st_size)+methodfile.stat().st_size+state_extra
 info.update({'common_model_bytes':base,'method_state_bytes':state,'complete_payload_bytes':base+state,
              'coefficients_bytes':coeff.stat().st_size if method!='seasonal_naive' else 0,
              'metadata_bytes':methodfile.stat().st_size,'normalization_bytes':state_extra})
# Replay using values decoded from the exact saved little-endian float32 payload.
for method,info in result['payloads'].items():
 folder=fc.RUNS/'audit/payloads'/method
 if info['normalization_bytes']:
  q=np.fromfile(folder/'normalization.f32le',dtype='<f4').reshape(2,2,8)
  for f in (0,1):
   manifest['data']['training_only_series_scales'][str(f)]['mean']=q[f,0].astype(float).tolist()
   manifest['data']['training_only_series_scales'][str(f)]['std']=q[f,1].astype(float).tolist()
for seed in (105702,105703):
 rows=fc.gather('audit',(fc.AUDIT_CELL,),selected['patch_size'],seed)
 for method in fc.METHODS:
  folder=fc.RUNS/'audit/payloads'/method
  code=np.fromfile(folder/'coefficients.f32le',dtype='<f4') if method not in ('native','seasonal_naive') else np.zeros(0)
  score,per=fc.evaluate(rows,method,code,manifest)
  saved=next(r for r in result['results'] if r['seed']==seed and r['method']==method)
  saved['metrics']=score; saved['per_sensor']=per
result['accounting_correction']='Post-freeze serialized-state correction: materialized inference normalization, reloaded f32 coefficients/stats, and removed unused Moirai checkpoint from seasonal-naive baseline; no fit, hyperparameter, or audit selection changed.'
(fc.RUNS/'audit/results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'payloads':result['payloads'],'replayed_metrics':[{k:r['metrics'] for k in ('seed','method','metrics')} for r in result['results']]},indent=2))
