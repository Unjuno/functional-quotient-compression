#!/usr/bin/env python3
"""Replay MA-1059 metrics and inference from the actual serialized audit payloads."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'runs'
spec=importlib.util.spec_from_file_location('ma1059',ROOT/'source/ma1059.py')
ma=importlib.util.module_from_spec(spec); spec.loader.exec_module(ma)

def read_f32(path,shape):
    a=np.fromfile(path,dtype='<f4')
    if a.size!=int(np.prod(shape)): raise AssertionError((path,a.size,shape))
    return a.reshape(shape)

def load_basis(which):
    folder=RUNS/'audit/payloads'/which
    return torch.from_numpy(read_f32(folder/'A.f32le',(2,256)).copy()),torch.from_numpy(read_f32(folder/'B.f32le',(4096,2)).copy())

def replay_metrics(results):
    worst=0.0; rows=0; _,all_series=ma.load_series(ma.AUDIT_IDS)
    for seed in results['seeds']:
        for method,summary in results['methods'][str(seed)].items():
            z=np.load(RUNS/'audit'/f'{method}_seed{seed}.npz')
            ids=z['series_ids'].tolist(); samples=z['samples']; targets=z['targets']
            per=summary['per_series']
            assert ids==[r['series_id'] for r in per]
            m=[]; c=[]
            for i,r in enumerate(per):
                sid=ids[i]
                # MASE denominator uses only the first 60% of the corresponding series.
                den=ma.mase_den(all_series[sid])
                mm=ma.mase(samples[i],targets[i],den); cc=ma.crps(samples[i],targets[i])
                worst=max(worst,abs(mm-r['mase']),abs(cc-r['crps'])); m.append(mm); c.append(cc)
            assert abs(float(np.mean(m))-summary['macro_mase'])<1e-10
            assert abs(float(np.mean(c))-summary['macro_crps'])<1e-8
            rows+=len(ids)
    return rows,worst

def replay_payload(results):
    _,series=ma.load_series(ma.AUDIT_IDS); pipe=ma.load_pipe(); maxerr=0.0; exact=True
    sharedA,sharedB=load_basis('shared_basis'); pooledA,pooledB=load_basis('pooled_basis')
    shared_hook=ma.OutputLoRA(pipe,sharedA,sharedB,False); pooled_hook=ma.OutputLoRA(pipe,pooledA,pooledB,False)
    method_meta={m:json.loads((RUNS/'audit/payloads'/m/'method.json').read_text()) for m in results['methods']['105903']}
    codes={}
    for method in ('mirror','diag','full'):
        codes[method]=read_f32(RUNS/'audit/payloads'/method/'series_codes.f32le',(len(ma.AUDIT_IDS), {'mirror':1,'diag':1,'full':2}[method]))
    independent=read_f32(RUNS/'audit/payloads/independent/series_codes.f32le',(len(ma.AUDIT_IDS),8704))
    pooled_angle=float(read_f32(RUNS/'audit/payloads/pooled_lora/global_angle.f32le',(1,))[0])
    residual_meta=method_meta['mirror_trend']['residual_series_ids']
    residual_vals=read_f32(RUNS/'audit/payloads/mirror_trend/trend_residuals.f32le',(len(residual_meta),))
    residuals=dict(zip(residual_meta,residual_vals.tolist()))
    for seed in results['seeds']:
        for method in results['methods'][str(seed)]:
            saved=np.load(RUNS/'audit'/f'{method}_seed{seed}.npz'); expected=saved['samples']
            got=[]
            for i,sid in enumerate(ma.AUDIT_IDS):
                x=series[sid]; ctx,y,_=ma.windows(x)['query'][0]
                shared_hook.set('inactive',None); pooled_hook.set('inactive',None)
                if method=='native': pred=ma.predict(pipe,shared_hook,'inactive',None,ctx,seed+int(sid[1:])*31)
                elif method=='pooled_lora':
                    code=torch.tensor(pooled_angle,dtype=torch.float32); pred=ma.predict(pipe,pooled_hook,'pooled',code,ctx,seed+int(sid[1:])*31)
                elif method in ('mirror','diag','full'):
                    code=torch.from_numpy(codes[method][i].copy()); pred=ma.predict(pipe,shared_hook,method,code,ctx,seed+int(sid[1:])*31)
                elif method=='mirror_trend':
                    code=torch.from_numpy(codes['mirror'][i].copy()); pred=ma.predict(pipe,shared_hook,'mirror',code,ctx,seed+int(sid[1:])*31)
                    if sid in residuals: pred=ma.apply_resid(pred,residuals[sid])
                elif method=='independent':
                    vec=independent[i]; a=torch.from_numpy(vec[:512].reshape(2,256).copy()); b=torch.from_numpy(vec[512:].reshape(4096,2).copy())
                    hook=ma.OutputLoRA(pipe,a,b,False); pred=ma.predict(pipe,hook,'independent',(a,b),ctx,seed+int(sid[1:])*31); hook.close()
                elif method=='persistence': pred=np.tile(ctx[-1],ma.H)[None,:].astype(np.float32)
                elif method=='seasonal_naive': pred=np.tile(ctx[-12:],1)[None,:].astype(np.float32)
                else: raise AssertionError(method)
                got.append(pred)
            got=np.stack(got)
            diff=float(np.max(np.abs(got-expected))); maxerr=max(maxerr,diff); exact &= np.array_equal(got,expected)
    shared_hook.close(); pooled_hook.close()
    return exact,maxerr

def main():
    results=json.loads((RUNS/'audit/results.json').read_text())
    rows,metric_diff=replay_metrics(results)
    exact,payload_diff=replay_payload(results)
    out={'status':'PASS' if exact and metric_diff<1e-8 and payload_diff==0 else 'FAIL','saved_metric_rows_replayed':rows,'max_metric_difference':metric_diff,'serialized_payload_forecast_exact':exact,'max_serialized_forecast_difference':payload_diff,'actual_payload_bytes':{m:v['complete_payload_bytes'] for m,v in results['payloads'].items()}}
    (RUNS/'audit/verification_replay.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    if out['status']!='PASS': raise SystemExit(1)
if __name__=='__main__': main()
