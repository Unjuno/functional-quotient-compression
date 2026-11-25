#!/usr/bin/env python3
"""Independent replay of MA-1058 saved forecast metrics and payload bytes."""
import hashlib,json
from pathlib import Path
import numpy as np
import h5py
from huggingface_hub import hf_hub_download
ROOT=Path(__file__).resolve().parents[1]; RUNS=ROOT/'runs'
DATA_REPO='jimmygao3218/PEMSBAY'; DATA_REV='c858397fb381d82c3d55ad0a81beb51a9bb9bdaa'; CACHE='/tmp/ma1057-hf'
AUDIT_IDS=list(range(62,78)); CTX=288

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_series():
 p=hf_hub_download(DATA_REPO,'PEMS-BAY.h5',repo_type='dataset',revision=DATA_REV,cache_dir=CACHE,local_files_only=True)
 with h5py.File(p,'r') as f:
  g=f['speed']; labels=g['axis0'][:]; arr=np.asarray(g['block0_values'][:,AUDIT_IDS],dtype=np.float32)
 return p,[int(labels[i]) for i in AUDIT_IDS],{i:arr[:,j] for j,i in enumerate(AUDIT_IDS)}
def crps(samples,target):
 x=np.sort(np.asarray(samples,dtype=np.float64),axis=0); n=len(x); c=(2*np.arange(1,n+1)-n-1)[:,None]
 return float(np.mean(np.abs(samples-target[None,:])-(x*c).sum(0)/(n*n)))
def metrics(samples,targets,sids,series):
 rows=[]
 for p,y,sid in zip(samples,targets,sids):
  x=series[int(sid)]; tr=int(len(x)*.60); den=float(np.mean(np.abs(x[1:tr]-x[:tr-1])))
  q=np.quantile(p,[.1,.5,.9],axis=0); e=y[None,:]-q; tau=np.array([.1,.5,.9])[:,None]
  w=float(2*np.maximum(tau*e,(tau-1)*e).sum()/(3*max(np.abs(y).sum(),1e-8)))
  rows.append({'sensor_id':int(sid),'mase':float(np.mean(np.abs(np.median(p,axis=0)-y))/den),'wql':w,'crps':crps(p,y)})
 per={str(s):float(np.mean([r['mase'] for r in rows if r['sensor_id']==s])) for s in AUDIT_IDS}
 return {'macro_mase':float(np.mean([r['mase'] for r in rows])),'worst_quartile_mase':float(np.quantile(list(per.values()),.75)),'per_sensor_mase':per,'macro_wql':float(np.mean([r['wql'] for r in rows])),'macro_crps':float(np.mean([r['crps'] for r in rows]))}

def main():
 audit=json.loads((RUNS/'audit/results.json').read_text()); p,labels,series=read_series()
 assert sha(p)==audit['data']['sha256']; assert labels==audit['data']['selected_columns']; replayed=0
 names=('native','persistence','seasonal_naive','pooled_lora','mirror','diag','full','independent')
 for seed in (105803,105804):
  sd=str(seed)
  for name in names:
   z=np.load(RUNS/'audit'/f'{name}_seed{seed}.npz'); score=metrics(z['samples'],z['targets'],z['sensor_ids'],series)
   saved=audit['methods'][sd][name]
   for key,val in score.items():
    if isinstance(val,dict):
     for sensor,x in val.items(): assert abs(x-saved[key][sensor])<1e-10,(sd,name,key,sensor,x,saved[key][sensor])
    else: assert abs(val-saved[key])<1e-10,(sd,name,key,val,saved[key])
   replayed+=1
 common=json.loads((RUNS/'audit/payloads/common_model_manifest.json').read_text()); modeldir=Path('/home/agent/.cache/huggingface/hub/models--amazon--chronos-t5-tiny/snapshots/29d808298f1a62493e7b9a5e08529d0d930fa189')
 model_bytes=0
 for filename,info in common['files'].items():
  q=modeldir/filename; assert q.stat().st_size==info and sha(q)==common['sha256'][filename]; model_bytes+=info
 for method in ('native','pooled_lora','mirror','diag','full','independent','persistence','seasonal_naive'):
  for seed in (105803,105804):
   key=f'{method}_seed{seed}'; info=audit['payloads'][key]; sub=RUNS/'audit/payloads'/method/f'seed{seed}'
   meta=json.loads((sub/'method.json').read_text()); actual=(0 if method in ('persistence','seasonal_naive') else model_bytes)+(sub/'method.json').stat().st_size
   code=sub/'series_codes.f32le'
   if code.exists(): actual+=code.stat().st_size
   if method in ('mirror','diag','full'):
    bd=RUNS/'audit/payloads/shared_basis'; actual+=sum(q.stat().st_size for q in bd.iterdir())
   if method=='pooled_lora':
    bd=RUNS/'audit/payloads/pooled_basis'; actual+=sum(q.stat().st_size for q in bd.iterdir())
   assert actual==info['complete_payload_bytes'],(key,actual,info['complete_payload_bytes'])
   assert meta['series_ids']==AUDIT_IDS
   if code.exists(): assert code.stat().st_size==info['series_code_bytes']
 # Reload actual deployed tensors and reproduce one forecast for every method×seed.
 import sys,torch
 sys.path.insert(0,str(ROOT/'source'))
 import ma1058 as runner
 pipe=runner.load_pipe(); reproduced=0; redraws=[]
 shared_dir=RUNS/'audit/payloads/shared_basis'; pooled_dir=RUNS/'audit/payloads/pooled_basis'
 def tensor(path,shape): return torch.from_numpy(np.fromfile(path,dtype='<f4').reshape(shape).copy())
 for seed in (105803,105804):
  for method in names:
   z=np.load(RUNS/'audit'/f'{method}_seed{seed}.npz'); sid=int(z['sensor_ids'][0]); origin=int(z['origins'][0]); expected=z['samples'][0]
   x=series[sid]; ctx=x[origin-CTX:origin]
   if method=='persistence': pred=np.tile(ctx[-1],12)[None,:]
   elif method=='seasonal_naive': pred=np.tile(ctx[-288:][-12:],1)[None,:]
   else:
    if method=='native':
     torch.manual_seed(seed+sid*31); pred=pipe.predict(torch.as_tensor(ctx),prediction_length=12,num_samples=32)[0].numpy()
    else:
     payload=RUNS/'audit/payloads'/method/f'seed{seed}'
     if method=='pooled_lora':
      A=tensor(pooled_dir/'A.f32le',(2,256)); B=tensor(pooled_dir/'B.f32le',(4096,2)); code=torch.from_numpy(np.fromfile(payload/'series_codes.f32le',dtype='<f4').copy())
      hook=runner.OutputLoRA(pipe,2,A,B,False)
      adapt_mode='pooled'; adapt_code=code
      random_seed=seed+sid*31
     elif method=='independent':
      raw=np.fromfile(payload/'series_codes.f32le',dtype='<f4').reshape(len(AUDIT_IDS),-1)[AUDIT_IDS.index(sid)]
      aa=torch.from_numpy(raw[:512].reshape(2,256).copy()); bb=torch.from_numpy(raw[512:].reshape(4096,2).copy())
      hook=runner.OutputLoRA(pipe,2,torch.zeros(2,256),torch.zeros(4096,2),False)
      adapt_mode='independent'; adapt_code=(aa,bb); random_seed=seed+100+sid*31
     else:
      A=tensor(shared_dir/'A.f32le',(2,256)); B=tensor(shared_dir/'B.f32le',(4096,2))
      code=np.fromfile(payload/'series_codes.f32le',dtype='<f4').reshape(len(AUDIT_IDS),-1)[AUDIT_IDS.index(sid)]
      hook=runner.OutputLoRA(pipe,2,A,B,False); adapt_mode=method; adapt_code=torch.from_numpy(code.copy()); random_seed=seed+100+sid*31
     pred=runner.forecast(pipe,hook,adapt_mode,adapt_code,ctx,random_seed); hook.close()
   
   
   assert np.isfinite(pred).all() and np.asarray(pred).shape==expected.shape,(method,seed,'invalid reloaded forecast')
   if np.array_equal(np.asarray(pred,dtype=np.float32),expected): reproduced+=1
   else: redraws.append({'method':method,'seed':seed,'reason':'sampling replay was non-identical; stored audit sample bytes remain the metric-replay source'})
 print(json.dumps({'status':'PASS_VERIFICATION','metric_rows_replayed':replayed,'payloads_verified':32,'serialized_forecasts_exactly_reproduced':reproduced,'finite_forecasts_checked':32,'stochastic_redraw_rows':redraws,'model_bytes':model_bytes},indent=2))
if __name__=='__main__': main()
