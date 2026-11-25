#!/usr/bin/env python3
"""MA-1058 Chronos few-shot cold-start View development/audit runner."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from chronos import ChronosPipeline

ROOT=Path(__file__).resolve().parents[1]; RUNS=ROOT/'runs'
DATA_REPO='jimmygao3218/PEMSBAY'; DATA_REV='c858397fb381d82c3d55ad0a81beb51a9bb9bdaa'
MODEL_ID='amazon/chronos-t5-tiny'; MODEL_REV='29d808298f1a62493e7b9a5e08529d0d930fa189'
CACHE='/tmp/ma1057-hf'; SNAP='/home/agent/.cache/huggingface/hub/models--amazon--chronos-t5-tiny/snapshots/'+MODEL_REV
FIT_IDS=list(range(22,38)); DEV_IDS=list(range(38,46)); AUDIT_IDS=list(range(62,78))
CTX=288; H=12; PRED_LEN=64; NSAMP=32; torch.set_num_threads(1)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def get_series(ids):
 p=hf_hub_download(DATA_REPO,'PEMS-BAY.h5',repo_type='dataset',revision=DATA_REV,cache_dir=CACHE,local_files_only=True)
 # The pinned HDF is pandas fixed-format. Read only requested value columns via HDF5 hyperslab;
 # never load a full frame containing held-out series.
 with h5py.File(p,'r') as f:
  group=f['speed']; labels=group['axis0'][:]
  values=group['block0_values']; arr=np.asarray(values[:,ids],dtype=np.float32)
 selected_labels=[int(labels[i]) for i in ids]
 if not np.isfinite(arr).all(): raise ValueError('predeclared series contain missing/nonfinite values')
 return p,selected_labels,{i:arr[:,j] for j,i in enumerate(ids)}

def origins(n,split):
 tr=int(n*.60); audit_start=int(n*.80)
 if split=='support': return np.unique(np.rint(np.linspace(CTX,tr-PRED_LEN,6)).astype(int)).tolist()
 if split=='query': return np.unique(np.rint(np.linspace(audit_start,n-PRED_LEN,3)).astype(int)).tolist()
 raise ValueError(split)
def examples(x):
 return {'support':[ (x[o-CTX:o],x[o:o+PRED_LEN],o) for o in origins(len(x),'support') ],
         'query':[ (x[o-CTX:o],x[o:o+H],o) for o in origins(len(x),'query') ]}

class OutputLoRA:
 def __init__(self,pipe,rank=2,A=None,B=None,train_basis=False):
  self.pipe=pipe; self.layer=pipe.model.model.lm_head; self.rank=rank
  if A is None:
   torch.manual_seed(105808); self.A=torch.nn.Parameter(torch.randn(rank,256)*.002)
   self.B=torch.nn.Parameter(torch.zeros(4096,rank))
  else:
   self.A=torch.nn.Parameter(A.detach().clone(),requires_grad=train_basis)
   self.B=torch.nn.Parameter(B.detach().clone(),requires_grad=train_basis)
  self.mode='mirror'; self.code=None
  self.handle=self.layer.register_forward_hook(self._hook)
 def set(self,mode,code): self.mode=mode; self.code=code
 def _hook(self,module,args,out):
  h=args[0]; z=F.linear(h,self.A); r=self.rank
  if self.mode=='mirror':
   t=self.code.reshape(-1)[0]; c=torch.cos(t); s=torch.sin(t)
   if r==1: zr=z
   else: zr=torch.stack((c*z[...,0]-s*z[...,1],s*z[...,0]+c*z[...,1]),dim=-1)
  elif self.mode=='diag':
   g=self.code.reshape(-1); scale=torch.ones(r,device=z.device,dtype=z.dtype); scale[0]=1+g[0]; zr=z*scale
  elif self.mode=='full':
   zr=z*self.code.reshape(-1)[:r]
  elif self.mode=='pooled':
   t=self.code.reshape(-1)[0]; c=torch.cos(t); s=torch.sin(t)
   zr=z if r==1 else torch.stack((c*z[...,0]-s*z[...,1],s*z[...,0]+c*z[...,1]),dim=-1)
  elif self.mode=='independent':
   a,b=self.code; zr=F.linear(h,a)
   return out+F.linear(zr,b)
  else: return out
  return out+F.linear(zr,self.B)
 def forward_loss(self,ctx,target,mode,code):
  self.set(mode,code)
  x=torch.as_tensor(ctx,dtype=torch.float32).reshape(1,-1)
  y=torch.as_tensor(target,dtype=torch.float32).reshape(1,-1)
  ids,mask,state=self.pipe.tokenizer.context_input_transform(x)
  lab,_=self.pipe.tokenizer.label_input_transform(y,state)
  out=self.pipe.model.model(input_ids=ids,attention_mask=mask,labels=lab)
  # First 12 logits predict the registered 12-step horizon; the remaining 52 tokens
  # satisfy Chronos' native training target length and are causally masked from earlier leads.
  return F.cross_entropy(out.logits[:,:H].reshape(-1,4096),lab[:,:H].reshape(-1))
 def close(self): self.handle.remove()

def load_pipe():
 pipe=ChronosPipeline.from_pretrained(SNAP,device_map='cpu',torch_dtype=torch.float32)
 pipe.model.eval()
 for p in pipe.model.parameters(): p.requires_grad_(False)
 return pipe

def fit_shared(pipe,series,updates=30,family='mirror'):
 lora=OutputLoRA(pipe,2,train_basis=True)
 if family=='mirror':
  theta=torch.nn.ParameterDict({str(i):torch.nn.Parameter(torch.tensor((i%5-2)*.16)) for i in FIT_IDS})
  params=[lora.A,lora.B,*theta.values()]
 else:
  theta=torch.nn.Parameter(torch.zeros(())); params=[lora.A,lora.B,theta]
 opt=torch.optim.AdamW(params,lr=.01,weight_decay=.01)
 rng=np.random.default_rng(105801); rows=[(sid,e) for sid in FIT_IDS for e in examples(series[sid])['support']]
 start=time.perf_counter()
 for step in range(updates):
  sid,(ctx,y,_)=rows[int(rng.integers(len(rows)))]; opt.zero_grad()
  if family=='mirror': mode='mirror'; code=theta[str(sid)]
  else: mode='pooled'; code=theta
  loss=lora.forward_loss(ctx,y,mode,code); loss.backward(); opt.step()
 lora.close(); elapsed=time.perf_counter()-start
 if family=='mirror': saved={k:v.detach().cpu() for k,v in theta.items()}
 else: saved={'shared':theta.detach().cpu()}
 return {'A':lora.A.detach().cpu(),'B':lora.B.detach().cpu(),'fit_theta':saved,'updates':updates,'wall_seconds':elapsed}

def adapt_code(pipe,A,B,support,mode='mirror',updates=30,seed=0):
 lora=OutputLoRA(pipe,2,A,B,False)
 if mode=='independent':
  a=torch.nn.Parameter(A.detach().clone()); b=torch.nn.Parameter(B.detach().clone()); code=(a,b); params=[a,b]
 elif mode=='full':
  code=torch.nn.Parameter(torch.ones(2)); params=[code]
 else:
  code=torch.nn.Parameter(torch.zeros(1)); params=[code]
 opt=torch.optim.AdamW(params,lr=.01,weight_decay=.01)
 rng=np.random.default_rng(seed); start=time.perf_counter()
 for step in range(updates):
  ctx,y,_=support[int(rng.integers(len(support)))]; opt.zero_grad(); loss=lora.forward_loss(ctx,y,mode,code)
  loss.backward(); opt.step()
 sec=time.perf_counter()-start
 if mode=='independent': saved=(code[0].detach().cpu(),code[1].detach().cpu())
 else: saved=code.detach().cpu()
 lora.close()
 return saved,sec

@torch.no_grad()
def forecast(pipe,adapter,mode,code,ctx,seed):
 adapter.set(mode,code); torch.manual_seed(int(seed))
 out=pipe.predict(torch.as_tensor(ctx,dtype=torch.float32),prediction_length=H,num_samples=NSAMP)
 return out[0].numpy().astype(np.float32)

def mase(pred,target,den): return float(np.mean(np.abs(np.median(pred,axis=0)-target))/den)
def score_adapter(pipe,adapter,A,B,series,ids,updates,mode,seed_base):
 adapter.set('inactive',None)
 rows=[]; fit_s=0.; infer=[]
 for sid in ids:
  x=series[sid]; ex=examples(x); code,elapsed=adapt_code(pipe,A,B,ex['support'],mode,updates,seed_base+sid); fit_s+=elapsed
  den=float(np.mean(np.abs(x[1:int(len(x)*.6)]-x[:int(len(x)*.6)-1])))
  for j,(ctx,y,o) in enumerate(ex['query']):
   t=time.perf_counter(); pred=forecast(pipe,adapter,mode,code,ctx,seed_base+sid*31+j); infer.append(time.perf_counter()-t)
   row=scored_row(pred,y,sid,o,den)
   code_payload=(torch.cat([code[0].reshape(-1),code[1].reshape(-1)]).detach().cpu().numpy() if mode=='independent' else code.detach().reshape(-1).cpu().numpy())
   row['adapter_code']=code_payload; rows.append(row)
 return rows,fit_s,infer

def run_dev(args):
 ids=FIT_IDS+DEV_IDS; path,labels,series=get_series(ids); pipe=load_pipe(); shared=fit_shared(pipe,series,args.shared_updates,'mirror'); pooled_shared=fit_shared(pipe,series,args.shared_updates,'pooled')
 (RUNS/'dev').mkdir(parents=True,exist_ok=True); torch.save({'A':shared['A'],'B':shared['B'],'fit_theta':shared['fit_theta']},RUNS/'dev/shared_basis.pt'); torch.save({'A':pooled_shared['A'],'B':pooled_shared['B'],'fit_theta':pooled_shared['fit_theta']},RUNS/'dev/pooled_basis.pt')
 outputs={'fit_updates':shared['updates'],'mirror_fit_wall_seconds':shared['wall_seconds'],'pooled_fit_wall_seconds':pooled_shared['wall_seconds'],'models':{},'data':{'sha256':sha(path),'rows':len(next(iter(series.values()))),'selected_ids':ids,'selected_columns':labels,'model_revision':MODEL_REV,'model_sha256':sha(Path(SNAP)/'model.safetensors'),'config_sha256':sha(Path(SNAP)/'config.json'),'chronos_version':'1.5.3','transformers_version':'4.46.3','torch_version':torch.__version__}}
 adapter=OutputLoRA(pipe,2,shared['A'],shared['B'],False)
 for method in ('mirror','diag','full','independent'):
  for updates in (10,30,60):
   rows,fit_s,infer=score_adapter(pipe,adapter,shared['A'],shared['B'],series,DEV_IDS,updates,method,105810+updates)
   met=float(np.mean([r['mase'] for r in rows])); key=f'{method}_u{updates}'
   outputs['models'][key]={'macro_mase':met,'per_sensor_mase':{str(s):float(np.mean([r['mase'] for r in rows if r['sensor_id']==s])) for s in DEV_IDS},'fit_seconds':fit_s,'inference_p50_s':float(np.median(infer)),'inference_p95_s':float(np.quantile(infer,.95))}
 # Pooled LoRA: independently fit one shared rank-2 adapter and global angle on fit sensors.
 adapter.set('inactive',None)
 pooled=[]; pooled_adapter=OutputLoRA(pipe,2,pooled_shared['A'],pooled_shared['B'],False); pooled_code=pooled_shared['fit_theta']['shared']
 for sid in DEV_IDS:
  x=series[sid]; ex=examples(x)
  for j,(ctx,y,o) in enumerate(ex['query']):
   pred=forecast(pipe,pooled_adapter,'pooled',pooled_code,ctx,105820+sid*31+j)
   den=float(np.mean(np.abs(x[1:int(len(x)*.6)]-x[:int(len(x)*.6)-1])))
   pooled.append(mase(pred,y,den))
 pooled_adapter.close(); adapter.close();
 # Native, persistence, and seasonal-naive references use the same future windows.
 native=[]; naive=[]; seasonal_scores=[]
 for sid in DEV_IDS:
  x=series[sid]; ex=examples(x); den=float(np.mean(np.abs(x[1:int(len(x)*.6)]-x[:int(len(x)*.6)-1])))
  for j,(ctx,y,o) in enumerate(ex['query']):
   torch.manual_seed(105820+sid*31+j); pred=pipe.predict(torch.as_tensor(ctx),prediction_length=H,num_samples=NSAMP)[0].numpy()
   native.append(mase(pred,y,den)); naive.append(float(np.mean(np.abs(np.tile(ctx[-1],H)-y))/den))
   pred_seas=np.tile(ctx[-288:][-H:],int(np.ceil(H/H)))[:H]; seasonal_scores.append(float(np.mean(np.abs(pred_seas-y))/den))
 outputs['native_macro_mase']=float(np.mean(native)); outputs['pooled_lora_macro_mase']=float(np.mean(pooled)); outputs['persistence_macro_mase']=float(np.mean(naive)); outputs['seasonal_naive_macro_mase']=float(np.mean(seasonal_scores))
 selected={m:min((10,30,60),key=lambda u:outputs['models'][f'{m}_u{u}']['macro_mase']) for m in ('mirror','diag','full','independent')}
 outputs['selected_updates_by_method']=selected
 (RUNS/'dev/development.json').write_text(json.dumps(outputs,indent=2)+'\n')
 # Freeze all comparator-specific update counts before loading any audit-series values.
 (RUNS/'dev/selection.json').write_text(json.dumps({'updates_by_method':selected,'shared_updates':args.shared_updates,'seed':105801,'audit_methods':['native','pooled','mirror','diag','full','independent','persistence','seasonal_naive']},indent=2)+'\n')
 print(json.dumps({k:v for k,v in outputs.items() if k!='models'},indent=2)); print(json.dumps(outputs['models'],indent=2))

def main():
 p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['dev','audit'],required=True); p.add_argument('--shared-updates',type=int,default=30); a=p.parse_args(); run_dev(a) if a.phase=='dev' else run_audit()
def empirical_crps(samples,target):
 x=np.sort(np.asarray(samples,dtype=np.float64),axis=0); n=x.shape[0]; coef=(2*np.arange(1,n+1)-n-1)[:,None]
 pair=(x*coef).sum(0)/(n*n)
 return float(np.mean(np.abs(samples-target[None,:])-pair))

def scored_row(pred,target,sensor_id,origin,den):
 p=np.asarray(pred,dtype=np.float32); y=np.asarray(target,dtype=np.float32)
 q=np.quantile(p,[.1,.5,.9],axis=0); e=y[None,:]-q; tau=np.array([.1,.5,.9])[:,None]
 wql=float(2*np.maximum(tau*e,(tau-1)*e).sum()/(3*max(np.abs(y).sum(),1e-8)))
 return {'sensor_id':sensor_id,'origin':origin,'mase':mase(p,y,den),'wql':wql,'crps':empirical_crps(p,y),'pred_samples':p,'target':y}

def metrics_rows(rows):
 vals=np.asarray([r['mase'] for r in rows],dtype=float)
 return {'macro_mase':float(vals.mean()),'worst_quartile_mase':float(np.quantile([np.mean([r['mase'] for r in rows if r['sensor_id']==s]) for s in AUDIT_IDS],.75)),
         'per_sensor_mase':{str(s):float(np.mean([r['mase'] for r in rows if r['sensor_id']==s])) for s in AUDIT_IDS},
         'macro_wql':float(np.mean([r['wql'] for r in rows])),'macro_crps':float(np.mean([r['crps'] for r in rows]))}

def run_audit():
 selection=json.loads((RUNS/'dev/selection.json').read_text()); ids=AUDIT_IDS
 path,labels,series=get_series(ids); pipe=load_pipe()
 shared=torch.load(RUNS/'dev/shared_basis.pt',map_location='cpu',weights_only=False)
 pooled_shared=torch.load(RUNS/'dev/pooled_basis.pt',map_location='cpu',weights_only=False)
 mirror_adapter=OutputLoRA(pipe,2,shared['A'],shared['B'],False); mirror_adapter.set('inactive',None)
 pooled_adapter=OutputLoRA(pipe,2,pooled_shared['A'],pooled_shared['B'],False); pooled_adapter.set('inactive',None)
 out={'experiment_id':'MA-1058','audit_ids':ids,'seeds':[105803,105804],'data':{'sha256':sha(path),'rows':len(next(iter(series.values()))),'selected_columns':labels},'methods':{},'payloads':{}}
 for seed in (105803,105804):
  seed_key=str(seed); out['methods'][seed_key]={}
  # Frozen native and zero-parameter baselines.
  native=[]; persistence=[]; seasonal=[]
  for sid in ids:
   x=series[sid]; ex=examples(x); den=float(np.mean(np.abs(x[1:int(len(x)*.6)]-x[:int(len(x)*.6)-1])))
   for j,(ctx,y,o) in enumerate(ex['query']):
    torch.manual_seed(seed+sid*31+j); pred=pipe.predict(torch.as_tensor(ctx),prediction_length=H,num_samples=NSAMP)[0].numpy().astype(np.float32)
    native.append(scored_row(pred,y,sid,o,den))
    lp=np.tile(ctx[-1],H); persistence.append(scored_row(lp[None,:],y,sid,o,den))
    sp=np.tile(ctx[-288:][-H:],int(np.ceil(H/H)))[:H]; seasonal.append(scored_row(sp[None,:],y,sid,o,den))
  out['methods'][seed_key]['native']=metrics_rows(native); out['methods'][seed_key]['persistence']=metrics_rows(persistence); out['methods'][seed_key]['seasonal_naive']=metrics_rows(seasonal)
  (RUNS/'audit').mkdir(parents=True,exist_ok=True)
  for name,rows in [('native',native),('persistence',persistence),('seasonal_naive',seasonal)]:
   np.savez_compressed(RUNS/'audit'/f'{name}_seed{seed}.npz',sensor_ids=np.asarray([r['sensor_id'] for r in rows]),origins=np.asarray([r['origin'] for r in rows]),targets=np.stack([r['target'] for r in rows]),samples=np.stack([r['pred_samples'] for r in rows]))
  # Shared rank-2 bases with few-shot codes, 2D direct coordinate, and independent rank-2 experts.
  for method in ('mirror','diag','full','independent'):
   ups=selection['updates_by_method'][method]
   adapter=OutputLoRA(pipe,2,shared['A'],shared['B'],False)
   rows,fit_s,infer=score_adapter(pipe,adapter,shared['A'],shared['B'],series,ids,ups,method,seed+100)
   out['methods'][seed_key][method]=metrics_rows(rows)
   out['methods'][seed_key][method].update({'support_fit_seconds':fit_s,'query_p50_s':float(np.median(infer)),'query_p95_s':float(np.quantile(infer,.95)),'updates_per_series':ups})
   np.savez_compressed(RUNS/'audit'/f'{method}_seed{seed}.npz',sensor_ids=np.asarray([r['sensor_id'] for r in rows]),origins=np.asarray([r['origin'] for r in rows]),targets=np.stack([r['target'] for r in rows]),samples=np.stack([r['pred_samples'] for r in rows]))
   code_map={sid:next(r['adapter_code'] for r in rows if r['sensor_id']==sid) for sid in ids}
   code_array=np.stack([code_map[sid] for sid in ids]).astype('<f4')
   (RUNS/'audit'/f'{method}_seed{seed}.codes.f32le').write_bytes(code_array.tobytes())
   adapter.close()
  pooled_rows=[]; code=pooled_shared['fit_theta']['shared']
  for sid in ids:
   x=series[sid]; ex=examples(x); den=float(np.mean(np.abs(x[1:int(len(x)*.6)]-x[:int(len(x)*.6)-1])))
   for j,(ctx,y,o) in enumerate(ex['query']):
    pred=forecast(pipe,pooled_adapter,'pooled',code,ctx,seed+sid*31+j)
    pooled_rows.append(scored_row(pred,y,sid,o,den))
  out['methods'][seed_key]['pooled_lora']=metrics_rows(pooled_rows)
  np.asarray([float(code)],dtype='<f4').tofile(RUNS/'audit'/f'pooled_lora_seed{seed}.codes.f32le')
  np.savez_compressed(RUNS/'audit'/f'pooled_lora_seed{seed}.npz',sensor_ids=np.asarray([r['sensor_id'] for r in pooled_rows]),origins=np.asarray([r['origin'] for r in pooled_rows]),targets=np.stack([r['target'] for r in pooled_rows]),samples=np.stack([r['pred_samples'] for r in pooled_rows]))
 mirror_adapter.close(); pooled_adapter.close()
 # Materialize deployment payloads. Each stores only bytes needed for actual inference.
 folder=RUNS/'audit/payloads'; folder.mkdir(parents=True,exist_ok=True)
 common_files={name:(Path(SNAP)/name).stat().st_size for name in ('model.safetensors','config.json','generation_config.json')}
 (folder/'common_model_manifest.json').write_text(json.dumps({'model':MODEL_ID,'revision':MODEL_REV,'files':common_files,'sha256':{k:sha(Path(SNAP)/k) for k in common_files}},sort_keys=True,separators=(',',':'))+'\n')
 model_total=sum(common_files.values())
 # Recreate the actual shared factors/series codes from deterministic dev/adaptation states.

 basis_dir=folder/'shared_basis'; basis_dir.mkdir(exist_ok=True)
 for name,tensor in [('A',shared['A']),('B',shared['B'])]: np.asarray(tensor,dtype='<f4').tofile(basis_dir/f'{name}.f32le')
 (basis_dir/'basis.json').write_text(json.dumps({'rank':2,'input_dim':256,'output_dim':4096,'dtype':'float32-little-endian'},sort_keys=True,separators=(',',':'))+'\n')
 common_basis=sum(p.stat().st_size for p in basis_dir.iterdir())
 pooled_dir=folder/'pooled_basis'; pooled_dir.mkdir(exist_ok=True)
 for name,tensor in [('A',pooled_shared['A']),('B',pooled_shared['B'])]: np.asarray(tensor,dtype='<f4').tofile(pooled_dir/f'{name}.f32le')
 np.asarray(pooled_shared['fit_theta']['shared'],dtype='<f4').tofile(pooled_dir/'global_angle.f32le')
 (pooled_dir/'basis.json').write_text(json.dumps({'rank':2,'input_dim':256,'output_dim':4096,'dtype':'float32-little-endian'},sort_keys=True,separators=(',',':'))+'\n')
 pooled_basis_bytes=sum(p.stat().st_size for p in pooled_dir.iterdir())
 for method in ('native','pooled_lora','mirror','diag','full','independent','persistence','seasonal_naive'):
  for seed in (105803,105804):
   sub=folder/method/f'seed{seed}'; sub.mkdir(parents=True,exist_ok=True)
   state=RUNS/'audit'/f'{method}_seed{seed}.codes.f32le'
   code_bytes=state.stat().st_size if state.exists() else 0
   meta={'method':method,'seed':seed,'series_ids':ids,'frequency_minutes':5,'context_steps':CTX,'horizon_steps':H,'num_samples':NSAMP if method not in ('persistence','seasonal_naive') else 1,'model_revision':MODEL_REV if method not in ('persistence','seasonal_naive') else None,'code_dtype':'float32-little-endian','code_shape':([1] if method=='pooled_lora' else [len(ids),code_bytes//4//len(ids)] if code_bytes else [0])}
   if method=='mirror': meta['adapter']='shared rank-2 A/B with per-series rotation angle'
   if method=='diag': meta['adapter']='shared rank-2 A/B with per-series first-axis gain'
   if method=='full': meta['adapter']='shared rank-2 A/B with per-series 2-coefficient diagonal gain'
   if method=='independent': meta['adapter']='per-series independent rank-2 A/B factors'
   if method=='pooled_lora': meta['adapter']='pooled rank-2 A/B and global rotation angle'
   (sub/'method.json').write_text(json.dumps(meta,sort_keys=True,separators=(',',':'))+'\n')
   if state.exists(): (sub/'series_codes.f32le').write_bytes(state.read_bytes())
   basis_bytes=(pooled_basis_bytes if method=='pooled_lora' else common_basis if method in ('mirror','diag','full') else 0)
   if method=='independent': basis_bytes=0
   complete=(0 if method in ('persistence','seasonal_naive') else model_total)+basis_bytes+code_bytes+(sub/'method.json').stat().st_size
   out['payloads'][f'{method}_seed{seed}']={'common_model_bytes':0 if method in ('persistence','seasonal_naive') else model_total,'adapter_basis_bytes':basis_bytes,'series_code_bytes':code_bytes,'metadata_bytes':(sub/'method.json').stat().st_size,'complete_payload_bytes':complete}
 (RUNS/'audit/results.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:{m:v['macro_mase'] for m,v in q.items()} for k,q in out['methods'].items()},indent=2))


if __name__=='__main__': main()
