"""MA-886 producer-once cross-model KV fan-out screen."""
from __future__ import annotations
import os
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('HF_DATASETS_OFFLINE','1')
import hashlib,json,resource,struct,time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache

SRC='EleutherAI/pythia-70m-deduped'; TARGETS=('EleutherAI/pythia-160m-deduped','HuggingFaceTB/SmolLM2-135M'); MODELS=(SRC,)+TARGETS
D=64; ALPHA=.01; SEEDS=(10883,10889,10897); NLAYERS={'EleutherAI/pythia-160m-deduped':12,'HuggingFaceTB/SmolLM2-135M':30}; NHEADS={'EleutherAI/pythia-160m-deduped':12,'HuggingFaceTB/SmolLM2-135M':3}

def exp_root(): return Path(__file__).resolve().parents[1]
def snapshot(mid):
 p=Path.home()/'.cache/huggingface/hub'/('models--'+mid.replace('/','--'))/'snapshots'; return sorted(p.iterdir())[0]
def load_assets():
 torch.set_num_threads(min(4,os.cpu_count() or 1)); models={};toks={};paths={}
 for mid in MODELS:
  paths[mid]=snapshot(mid); toks[mid]=AutoTokenizer.from_pretrained(paths[mid],local_files_only=True,use_fast=True)
  models[mid]=AutoModelForCausalLM.from_pretrained(paths[mid],local_files_only=True,attn_implementation='eager').eval()
  for p in models[mid].parameters(): p.requires_grad_(False)
 for t in TARGETS:
  cfg=models[t].config; nh=getattr(cfg,'num_key_value_heads',None) or cfg.num_attention_heads
  if (cfg.num_hidden_layers,nh)!=(NLAYERS[t],NHEADS[t]): raise ValueError(f'unexpected target dimensions: {t}')
 return models,toks,paths

def model_provenance(paths):
 out={}
 for mid,p in paths.items():
  files=sorted(p.glob('*.safetensors')); h=hashlib.sha256(); n=0
  for f in files: h.update(f.name.encode()); h.update(f.read_bytes()); n+=f.stat().st_size
  out[mid]={'snapshot':p.name,'weight_files':len(files),'weights_sha256':h.hexdigest(),'checkpoint_bytes':n}
 return out

def prompt_groups(toks):
 manifest=json.loads((exp_root()/'prompt_manifest.json').read_text()); ds=load_dataset('Salesforce/wikitext','wikitext-103-raw-v1',split='train'); groups={}
 for bucket,items in manifest['buckets'].items():
  rows=[]
  for rec in items:
   text=ds[int(rec['row'])]['text'].strip()
   if hashlib.sha256(text.encode()).hexdigest()!=rec['text_sha256']: raise AssertionError('prompt hash mismatch')
   enc={mid:toks[mid](text,add_special_tokens=False,return_offsets_mapping=True) for mid in MODELS}
   ref=enc[SRC]['offset_mapping']
   if any(enc[mid]['offset_mapping']!=ref for mid in TARGETS): raise AssertionError('token offsets differ')
   ns={mid:len(enc[mid]['input_ids']) for mid in MODELS}
   if len(set(ns.values()))!=1 or ns[SRC]!=rec['tokens']: raise AssertionError('token count changed')
   rows.append({'row':int(rec['row']),'title_sha256':rec['title_sha256'],'text':text,'ids':{m:enc[m]['input_ids'] for m in MODELS},'tokens':ns[SRC]})
  groups[str(bucket)]={'map_fit':rows[:24],'check':rows[24:32],'test':rows[32:48]}
 return groups

def cache_numpy(model,ids):
 t=time.perf_counter()
 with torch.inference_mode(): out=model(input_ids=torch.tensor(ids,dtype=torch.long)[None,:],use_cache=True)
 model_sec=time.perf_counter()-t;te=time.perf_counter(); past=out.past_key_values
 if hasattr(past,'to_legacy_cache'): past=past.to_legacy_cache()
 arr=[(k[0].detach().cpu().numpy().astype(np.float32),v[0].detach().cpu().numpy().astype(np.float32)) for k,v in past]
 return arr,out,model_sec,time.perf_counter()-te

def inv_rope_np(x,cfg):
 z=np.asarray(x,np.float32).copy(); typ=getattr(cfg,'model_type','')
 if typ not in ('gpt_neox','llama','mistral','qwen2'): return z
 dim=int((cfg.hidden_size//cfg.num_attention_heads)*float(cfg.rotary_pct)) if typ=='gpt_neox' else z.shape[-1]
 theta=float(getattr(cfg,'rope_theta',10000.)); inv=1/(theta**(np.arange(0,dim,2,dtype=np.float64)/dim)); ang=np.outer(np.arange(z.shape[-2],dtype=np.float64),inv)
 co=np.concatenate([np.cos(ang),np.cos(ang)],-1); si=-np.concatenate([np.sin(ang),np.sin(ang)],-1)
 if typ=='gpt_neox': co=np.concatenate([co,np.ones((len(co),z.shape[-1]-dim))],-1); si=np.concatenate([si,np.zeros((len(si),z.shape[-1]-dim))],-1)
 shape=(1,)*(z.ndim-2)+(len(co),co.shape[-1]); co=co.reshape(shape);si=si.reshape(shape)
 def rot(a): h=a.shape[-1]//2; return np.concatenate([-a[...,h:],a[...,:h]],-1)
 if typ=='gpt_neox':
  a,b=z[...,:dim],z[...,dim:]; return np.concatenate([a*co[...,:dim]+rot(a)*si[...,:dim],b],-1)
 return z*co+rot(z)*si

def torch_rope(x,cfg):
 typ=getattr(cfg,'model_type','')
 if typ not in ('gpt_neox','llama','mistral','qwen2'): return x
 dim=int((cfg.hidden_size//cfg.num_attention_heads)*float(cfg.rotary_pct)) if typ=='gpt_neox' else x.shape[-1]
 theta=float(getattr(cfg,'rope_theta',10000.)); idx=torch.arange(0,dim,2,dtype=x.dtype,device=x.device); inv=1/(theta**(idx/dim)); pos=torch.arange(x.shape[0],dtype=x.dtype,device=x.device); ang=pos[:,None]*inv[None,:]
 co=torch.cat([ang.cos(),ang.cos()],-1);si=torch.cat([ang.sin(),ang.sin()],-1)
 if typ=='gpt_neox':
  co=torch.cat([co,torch.ones((len(pos),x.shape[-1]-dim),device=x.device,dtype=x.dtype)],-1);si=torch.cat([si,torch.zeros((len(pos),x.shape[-1]-dim),device=x.device,dtype=x.dtype)],-1)
 def rot(a): h=a.shape[-1]//2; return torch.cat([-a[...,h:],a[...,:h]],-1)
 if typ=='gpt_neox':
  a,b=x[...,:dim],x[...,dim:]; return torch.cat([a*co[...,:dim]+rot(a)*si[...,:dim],b],-1)
 return x*co+rot(x)*si

def nearest(i,n,m): return int(round(i*(n-1)/max(1,m-1)))
def cache_roles(src_cache,tgt_cache,src_cfg,tgt_cfg):
 nl=tgt_cfg.num_hidden_layers; nh=getattr(tgt_cfg,'num_key_value_heads',None) or tgt_cfg.num_attention_heads; n=min(src_cache[0][0].shape[1],tgt_cache[0][0].shape[1]); roles=nl*nh*2
 x=np.empty((n,roles,D),np.float32);y=np.empty_like(x)
 sh=getattr(src_cfg,'num_key_value_heads',None) or src_cfg.num_attention_heads
 for l in range(nl):
  sl=nearest(l,src_cfg.num_hidden_layers,nl)
  for h in range(nh):
   hs=nearest(h,sh,nh)
   for kv in range(2):
    r=(l*nh+h)*2+kv; xv=src_cache[sl][kv][hs,:n];yv=tgt_cache[l][kv][h,:n]
    if kv==0: xv=inv_rope_np(xv,src_cfg);yv=inv_rope_np(yv,tgt_cfg)
    x[:,r]=xv;y[:,r]=yv
 return x,y

def prepare(rows,models):
 out={t:{'map_fit':[],'check':[],'test':[]} for t in TARGETS}; source_seconds={}
 for part,items in rows.items():
  for item in items:
   src_ids=item['ids'][SRC][:-2]; query_src=item['ids'][SRC][-2]
   sc,_,ss,se=cache_numpy(models[SRC],src_ids); source_seconds[item['row']]=ss+se
   for target in TARGETS:
    ids=item['ids'][target]; tc,to,ts,_=cache_numpy(models[target],ids[:-2]); query=ids[-2]; label=ids[-1]
    x,y=cache_roles(sc,tc,models[SRC].config,models[target].config)
    tr=time.perf_counter()
    with torch.inference_mode(): no=models[target](input_ids=torch.tensor([[query]]),past_key_values=DynamicCache.from_legacy_cache(to.past_key_values.to_legacy_cache() if hasattr(to.past_key_values,'to_legacy_cache') else to.past_key_values),use_cache=True)
    replay_seconds=time.perf_counter()-tr
    native=float(F.cross_entropy(no.logits[0,-1],torch.tensor(label,dtype=torch.long)))
    out[target][part].append({'item':item,'x':x,'y':y,'native_nll':native,'native_prefill_seconds':ts,'native_replay_seconds':replay_seconds,'query':query,'label':label,'source_prefill_seconds':ss})
 return out

def fit_global(data_by_target):
 xs=[];ys=[]
 for data in data_by_target.values():
  for d in data:
   xs.append(d['x'].reshape(-1,D));ys.append(d['y'].reshape(-1,D))
 x=np.concatenate(xs).astype(np.float64);y=np.concatenate(ys).astype(np.float64);xa=np.concatenate([x,np.ones((len(x),1))],1)
 return np.linalg.solve(xa.T@xa+ALPHA*np.diag([1.]*D+[0.]),xa.T@y).astype(np.float32)
def fit_role_maps(data):
 nr=data[0]['x'].shape[1]; out=np.empty((nr,D+1,D),np.float32)
 for r in range(nr):
  xs=np.concatenate([d['x'][:,r] for d in data]).astype(np.float64);ys=np.concatenate([d['y'][:,r] for d in data]).astype(np.float64);xa=np.concatenate([xs,np.ones((len(xs),1))],1)
  out[r]=np.linalg.solve(xa.T@xa+ALPHA*np.diag([1.]*D+[0.]),xa.T@ys).astype(np.float32)
 return out
def predict_global(x,w): return np.einsum('nri,io->nro',np.concatenate([x,np.ones((*x.shape[:2],1),np.float32)],-1),w,optimize=True)
def predict_roles(x,w): return np.einsum('nri,rio->nro',np.concatenate([x,np.ones((*x.shape[:2],1),np.float32)],-1),w,optimize=True)
def fit_codes(data,w):
 nr=data[0]['x'].shape[1];g=np.ones((nr,1),np.float32);b=np.zeros((nr,D),np.float32)
 for r in range(nr):
  z=np.concatenate([predict_global(d['x'],w)[:,r] for d in data]);y=np.concatenate([d['y'][:,r] for d in data]);zm=z.mean(0);ym=y.mean(0);den=float(np.square(z-zm).sum())
  if den>1e-20:g[r,0]=float(((z-zm)*(y-ym)).sum()/den)
  b[r]=ym-g[r,0]*zm
 return g,b
def apply(target,d,method,bundle):
 if method=='shared_global_view':
  z=predict_global(d['x'],bundle['shared']);g,b=bundle['codes'][target];return z*g[None,:,:]+b[None,:,:]
 if method=='target_global':return predict_global(d['x'],bundle['target_global'][target])
 if method=='full_ridge':return predict_roles(d['x'],bundle['full'][target])
 raise ValueError(method)
def to_past(pred,prefix,cfg):
 nh=getattr(cfg,'num_key_value_heads',None) or cfg.num_attention_heads; result=[]
 for l in range(cfg.num_hidden_layers):
  ks=[];vs=[]
  for h in range(nh):
   r=(l*nh+h)*2;k=torch_rope(pred[:prefix,r],cfg);v=pred[:prefix,r+1];ks.append(k[None,None]);vs.append(v[None,None])
  result.append((torch.cat(ks,dim=1),torch.cat(vs,dim=1)))
 return tuple(result)

def serialize(mode,shared,codes,tg,full):
 ids={'shared_global_view':0,'target_global':1,'full_ridge':2};meta={'mode':mode,'source':SRC,'targets':TARGETS,'dims':{t:[NLAYERS[t],NHEADS[t]] for t in TARGETS}}
 b=bytearray(struct.pack('<5sBI',b'MA886',ids[mode],len(json.dumps(meta,separators=(',',':')).encode())));b.extend(json.dumps(meta,separators=(',',':')).encode())
 def put(a): b.extend(np.asarray(a,dtype='<f4').tobytes())
 if mode=='shared_global_view':
  put(shared)
  for t in TARGETS:put(codes[t][0]);put(codes[t][1])
 elif mode=='target_global':
  for t in TARGETS:put(tg[t])
 else:
  for t in TARGETS:put(full[t])
 return bytes(b)
def deserialize(raw):
 magic,mode,nm=struct.unpack_from('<5sBI',raw);pos=struct.calcsize('<5sBI');meta=json.loads(raw[pos:pos+nm]);pos+=nm; bundle={'shared':None,'codes':{},'target_global':{},'full':{}}
 def get(shape):
  nonlocal pos;n=int(np.prod(shape));a=np.frombuffer(raw,dtype='<f4',count=n,offset=pos).reshape(shape).copy();pos+=n*4;return a
 if mode==0:
  bundle['shared']=get((D+1,D))
  for t in TARGETS:
   nr=NLAYERS[t]*NHEADS[t]*2;bundle['codes'][t]=(get((nr,1)),get((nr,D)))
 elif mode==1:
  for t in TARGETS:bundle['target_global'][t]=get((D+1,D))
 else:
  for t in TARGETS:bundle['full'][t]=get((NLAYERS[t]*NHEADS[t]*2,D+1,D))
 if pos!=len(raw):raise ValueError('payload trailing bytes')
 bundle['mode']=mode;return bundle

def evaluate(data_by_target,bundles,models,methods):
 result={'native':{t:[] for t in TARGETS},'methods':{m:{t:[] for t in TARGETS} for m in methods},'timing':{m:{'apply':[],'replay':[]} for m in methods},'native_timing':{t:[] for t in TARGETS},'identity_error':[]}
 for t,parts in data_by_target.items():
  for d in parts:
   item=d['item'];prefix=len(item['ids'][t])-2;query=d['query'];label=d['label'];result['native'][t].append(d['native_nll']);result['native_timing'][t].append(d['native_prefill_seconds']+d['native_replay_seconds'])
   for method in methods:
    bundle=bundles[method];ta=time.perf_counter();pred=apply(t,d,method,bundle);past=to_past(torch.as_tensor(pred),prefix,models[t].config);at=time.perf_counter()-ta
    tr=time.perf_counter()
    with torch.inference_mode():o=models[t](input_ids=torch.tensor([[query]]),past_key_values=DynamicCache.from_legacy_cache(past),use_cache=True)
    rt=time.perf_counter()-tr;val=float(F.cross_entropy(o.logits[0,-1],torch.tensor(label,dtype=torch.long)));result['methods'][method][t].append(val);result['timing'][method]['apply'].append(at);result['timing'][method]['replay'].append(rt)
 for t in TARGETS:
  # identity is exact numerical check on native path metrics and stored model-native output.
  result['native'][t+'_mean']=float(np.mean(result['native'][t]));result['native_timing'][t+'_mean']=float(np.mean(result['native_timing'][t]))
 for method in methods:
  result['methods'][method+'_mean']={t:float(np.mean(result['methods'][method][t])) for t in TARGETS}
  result['methods'][method+'_per_target']={t:float(np.mean(result['methods'][method][t])) for t in TARGETS}
  # Each prompt has one source prefill shared across both targets, charged once in combined fanout latency.
  data_order=[(t,d) for t in TARGETS for d in data_by_target[t]]
  source_by_row={d['item']['row']:d['source_prefill_seconds'] for t in TARGETS for d in data_by_target[t]}
  native_by_row={}
  for t in TARGETS:
   for d in data_by_target[t]:
    row=d['item']['row'];native_by_row[row]=native_by_row.get(row,0.)+d['native_prefill_seconds']+d['native_replay_seconds']
  fanout_by_row=dict(source_by_row)
  for idx,(t,d) in enumerate(data_order):
   row=d['item']['row'];fanout_by_row[row]+=result['timing'][method]['apply'][idx]+result['timing'][method]['replay'][idx]
  result['timing'][method+'_mean_fanout_seconds_per_prompt']=float(np.mean(list(fanout_by_row.values()))) if fanout_by_row else None
  result['timing'][method+'_mean_native_seconds_per_prompt']=float(np.mean(list(native_by_row.values()))) if native_by_row else None
 return result

def train_bundle(data):
 fit={t:data[t]['map_fit'] for t in TARGETS}; shared=fit_global(fit); codes={t:fit_codes(fit[t],shared) for t in TARGETS};tg={t:fit_global({t:fit[t]}) for t in TARGETS};full={t:fit_role_maps(fit[t]) for t in TARGETS}
 return {'shared_global_view':{'shared':shared,'codes':codes},'target_global':{'target_global':tg},'full_ridge':{'full':full}}, {'shared':shared,'codes':codes,'target_global':tg,'full':full}

def run(mode,seeds=SEEDS):
 models,toks,paths=load_assets();groups=prompt_groups(toks);rows=[]
 selected=(('0',) if mode=='dev' else tuple(str(i) for i in range(1,4)))
 for ix,bucket in enumerate(selected):
  prep_t=time.perf_counter();prep=prepare(groups[bucket],models);prep_seconds=time.perf_counter()-prep_t;fit_t=time.perf_counter();bundles,fitdata=train_bundle(prep);fit_seconds=time.perf_counter()-fit_t;methods=('shared_global_view','target_global','full_ridge');ev=evaluate({t:prep[t]['test'] for t in TARGETS},bundles,models,methods)
  raw={m:serialize(m,fitdata['shared'],fitdata['codes'],fitdata['target_global'],fitdata['full']) for m in methods};dec={m:deserialize(raw[m]) for m in methods}
  # Canonicalized bundles let the saved-payload replay compare the actual bytes with the in-memory model.
  decoded={'shared_global_view':{'shared':dec['shared_global_view']['shared'],'codes':dec['shared_global_view']['codes']},'target_global':{'target_global':dec['target_global']['target_global']},'full_ridge':{'full':dec['full_ridge']['full']}}
  rt=evaluate({t:prep[t]['test'] for t in TARGETS},decoded,models,methods)
  roundtrip={m:max(abs(ev['methods'][m+'_per_target'][t]-rt['methods'][m+'_per_target'][t]) for t in TARGETS) for m in methods}
  payload={m:len(raw[m]) for m in methods}; hashes={m:hashlib.sha256(raw[m]).hexdigest() for m in methods}
  base=exp_root()/'artifacts';base.mkdir(exist_ok=True)
  artifacts={}
  for m in methods:
   p=base/f'{m}_{bucket}.bin';p.write_bytes(raw[m]);artifacts[m]=p.name
  # actual compute includes a single producer prefill per row and target mapper/replay wall time.
  row={'bucket':int(bucket),'seed':None if mode=='dev' else SEEDS[ix],'prompt_rows':{t:{k:[d['item']['row'] for d in prep[t][k]] for k in ('map_fit','check','test')} for t in TARGETS},'metrics':ev,'payload_bytes':payload,'payload_sha256':hashes,'artifact_files':artifacts,'serialization_roundtrip_max_abs_nll_delta':roundtrip,'calibration_prepare_seconds':prep_seconds,'calibration_fit_seconds':fit_seconds,'active_mapper_macs_per_token':(D*(D+1)+D)*(sum(NLAYERS[t]*NHEADS[t]*2 for t in TARGETS)),'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
  rows.append(row)
  dest=exp_root()/('development_results.json' if mode=='dev' else 'fresh_results.partial.json');dest.write_text(json.dumps({'experiment_id':'MA-886','mode':mode,'rows':rows,'models':model_provenance(paths),'cuda_available':torch.cuda.is_available()},indent=2)+'\n')
 return {'experiment_id':'MA-886','mode':mode,'rows':rows,'models':model_provenance(paths),'cuda_available':torch.cuda.is_available()}

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['dev','fresh']);a=ap.parse_args();r=run(a.mode)
 p=exp_root()/('development_results.json' if a.mode=='dev' else 'fresh_results.json');p.write_text(json.dumps(r,indent=2)+'\n');print(p)
