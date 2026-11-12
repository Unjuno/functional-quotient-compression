"""MA-887 audit of exact RoPE position Views and approximate cross-model ridge."""
from __future__ import annotations
import os
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('HF_DATASETS_OFFLINE','1')
import hashlib,json,resource,struct,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
SRC='EleutherAI/pythia-70m-deduped';TGT='HuggingFaceTB/SmolLM2-135M';MODELS=(SRC,TGT);D=64;ALPHA=.01;DELTA=17;SEEDS=(11887,11891,11897)
def root():return Path(__file__).resolve().parents[1]
def snap(mid):
 p=Path.home()/'.cache/huggingface/hub'/('models--'+mid.replace('/','--'))/'snapshots';return sorted(p.iterdir())[0]
def assets():
 torch.set_num_threads(min(4,os.cpu_count() or 1)); models={};toks={};paths={}
 for mid in MODELS:
  paths[mid]=snap(mid);toks[mid]=AutoTokenizer.from_pretrained(paths[mid],local_files_only=True,use_fast=True);models[mid]=AutoModelForCausalLM.from_pretrained(paths[mid],local_files_only=True,attn_implementation='eager').eval()
  for p in models[mid].parameters():p.requires_grad_(False)
 return models,toks,paths
def provenance(paths):
 out={}
 for mid,p in paths.items():
  h=hashlib.sha256();n=0;files=sorted(p.glob('*.safetensors'))
  for f in files:h.update(f.name.encode());h.update(f.read_bytes());n+=f.stat().st_size
  out[mid]={'snapshot':p.name,'weights_sha256':h.hexdigest(),'checkpoint_bytes':n}
 return out
def groups(toks):
 m=json.loads((root()/'prompt_manifest.json').read_text());ds=load_dataset('Salesforce/wikitext','wikitext-103-raw-v1',split='train');out={}
 for b,items in m['buckets'].items():
  rows=[]
  for r in items:
   text=ds[int(r['row'])]['text'].strip();assert hashlib.sha256(text.encode()).hexdigest()==r['text_sha256'];enc={mid:toks[mid](text,add_special_tokens=False,return_offsets_mapping=True) for mid in MODELS};assert enc[SRC]['offset_mapping']==enc[TGT]['offset_mapping'];assert len(enc[SRC]['input_ids'])==r['tokens']
   rows.append({'row':int(r['row']),'title_sha256':r['title_sha256'],'text':text,'ids':{mid:enc[mid]['input_ids'] for mid in MODELS},'tokens':r['tokens']})
  out[b]={'map_fit':rows[:24],'check':rows[24:32],'test':rows[32:48]}
 return out
def rotate_np(x,cfg,offset=0,inverse=False):
 z=np.asarray(x,np.float32).copy();typ=cfg.model_type
 if typ not in ('gpt_neox','llama','mistral','qwen2'):return z
 dim=int((cfg.hidden_size//cfg.num_attention_heads)*float(cfg.rotary_pct)) if typ=='gpt_neox' else z.shape[-1];theta=float(getattr(cfg,'rope_theta',10000.));inv=1/(theta**(np.arange(0,dim,2,dtype=np.float64)/dim));pos=np.arange(z.shape[-2],dtype=np.float64)+offset;ang=np.outer(pos,inv)
 co=np.concatenate([np.cos(ang),np.cos(ang)],-1);si=np.concatenate([np.sin(ang),np.sin(ang)],-1)*(-1 if inverse else 1)
 if typ=='gpt_neox':co=np.concatenate([co,np.ones((len(co),z.shape[-1]-dim))],-1);si=np.concatenate([si,np.zeros((len(si),z.shape[-1]-dim))],-1)
 shape=(1,)*(z.ndim-2)+(len(co),co.shape[-1]);co=co.reshape(shape);si=si.reshape(shape)
 def rot(a):h=a.shape[-1]//2;return np.concatenate([-a[...,h:],a[...,:h]],-1)
 if typ=='gpt_neox':
  a,b=z[...,:dim],z[...,dim:];return np.concatenate([a*co[...,:dim]+rot(a)*si[...,:dim],b],-1)
 return z*co+rot(z)*si
def rope_t(x,cfg,positions):
 typ=cfg.model_type
 if typ not in ('gpt_neox','llama','mistral','qwen2'):return x
 dim=int((cfg.hidden_size//cfg.num_attention_heads)*float(cfg.rotary_pct)) if typ=='gpt_neox' else x.shape[-1];theta=float(getattr(cfg,'rope_theta',10000.));idx=torch.arange(0,dim,2,dtype=x.dtype,device=x.device);inv=1/(theta**(idx/dim));pos=positions.to(device=x.device,dtype=x.dtype);ang=pos[:,None]*inv[None,:];co=torch.cat([ang.cos(),ang.cos()],-1);si=torch.cat([ang.sin(),ang.sin()],-1)
 if typ=='gpt_neox':co=torch.cat([co,torch.ones((len(pos),x.shape[-1]-dim),device=x.device,dtype=x.dtype)],-1);si=torch.cat([si,torch.zeros((len(pos),x.shape[-1]-dim),device=x.device,dtype=x.dtype)],-1)
 def rot(a):h=a.shape[-1]//2;return torch.cat([-a[...,h:],a[...,:h]],-1)
 if typ=='gpt_neox':
  a,b=x[...,:dim],x[...,dim:];return torch.cat([a*co[...,:dim]+rot(a)*si[...,:dim],b],-1)
 return x*co+rot(x)*si
def forward(model,ids,offset=0,past=None,cache_position=None):
 q=torch.tensor([ids],dtype=torch.long);pos=torch.arange(len(ids),dtype=torch.long)[None,:]+offset;kw={'input_ids':q,'position_ids':pos,'use_cache':True}
 if past is not None:
  past_len=past[0][0].shape[-2]
  def batch(a):
   t=torch.as_tensor(a,dtype=torch.float32);return t[None] if t.ndim==3 else t
  past_t=tuple((batch(k),batch(v)) for k,v in past);kw['past_key_values']=DynamicCache.from_legacy_cache(past_t);kw['attention_mask']=torch.ones((1,past_len+len(ids)),dtype=torch.long);kw['cache_position']=cache_position
 with torch.inference_mode():return model(**kw)
def legacy(out):
 p=out.past_key_values
 return p.to_legacy_cache() if hasattr(p,'to_legacy_cache') else p
def cache_np(model,ids,offset=0):
 t=time.perf_counter();out=forward(model,ids,offset);modelsec=time.perf_counter()-t;tc=time.perf_counter();past=legacy(out);arr=[(k[0].cpu().numpy().astype(np.float32),v[0].cpu().numpy().astype(np.float32)) for k,v in past];return arr,out,modelsec,time.perf_counter()-tc
def nll(logits,label):return float(F.cross_entropy(logits.reshape(1,-1),torch.tensor([label],dtype=torch.long)))
def replay(model,query,label,past,position,cache_position):
 t=time.perf_counter();out=forward(model,[query],position,past,torch.tensor([cache_position],dtype=torch.long));sec=time.perf_counter()-t;return nll(out.logits[0,-1],label),out.logits[0,-1].detach().cpu().numpy(),sec
def role_data(sc,tc,scfg,tcfg):
 nl=tcfg.num_hidden_layers;nh=getattr(tcfg,'num_key_value_heads',None) or tcfg.num_attention_heads;sh=getattr(scfg,'num_key_value_heads',None) or scfg.num_attention_heads;n=min(sc[0][0].shape[1],tc[0][0].shape[1]);x=np.empty((n,nl*nh*2,D),np.float32);y=np.empty_like(x)
 def nearest(i,a,b):return int(round(i*(a-1)/max(1,b-1)))
 for l in range(nl):
  sl=nearest(l,scfg.num_hidden_layers,nl)
  for h in range(nh):
   hs=nearest(h,sh,nh)
   for kv in range(2):
    r=(l*nh+h)*2+kv;xx=sc[sl][kv][hs,:n];yy=tc[l][kv][h,:n]
    if kv==0:xx=rotate_np(xx,scfg,inverse=True);yy=rotate_np(yy,tcfg,inverse=True)
    x[:,r]=xx;y[:,r]=yy
 return x,y
def fit_maps(data):
 nr=data[0]['x'].shape[1];maps=np.empty((nr,D+1,D),np.float32);eye=np.eye(D+1);eye[-1,-1]=0
 for r in range(nr):
  xx=np.concatenate([d['x'][:,r] for d in data]).astype(np.float64);yy=np.concatenate([d['y'][:,r] for d in data]).astype(np.float64);xa=np.concatenate([xx,np.ones((len(xx),1))],1);maps[r]=np.linalg.solve(xa.T@xa+ALPHA*eye,xa.T@yy).astype(np.float32)
 return maps
def predict(x,w):return np.einsum('nri,rio->nro',np.concatenate([x,np.ones((*x.shape[:2],1),np.float32)],-1),w,optimize=True)
def target_past(pred,prefix,cfg):
 nh=getattr(cfg,'num_key_value_heads',None) or cfg.num_attention_heads;res=[]
 for l in range(cfg.num_hidden_layers):
  ks=[];vs=[]
  for h in range(nh):
   r=(l*nh+h)*2;k=rope_t(torch.as_tensor(pred[:prefix,r]),cfg,torch.arange(prefix));v=torch.as_tensor(pred[:prefix,r+1]);ks.append(k[None,None]);vs.append(v[None,None])
  res.append((torch.cat(ks,1),torch.cat(vs,1)))
 return tuple(res)
def serialize_full(maps):
 meta=json.dumps({'source':SRC,'target':TGT,'layers':30,'heads':3,'dim':D},separators=(',',':')).encode();return struct.pack('<5sBI',b'MA887',0,len(meta))+meta+np.asarray(maps,dtype='<f4').tobytes()
def serialize_phase(delta):
 meta=json.dumps({'model':SRC,'view':'global_rope_shift'},separators=(',',':')).encode();return struct.pack('<5sBI',b'MA887',1,len(meta))+meta+struct.pack('<i',delta)
def deserialize_full(raw):
 magic,mode,nm=struct.unpack_from('<5sBI',raw);pos=struct.calcsize('<5sBI');meta=json.loads(raw[pos:pos+nm]);pos+=nm
 if magic!=b'MA887' or mode!=0:raise ValueError('bad ridge payload')
 n=30*3*2*(D+1)*D;maps=np.frombuffer(raw,dtype='<f4',count=n,offset=pos).reshape(180,D+1,D).copy();pos+=n*4
 if pos!=len(raw):raise ValueError('ridge payload trailing bytes')
 return maps
def deserialize_phase(raw):
 magic,mode,nm=struct.unpack_from('<5sBI',raw);pos=struct.calcsize('<5sBI');meta=json.loads(raw[pos:pos+nm]);pos+=nm
 if magic!=b'MA887' or mode!=1 or pos+4!=len(raw):raise ValueError('bad phase payload')
 return struct.unpack_from('<i',raw,pos)[0]
def prepare(items,models):
 prepared=[]
 for item in items:
  ids=item['ids'][SRC];tids=item['ids'][TGT];q=ids[-2];label=ids[-1];prefix_ids=ids[:-2]
  sc,source_out,ss,se=cache_np(models[SRC],prefix_ids,0)
  base_nll,base_logits,base_replay=replay(models[SRC],q,label,legacy(source_out),len(prefix_ids),len(prefix_ids))
  st=time.perf_counter();shifted_full=forward(models[SRC],ids[:-1],DELTA);shift_prefill=time.perf_counter()-st;shift_nll=nll(shifted_full.logits[0,-1],label);shift_logits=shifted_full.logits[0,-1].detach().cpu().numpy()
  # Transform cached keys from physical phase p to p+Delta; V is invariant.
  tt=time.perf_counter();shifted=[]
  for k,v in sc:
   shifted.append((rotate_np(rotate_np(k,models[SRC].config,inverse=True),models[SRC].config,DELTA),v))
  transform_seconds=time.perf_counter()-tt
  view_nll,view_logits,view_replay=replay(models[SRC],q,label,shifted,DELTA+len(prefix_ids),len(prefix_ids))
  tc,to,ts,te=cache_np(models[TGT],tids[:-2]);tnll,_,treplay=replay(models[TGT],tids[-2],tids[-1],legacy(to),len(tids[:-2]),len(tids[:-2]));x,y=role_data(sc,tc,models[SRC].config,models[TGT].config)
  prepared.append({'item':item,'x':x,'y':y,'native_target_nll':tnll,'target_prefill_seconds':ts,'target_replay_seconds':treplay,'source_prefill_seconds':ss+se,'source_nll':base_nll,'shift_nll':shift_nll,'view_nll':view_nll,'base_logits':base_logits,'shift_logits':shift_logits,'view_logits':view_logits,'shift_prefill_seconds':shift_prefill,'base_replay_seconds':base_replay,'view_replay_seconds':view_replay,'cache_bytes':sum(k.nbytes+v.nbytes for k,v in sc),'shifted_key_bytes':sum(k.nbytes for k,v in shifted),'phase_transform_seconds':transform_seconds,'phase_payload_bytes':len(serialize_phase(DELTA))})
 return prepared
def evaluate_approx(data,maps,models):
 vals=[];apply=[];replaysec=[]
 for d in data:
  t=time.perf_counter();pred=predict(d['x'],maps);past=target_past(pred,len(d['item']['ids'][TGT])-2,models[TGT].config);apply.append(time.perf_counter()-t)
  v,_,s=replay(models[TGT],d['item']['ids'][TGT][-2],d['item']['ids'][TGT][-1],past,len(d['item']['ids'][TGT])-2,len(d['item']['ids'][TGT])-2);vals.append(v);replaysec.append(s)
 return {'mean_nll':float(np.mean(vals)),'per_prompt_nll':vals,'mean_apply_seconds':float(np.mean(apply)),'mean_replay_seconds':float(np.mean(replaysec))}
def exact_metrics(data):
 return {'mean_native_nll':float(np.mean([d['source_nll'] for d in data])),'mean_shifted_native_nll':float(np.mean([d['shift_nll'] for d in data])),'mean_phase_view_nll':float(np.mean([d['view_nll'] for d in data])),'max_abs_logit_delta_vs_shifted':float(max(np.max(np.abs(d['view_logits']-d['shift_logits'])) for d in data)),'max_abs_nll_delta_vs_shifted':float(max(abs(d['view_nll']-d['shift_nll']) for d in data)),'mean_cache_bytes':float(np.mean([d['cache_bytes'] for d in data])),'mean_shifted_key_bytes':float(np.mean([d['shifted_key_bytes'] for d in data])),'phase_payload_bytes':data[0]['phase_payload_bytes'],'phase_to_keycache_ratio':data[0]['phase_payload_bytes']/np.mean([d['shifted_key_bytes'] for d in data]),'mean_source_prefill_seconds':float(np.mean([d['source_prefill_seconds'] for d in data])),'mean_native_shift_prefill_seconds':float(np.mean([d['shift_prefill_seconds'] for d in data])),'mean_phase_transform_seconds':float(np.mean([d['phase_transform_seconds'] for d in data])),'mean_phase_replay_seconds':float(np.mean([d['view_replay_seconds'] for d in data]))}
def run(mode):
 models,toks,paths=assets();gs=groups(toks); buckets=['0'] if mode=='dev' else ['1','2','3'];out=[]
 for ix,b in enumerate(buckets):
  parts={k:prepare(v,models) for k,v in gs[b].items()};fit_t=time.perf_counter();fit=fit_maps(parts['map_fit']);fit_seconds=time.perf_counter()-fit_t;full=serialize_full(fit);saved_maps=deserialize_full(full);dev=parts['test'];exact=exact_metrics(dev);approx=evaluate_approx(dev,fit,models);saved_approx=evaluate_approx(dev,saved_maps,models)
  approx['serialization_roundtrip_max_abs_nll_delta']=abs(approx['mean_nll']-saved_approx['mean_nll']);phase_raw=serialize_phase(DELTA);assert deserialize_phase(phase_raw)==DELTA
  approx['native_target_mean_nll']=float(np.mean([d['native_target_nll'] for d in dev]));approx['native_target_prefill_plus_replay_seconds']=float(np.mean([d['target_prefill_seconds']+d['target_replay_seconds'] for d in dev]));approx['source_prefill_plus_apply_replay_seconds']=float(np.mean([d['source_prefill_seconds'] for d in dev]))+approx['mean_apply_seconds']+approx['mean_replay_seconds']
  # Timed native shifted full-prefix prefill on test prompts; source-only cache is shared for the algebraic view.
  exact['mean_view_total_seconds']=exact['mean_source_prefill_seconds']+exact['mean_phase_transform_seconds']+exact['mean_phase_replay_seconds'];exact['mean_native_shift_total_seconds']=exact['mean_native_shift_prefill_seconds']
  mapsraw=full;art=root()/'artifacts';art.mkdir(exist_ok=True);path=art/f'full_ridge_{b}.bin';path.write_bytes(mapsraw)
  row={'bucket':int(b),'seed':None if mode=='dev' else SEEDS[ix],'prompt_rows':{k:[d['item']['row'] for d in v] for k,v in parts.items()},'exact':exact,'approximate_cross_model':approx,'full_ridge_payload_bytes':len(full),'full_ridge_sha256':hashlib.sha256(full).hexdigest(),'phase_payload_bytes':len(phase_raw),'phase_payload_sha256':hashlib.sha256(phase_raw).hexdigest(),'source_prefill_plus_phase_view_seconds':exact['mean_view_total_seconds'],'native_shifted_prefill_seconds':exact['mean_native_shift_total_seconds'],'approx_target_native_mean_nll':float(np.mean([d['native_target_nll'] for d in dev])),'source_model_prefill_plus_transform_seconds':sum(d['source_prefill_seconds'] for d in dev),'phase_view_active_macs_per_token':D*2*models[SRC].config.num_hidden_layers*models[SRC].config.num_attention_heads,'approx_mapper_macs_per_token':180*(D+1)*D,'fit_seconds':fit_seconds,'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
  out.append(row);(root()/('development_results.json' if mode=='dev' else 'fresh_results.partial.json')).write_text(json.dumps({'experiment_id':'MA-887','mode':mode,'rows':out,'models':provenance(paths),'cuda_available':torch.cuda.is_available()},indent=2)+'\n')
 return {'experiment_id':'MA-887','mode':mode,'rows':out,'models':provenance(paths),'cuda_available':torch.cuda.is_available()}
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['dev','fresh']);a=ap.parse_args();result=run(a.mode);p=root()/('development_results.json' if a.mode=='dev' else 'fresh_results.json');p.write_text(json.dumps(result,indent=2)+'\n');print(p)
