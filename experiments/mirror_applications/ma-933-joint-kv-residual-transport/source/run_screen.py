"""MA-933 frozen TinyStories residual+KV coordinate chart mechanism screen."""
import csv,json,random,time,hashlib
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer,GPTNeoForCausalLM
from transformers.cache_utils import DynamicCache
ROOT=Path(__file__).resolve().parents[1]; CACHE=Path('/tmp/ma930-cache'); CACHE.mkdir(exist_ok=True)
torch.set_num_threads(1); DEVICE='cpu'; RANK=8; UPDATES=1200; BATCH=256
MODEL=[('roneneldan/TinyStories-1M','77f1b168e219585646439073245fe87e56b3023e',64),('roneneldan/TinyStories-8M','8612e3b15c66ffa94eaa6ee0de5c96edd2d630af',256)]
DATA_REV='f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'; TRAIN_FILE='data/train-00000-of-00004-2d5a1467fff1081b.parquet'; VAL_FILE='data/validation-00000-of-00001-869c898b519ad725.parquet'
SEEDS=[93321,93327,93333]; LAYERS=list(range(8)); ROLES=['residual','key','value']
def load_texts():
 tr=hf_hub_download('roneneldan/TinyStories',TRAIN_FILE,repo_type='dataset',revision=DATA_REV,cache_dir=str(CACHE)); va=hf_hub_download('roneneldan/TinyStories',VAL_FILE,repo_type='dataset',revision=DATA_REV,cache_dir=str(CACHE))
 td=Dataset.from_parquet(tr);vd=Dataset.from_parquet(va)
 return [td[i]['text'] for i in range(768,1024)],[vd[i]['text'] for i in range(512,640)],[tr,va]
def tokens(tok,texts):
 o=tok(texts,truncation=True,max_length=32,padding='max_length',return_tensors='pt'); keep=o.attention_mask.sum(1)>=18
 return o.input_ids[keep],o.attention_mask[keep]
@torch.inference_mode()
def collect(model,ids,mask,batch=4):
 result={r:[[] for _ in LAYERS] for r in ROLES}
 for st in range(0,len(ids),batch):
  ii=ids[st:st+batch];mm=mask[st:st+batch];o=model(input_ids=ii,attention_mask=mm,output_hidden_states=True,use_cache=True)
  for l in LAYERS:
   result['residual'][l].append(o.hidden_states[l+1].detach().cpu().float())
   for role,ix in [('key',0),('value',1)]:
    kv=o.past_key_values[l][ix].permute(0,2,1,3).contiguous().reshape(ii.shape[0],ii.shape[1],-1)
    result[role][l].append(kv.detach().cpu().float())
 return {r:[torch.cat(v,0) for v in vals] for r,vals in result.items()}
def flatten(t,mask):
 return t[:,1:,:].reshape(-1,t.shape[-1])[mask[:,1:].reshape(-1).bool()]
class Chart(nn.Module):
 def __init__(self):
  super().__init__();self.A=nn.Parameter(torch.empty(RANK,64));self.B=nn.Parameter(torch.empty(256,RANK));nn.init.xavier_uniform_(self.A);nn.init.xavier_uniform_(self.B)
  self.C=nn.Parameter(torch.empty(24,RANK,RANK));nn.init.xavier_uniform_(self.C);self.bias=nn.Parameter(torch.zeros(24,256))
 def idx(self,r,l):return ROLES.index(r)*8+l
 def forward(self,x,r,l):
  i=self.idx(r,l);return ((x@self.A.T)@self.C[i].T)@self.B.T+self.bias[i]
def fit_chart(chart,src,tgt,mask,seed):
 torch.manual_seed(seed);random.seed(seed);rng=torch.Generator().manual_seed(seed+99);opt=torch.optim.Adam(chart.parameters(),lr=.003);t0=time.perf_counter()
 xs={};ys={}
 for role in ROLES:
  for l in LAYERS:xs[(role,l)]=flatten(src[role][l],mask);ys[(role,l)]=flatten(tgt[role][l],mask)
 for _ in range(UPDATES):
  role=random.choice(ROLES);l=random.choice(LAYERS);x=xs[(role,l)];y=ys[(role,l)];ix=torch.randint(len(x),(BATCH,),generator=rng);loss=F.mse_loss(chart(x[ix],role,l),y[ix]);opt.zero_grad();loss.backward();opt.step()
 return time.perf_counter()-t0

def affine_fit(x,y,ridge=1e-4):
 # x,y are (examples, channels), paid affine map; fit in float64 for stable normal equations.
 xd=torch.cat([x.double(),torch.ones((len(x),1),dtype=torch.float64)],1);yd=y.double();g=xd.T@xd;g[:-1,:-1]+=ridge*torch.eye(x.shape[1],dtype=torch.float64);coef=torch.linalg.solve(g,xd.T@yd).float();return coef[:-1],coef[-1]
def affine_apply(x,entry):return x@entry[0]+entry[1]
def map_rows(x,chart,role,l):return chart(x,role,l)
def affine_rows(x,entry):return affine_apply(x,entry)
@torch.inference_mode()
def cache_nll(src_model,tgt_model,chart_or_maps,kind,ids,mask):
 t0=time.perf_counter();valid=mask.sum(1)>=18; ids=ids[valid];mask=mask[valid];prefix=ids[:,:15];pm=mask[:,:15];step=ids[:,15:16];fullmask=mask[:,:16]
 total=0.;n=0
 for st in range(0,len(ids),4):
  si=prefix[st:st+4];sm=pm[st:st+4];ti=step[st:st+4];fm=fullmask[st:st+4]
  so=src_model(input_ids=si,attention_mask=sm,use_cache=True);to=tgt_model(input_ids=si,attention_mask=sm,use_cache=True)
  mapped=[]
  for l,(sk,sv) in enumerate(so.past_key_values):
   pair=[]
   for role,ten in [('key',sk),('value',sv)]:
    shape=ten.shape;flat=ten.permute(0,2,1,3).contiguous().reshape(shape[0],shape[2],-1).reshape(-1,shape[1]*shape[3])
    out=flat if kind=='native' else ((flat@chart_or_maps[role,l][0]+chart_or_maps[role,l][1]) if kind=='affine' else chart_or_maps(flat,role,l))
    out=out.reshape(shape[0],shape[2],16,-1).permute(0,2,1,3).contiguous();pair.append(out)
   mapped.append(tuple(pair))
  mapped_cache=DynamicCache.from_legacy_cache(tuple(mapped));native_cache=DynamicCache.from_legacy_cache(to.past_key_values)
  nm=tgt_model(input_ids=ti,attention_mask=fm,past_key_values=mapped_cache,use_cache=False).logits[:,-1,:]
  nnative=tgt_model(input_ids=ti,attention_mask=fm,past_key_values=native_cache,use_cache=False).logits[:,-1,:]
  label=ids[st:st+4,16];lp=F.cross_entropy(nm,label,reduction='sum');ln=F.cross_entropy(nnative,label,reduction='sum');total+=lp.item();n+=len(label)
 return total/n, time.perf_counter()-t0

def main():
 texts_tr,texts_va,paths=load_texts();tok=AutoTokenizer.from_pretrained(MODEL[0][0],revision=MODEL[0][1],cache_dir=str(CACHE));tok.pad_token=tok.pad_token or tok.eos_token
 ids_tr,mask_tr=tokens(tok,texts_tr);ids_va,mask_va=tokens(tok,texts_va)
 models=[];dirs=[]
 for mid,rev,_ in MODEL:
  from huggingface_hub import snapshot_download
  d=snapshot_download(mid,revision=rev,cache_dir=str(CACHE),allow_patterns=['*.bin','*.json','*.txt','*.model','*.tiktoken']);dirs.append(d);m=GPTNeoForCausalLM.from_pretrained(mid,revision=rev,cache_dir=str(CACHE)).eval()
  for p in m.parameters():p.requires_grad_(False)
  models.append(m)
 src_tr=collect(models[0],ids_tr,mask_tr);tgt_tr=collect(models[1],ids_tr,mask_tr);src_va=collect(models[0],ids_va,mask_va);tgt_va=collect(models[1],ids_va,mask_va)
 flat_src={};flat_tgt={};ind={};affine_start=time.perf_counter()
 for role in ROLES:
  for l in LAYERS:
   x=flatten(src_tr[role][l],mask_tr);y=flatten(tgt_tr[role][l],mask_tr);flat_src[role,l]=x;flat_tgt[role,l]=y;ind[role,l]=affine_fit(x,y)
 affine_fit_seconds=time.perf_counter()-affine_start
 basebytes=[sum(p.stat().st_size for p in Path(d).rglob('*') if p.is_file()) for d in dirs];rows=[]
 for seed in SEEDS:
  random.seed(seed);torch.manual_seed(seed);chart=Chart();seconds=fit_chart(chart,src_tr,tgt_tr,mask_tr,seed)
  factor_path=ROOT/'source'/'factor.pt';ind_path=ROOT/'source'/'independent.npz'
  torch.save(chart.state_dict(),factor_path);np.savez(ind_path,**{f'{r}_{l}_w':v[0].numpy() for (r,l),v in ind.items()},**{f'{r}_{l}_b':v[1].numpy() for (r,l),v in ind.items()})
  fb=factor_path.stat().st_size;ib=ind_path.stat().st_size;chart2=Chart();chart2.load_state_dict(torch.load(factor_path,map_location='cpu',weights_only=True))
  with np.load(ind_path) as q:ind2={(r,l):(torch.from_numpy(q[f'{r}_{l}_w']),torch.from_numpy(q[f'{r}_{l}_b'])) for r in ROLES for l in LAYERS}
  replay=0.
  for role in ROLES:
   for l in LAYERS:
    xx=flat_src[role,l][:8]; replay=max(replay,float((chart2(xx,role,l)-chart(xx,role,l)).detach().abs().max()))
  for role in ROLES:
   for l in LAYERS:
    xv=flatten(src_va[role][l],mask_va);yv=flatten(tgt_va[role][l],mask_va)
    mf=chart2(xv,role,l);mi=affine_rows(xv,ind2[role,l]);
    for cond,pred,conn,mac in [('shared_chart',mf,fb,RANK*64+RANK*RANK+RANK*256+256),('independent_affine',mi,ib,64*256+256)]:
     mse=F.mse_loss(pred,yv).item();rows.append({'seed':seed,'role':role,'layer':l+1,'condition':cond,'reconstruction_mse':mse,'target_next_token_nll_delta':None,'connector_bytes':conn,'full_deployment_bytes':basebytes[0]+basebytes[1]+conn,'target_only_bytes':basebytes[1],'mac_proxy':mac,'train_examples':len(ids_tr),'validation_examples':len(ids_va),'updates':UPDATES if cond=='shared_chart' else 0,'train_seconds':seconds if cond=='shared_chart' else affine_fit_seconds,'reload_max_abs_error':replay})
  nf,tf=cache_nll(models[0],models[1],chart2,'factor',ids_va,mask_va);ni,ti=cache_nll(models[0],models[1],ind2,'affine',ids_va,mask_va);nnative,tn=cache_nll(models[1],models[1],{},'native',ids_va,mask_va)
  for cond,nll,conn in [('shared_chart',nf,fb),('independent_affine',ni,ib),('native_target',nnative,0)]:rows.append({'seed':seed,'role':'joint_kv_cache','layer':'all','condition':cond,'reconstruction_mse':None,'target_next_token_nll_delta':nll-nnative,'connector_bytes':conn,'full_deployment_bytes':basebytes[0]+basebytes[1]+conn if cond!='native_target' else basebytes[1],'target_only_bytes':basebytes[1],'mac_proxy':None,'train_examples':len(ids_tr),'validation_examples':len(ids_va),'updates':UPDATES if cond=='shared_chart' else 0,'train_seconds':seconds if cond=='shared_chart' else (affine_fit_seconds if cond=='independent_affine' else 0.0),'reload_max_abs_error':replay,'cache_eval_seconds':tf if cond=='shared_chart' else (ti if cond=='independent_affine' else tn)})
  factor_path.unlink();ind_path.unlink();print('seed',seed,'factor bytes',fb,'ind bytes',ib,'NLL delta',nf-nnative,ni-nnative,'replay',replay,flush=True)
 # Apply frozen gates and persist the final status with the core rows.
 nfac=sum(r['target_next_token_nll_delta'] for r in rows if r['condition']=='shared_chart' and r['role']=='joint_kv_cache')/len(SEEDS);nind=sum(r['target_next_token_nll_delta'] for r in rows if r['condition']=='independent_affine' and r['role']=='joint_kv_cache')/len(SEEDS)
 fac_mse=[r['reconstruction_mse'] for r in rows if r['condition']=='shared_chart' and r['reconstruction_mse'] is not None];ind_mse=[r['reconstruction_mse'] for r in rows if r['condition']=='independent_affine' and r['reconstruction_mse'] is not None];mf=sum(fac_mse)/len(fac_mse);mi=sum(ind_mse)/len(ind_mse)
 decision='PROMISING' if abs(nfac-nind)<=.5 and mf<=1.25*mi else 'FAIL'
 for r in rows:r['status']=decision
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 (ROOT/'source/results.json').write_text(json.dumps({'results':rows,'decision':decision,'seeds':SEEDS,'train_rows':'768:1024','validation_rows':'512:640','train_sequences':len(ids_tr),'validation_sequences':len(ids_va),'base_model_bytes':basebytes,'dataset_revision':DATA_REV,'torch':torch.__version__,'device':DEVICE},indent=2)+'\n')
 manifest={'train_file':paths[0],'validation_file':paths[1],'train_file_bytes':Path(paths[0]).stat().st_size,'validation_file_bytes':Path(paths[1]).stat().st_size,'train_file_sha256':hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest(),'validation_file_sha256':hashlib.sha256(Path(paths[1]).read_bytes()).hexdigest(),'train_rows':'768:1024','validation_rows':'512:640','model_ids':[x[0] for x in MODEL],'model_revisions':[x[1] for x in MODEL],'base_model_bytes':basebytes,'train_sequences':len(ids_tr),'validation_sequences':len(ids_va),'models':[]}
 for md,(mid,rev,_) in zip(dirs,MODEL):
  fs=[]
  for fp in sorted(Path(md).rglob('*')):
   if fp.is_file() and fp.suffix in {'.bin','.json','.txt','.model','.tiktoken'}:fs.append({'path':str(fp.relative_to(md)),'bytes':fp.stat().st_size,'sha256':hashlib.sha256(fp.read_bytes()).hexdigest()})
  manifest['models'].append({'id':mid,'revision':rev,'files':fs})
 (ROOT/'source/data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
