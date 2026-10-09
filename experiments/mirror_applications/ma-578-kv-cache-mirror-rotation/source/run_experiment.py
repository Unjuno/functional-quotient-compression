#!/usr/bin/env python3
"""MA-578 quantized KV-cache rotation code screen on Pythia/WikiText-2."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,GPTNeoXForCausalLM
from transformers.cache_utils import DynamicCache
MODEL_REV='e93a9faa9c77e5d09219f6c868bfc7a1bd65593c'
MODEL_SHA='3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd'
LAYERS=6;HEADS=8;DIM=64;GROUP=32;BLOCKS=2;ROLES=LAYERS*HEADS*2;K=16;KCB=4;PREFIX=64
DATA_SHA={'train.txt':'9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f','valid.txt':'f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8','test.txt':'d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fwht(x):
 y=np.array(x,dtype=np.float32,copy=True);n=y.shape[-1];h=1
 while h<n:
  z=y.reshape(*y.shape[:-1],n//(2*h),2,h);a=z[...,0,:].copy();b=z[...,1,:].copy();z[...,0,:]=a+b;z[...,1,:]=a-b;h*=2
 return y
def new_code(rng):return rng.choice(np.array([-1,1],np.int8),(BLOCKS,GROUP)),np.stack([rng.permutation(GROUP).astype(np.uint8) for _ in range(BLOCKS)])
def rotate(x,code):
 s,p=code;z=fwht(x.reshape(*x.shape[:-1],BLOCKS,GROUP))*(1/np.sqrt(GROUP));z*=s
 return np.take_along_axis(z,np.broadcast_to(p,z.shape),axis=-1).reshape(x.shape)
def basis(code):
 eye=np.eye(DIM,dtype=np.float32);return rotate(eye,code)
def unrotate(x,code):return x@basis(code).T
def qint4(x):
 g=x.reshape(-1,BLOCKS,GROUP);sc=(np.max(np.abs(g),axis=-1)/7).astype(np.float16);sc=np.maximum(sc.astype(np.float32),1e-8).astype(np.float16)
 q=np.clip(np.rint(g/sc.astype(np.float32)[...,None]),-7,7).astype(np.int8);q=(q+8).astype(np.uint8).reshape(-1,DIM)
 return (q[:,0::2]<<4)|q[:,1::2],sc
def dqint4(p,s):
 a=(p>>4).astype(np.int8)-8;b=(p&15).astype(np.int8)-8;q=np.stack((a,b),-1).reshape(-1,DIM).astype(np.float32)
 return (q.reshape(-1,BLOCKS,GROUP)*s.astype(np.float32)[...,None]).reshape(-1,DIM)
def quant_restore(x,code):
 z=rotate(x,code) if code is not None else x.copy();p,s=qint4(z);d=dqint4(p,s);d=unrotate(d,code) if code is not None else d
 return p,s,d
def load_text_tokens(path,tokenizer):
 raw=path.read_text(encoding='utf-8');return np.asarray(tokenizer.encode(raw,add_special_tokens=False),dtype=np.int32)
def offsets(tokens,seed,n,prefix=PREFIX):
 rng=np.random.default_rng(seed);high=len(tokens)-prefix-2
 if high<=1:raise ValueError('not enough tokens')
 return rng.choice(high,size=n,replace=False).tolist()
def windows(tokens,starts):return [tokens[i:i+PREFIX+2].astype(np.int64) for i in starts]
def cache_arrays(past):
 return [[x.detach().cpu().numpy().astype(np.float32,copy=False) for x in layer] for layer in past]
def cache_role(caches,role):
 layer=role//16;slot=role%2;head=(role%16)//2
 return [x[layer][slot][0,head] for x in caches]
def candidate_error(caches,code):
 out=[]
 for role in range(ROLES):
  vals=[]
  for x in cache_role(caches,role):
   _,_,d=quant_restore(x,code);vals.append(np.sum((d-x)**2,dtype=np.float64)/max(np.sum(x.astype(np.float64)**2),1e-20))
  out.append(float(np.mean(vals)))
 return np.asarray(out)
def fit_codebook(err,k=KCB):
 chosen=[];remain=set(range(err.shape[1]))
 for _ in range(k):
  j=min(remain,key=lambda z:float(np.mean(np.min(err[:,chosen+[z]],axis=1))))
  chosen.append(j);remain.remove(j)
 return np.asarray(chosen,np.int16)
def quantize_past(past,method,codes,assignments):
 t0=time.perf_counter();out=[];qcat=[];scat=[];errs=[]
 for li,(key,value) in enumerate(past):
  pair=[]
  for vi,t in enumerate((key,value)):
   arr=t.detach().cpu().numpy().astype(np.float32,copy=False);rec=np.empty_like(arr)
   for h in range(HEADS):
    role=li*16+h*2+vi;ci=assignments[role];code=None if method=='identity_int4' else codes[int(ci)]
    p,s,d=quant_restore(arr[0,h],code);rec[0,h]=d;qcat.append(p.ravel());scat.append(s.ravel());errs.append((d-arr[0,h]).astype(np.float32))
   pair.append(torch.from_numpy(rec))
  out.append(tuple(pair))
 return tuple(out),np.concatenate(qcat),np.concatenate(scat),time.perf_counter()-t0

def serialize_method(path,method,past,fp16_cache,qcat,scat,codes,assign,seed):
 args={'seed':np.array([seed],np.int32),'method':np.array([method.encode()],dtype='S32'),'cache_shape':np.asarray([LAYERS,2,HEADS,PREFIX,DIM],np.int32),'schema':np.array([578,1],np.int32)}
 if method=='fp16_cache':
  args['kv_fp16']=np.concatenate([x.detach().cpu().numpy().astype(np.float16).ravel() for layer in past for x in layer])
 elif method!='identity_int4':
  args['kv_q_int4']=qcat;args['kv_scales_fp16']=scat
  args['signs']=np.stack([c[0] for c in codes]);args['permutations']=np.stack([c[1] for c in codes]);args['role_code_ids']=np.asarray(assign,np.uint8)
 else:args['kv_q_int4']=qcat;args['kv_scales_fp16']=scat
 np.savez(path,**args);return path.stat().st_size
def score_query(model,query,target,past):
 q=torch.tensor([[int(query)]],dtype=torch.long);t0=time.perf_counter()
 with torch.inference_mode():o=model(input_ids=q,past_key_values=DynamicCache.from_legacy_cache(past),use_cache=False,return_dict=True)
 sec=time.perf_counter()-t0;loss=float(torch.nn.functional.cross_entropy(o.logits[:,-1,:].float(),torch.tensor([int(target)])))
 return loss,sec
def run(seed,split,model_dir,data_dir,out):
 wall=time.perf_counter();torch.set_num_threads(4);model_dir=Path(model_dir);data_dir=Path(data_dir);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if sha(model_dir/'model.safetensors')!=MODEL_SHA:raise ValueError('pinned Pythia hash mismatch')
 for n,h in DATA_SHA.items():
  if sha(data_dir/n)!=h:raise ValueError(f'WikiText hash mismatch: {n}')
 tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True);t0=time.perf_counter();model=GPTNeoXForCausalLM.from_pretrained(model_dir,local_files_only=True,torch_dtype=torch.float32).eval();model_load=time.perf_counter()-t0
 train=load_text_tokens(data_dir/'train.txt',tokenizer);eval_file='valid.txt' if split=='dev' else 'test.txt';ev=load_text_tokens(data_dir/eval_file,tokenizer)
 calwins=windows(train,offsets(train,seed+100,4));evalwins=windows(ev,offsets(ev,seed+200,16))
 def prefill(win):
  inp=torch.tensor(win[:PREFIX][None,:],dtype=torch.long)
  with torch.inference_mode():return model(input_ids=inp,use_cache=True,return_dict=True)
 t0=time.perf_counter();cal_caches=[cache_arrays(prefill(w).past_key_values) for w in calwins];prefill_cal_seconds=time.perf_counter()-t0
 rng=np.random.default_rng(seed);bank=[new_code(rng) for _ in range(K)];t0=time.perf_counter();err=np.stack([candidate_error(cal_caches,c) for c in bank],axis=1);candidate_error_seconds=time.perf_counter()-t0
 t0=time.perf_counter();cb=fit_codebook(err);ind=np.argmin(err,axis=1).astype(np.uint8);global_id=int(np.argmin(np.mean(err,axis=0)));rand_ids=np.argmin(err[:,np.arange(KCB)],axis=1).astype(np.uint8);cb_ids=np.argmin(err[:,cb],axis=1).astype(np.uint8);fit_seconds=time.perf_counter()-t0
 method_codes={'fp16_cache':[],'identity_int4':[],'quarot_global':[bank[global_id]],'independent16':[bank[int(i)] for i in ind], 'mirror_codebook4':[bank[int(i)] for i in cb], 'random_codebook4':[bank[i] for i in range(KCB)],'native_shared_codebook4':[bank[int(i)] for i in cb]}
 method_ids={'fp16_cache':np.zeros(ROLES,np.uint8),'identity_int4':np.zeros(ROLES,np.uint8),'quarot_global':np.zeros(ROLES,np.uint8),'independent16':np.arange(ROLES,dtype=np.uint8),'mirror_codebook4':cb_ids,'random_codebook4':rand_ids,'native_shared_codebook4':cb_ids}
 losses={m:[] for m in method_codes};nll_times={m:[] for m in method_codes};enc_times={m:[] for m in method_codes};cache_err={m:[] for m in method_codes};sizes={};sample_cache=None
 for wi,w in enumerate(evalwins):
  o=prefill(w);past=o.past_key_values;orig=cache_arrays(past)
  query,target=int(w[PREFIX]),int(w[PREFIX+1])
  for method,cs in method_codes.items():
   if method=='fp16_cache':
    t0=time.perf_counter();newpast=tuple(tuple(x.detach().cpu().half().float() for x in layer) for layer in past);enc=time.perf_counter()-t0;qcat=scat=None;ce=float(np.mean([np.mean((newpast[l][v].numpy()-past[l][v].detach().cpu().numpy())**2) for l in range(LAYERS) for v in range(2)]))
   else:
    if method=='identity_int4':assign=np.zeros(ROLES,dtype=np.uint8);use_codes=[]
    elif method=='quarot_global':assign=np.zeros(ROLES,dtype=np.uint8);use_codes=cs
    elif method=='independent16':assign=np.arange(ROLES,dtype=np.uint8);use_codes=cs
    elif method=='random_codebook4':assign=rand_ids;use_codes=cs
    else:assign=cb_ids;use_codes=cs
    newpast,qcat,scat,enc=quantize_past(past,method,use_codes,assign)
    ce=float(np.mean([np.mean((newpast[l][v].numpy()-past[l][v].numpy())**2) for l in range(LAYERS) for v in range(2)]))
   loss,qt=score_query(model,query,target,newpast);losses[method].append(loss);nll_times[method].append(qt);enc_times[method].append(enc);cache_err[method].append(ce)
   if wi==0:
    p=out/f'{method}_cache_payload.npz';sz=serialize_method(p,method,past,None,qcat,scat,cs,assign if method!='fp16_cache' else None,seed);sizes[method]={'payload_bytes':sz,'payload_file':p.name}
  if wi==0:sample_cache=orig
 methods={}
 for method in method_codes:
  methods[method]={'mean_next_token_nll':float(np.mean(losses[method])),'std_next_token_nll':float(np.std(losses[method],ddof=1)),'nll_delta_vs_fp16':float(np.mean(losses[method])-np.mean(losses['fp16_cache'])),'actual_cache_payload_bytes':sizes[method]['payload_bytes'],'mean_cache_encode_seconds':float(np.mean(enc_times[method])),'mean_query_seconds':float(np.mean(nll_times[method])),'mean_cache_mse':float(np.mean(cache_err[method]))}
 # Add one charged calibration-state file for shared and independent rotation banks: method cache files already include full codebooks/IDs.
 report={'experiment_id':'MA-578','seed':seed,'split':split,'model_revision':MODEL_REV,'model_sha256':MODEL_SHA,'dataset':'WikiText-2 raw','dataset_file':eval_file,'dataset_sha256':DATA_SHA[eval_file],'prefix_tokens':PREFIX,'query_target_pairs':len(evalwins),'calibration_prefixes':len(calwins),'candidate_count':K,'codebook_size':KCB,'selected_global_code':global_id,'fitted_codebook_candidate_ids':cb.tolist(),'methods':methods,'file_sizes':sizes,'compute':{'model_load_seconds':model_load,'calibration_prefill_seconds':prefill_cal_seconds,'codebook_fit_seconds':fit_seconds,'candidate_error_seconds':candidate_error_seconds, 'total_wall_seconds':time.perf_counter()-wall,'online_query_and_value_view_ops_per_token':LAYERS*HEADS*2*BLOCKS*GROUP*5,'optimizer_updates':0},'note':'NLL is measured on one next token after a quantized 64-token prefix cache; only prefix cache is quantized, query-token cache is not retained.'}
 (out/'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'seed':seed,'methods':methods,'native_alias':methods['mirror_codebook4']['mean_next_token_nll']==methods['native_shared_codebook4']['mean_next_token_nll'] and methods['mirror_codebook4']['actual_cache_payload_bytes']==methods['native_shared_codebook4']['actual_cache_payload_bytes']},indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['dev','fresh'],required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.seed,a.split,a.model_dir,a.data_dir,a.out)
if __name__=='__main__':main()
