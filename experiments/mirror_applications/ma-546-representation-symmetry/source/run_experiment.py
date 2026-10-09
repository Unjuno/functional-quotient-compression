#!/usr/bin/env python3
"""MA-546 exact gauge transformations at one frozen Pythia MLP interface."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from pinned_tasks import TASKS,MODEL_SHA,CONFIG_SHA,TOKENIZER_SHA,REVISION,split_task,load_model,enc

LAYER=3;K_MONO=16;D_FF=2048

def prompts_for_seed(seed):
 rows=[];manifest=[]
 for tid,(name,_pairs) in enumerate(TASKS):
  support,queries=split_task(tid,seed)
  manifest.append({'task_id':tid,'name':name,'support':support,'queries':queries})
  rows.extend({'task_id':tid,'input':x,'answer':y,'text':f'Input: {x}\nOutput:'} for x,y in queries)
 return rows,manifest

def hadamard(n):
 idx=np.arange(n,dtype=np.uint32)
 # H[i,j] = (-1)^popcount(i & j), recursively defined Sylvester matrix.
 bits=np.arange(n,dtype=np.uint32)
 parity=np.zeros((n,n),dtype=np.uint8)
 for b in range(int(np.log2(n))):parity^=(((idx[:,None]>>b)&1)&((bits[None,:]>>b)&1)).astype(np.uint8)
 return (1.-2.*parity.astype(np.float32))/np.sqrt(np.float32(n))

def make_codes(seed,width):
 rng=np.random.default_rng(seed)
 perm=rng.permuted(np.tile(np.arange(width,dtype=np.uint16),(K_MONO,1)),axis=1)
 signs=rng.choice(np.array([-1,1],np.int8),size=(K_MONO,width))
 scales=np.exp(rng.uniform(np.log(.5),np.log(2.),size=(K_MONO,width))).astype(np.float32)
 rows=rng.permutation(width).astype(np.uint16);cols=rng.permutation(width).astype(np.uint16)
 rs=rng.choice(np.array([-1,1],np.int8),size=width);cs=rng.choice(np.array([-1,1],np.int8),size=width)
 return perm,signs,scales,rows,cols,rs,cs

def get_batch(model,tok,torch,texts):
 seqs=[enc(tok,t) for t in texts];pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
 maxlen=max(map(len,seqs));ids=torch.full((len(seqs),maxlen),pad,dtype=torch.long);mask=torch.zeros_like(ids)
 for i,s in enumerate(seqs):ids[i,:len(s)]=torch.tensor(s);mask[i,:len(s)]=1
 return ids,mask,np.asarray([len(x)-1 for x in seqs],dtype=np.int64),sum(map(len,seqs))

def capture_base(model,tok,torch,texts):
 ids,mask,pos,ntokens=get_batch(model,tok,torch,texts);box={};mlp=model.gpt_neox.layers[LAYER].mlp
 def capture(module,args):box['z']=args[0].detach().clone()
 h=mlp.dense_4h_to_h.register_forward_pre_hook(capture)
 t0=time.perf_counter()
 try:
  with torch.no_grad():logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
 finally:h.remove()
 sec=time.perf_counter()-t0
 final=logits[torch.arange(len(pos)),torch.as_tensor(pos,dtype=torch.long)].detach()
 z=box['z'][torch.arange(len(pos)),torch.as_tensor(pos,dtype=torch.long)].detach()
 return ids,mask,pos,ntokens,final,z,sec

def transform_logits(model,torch,ids,mask,pos,kind,view=None,compensate=True):
 mlp=model.gpt_neox.layers[LAYER].mlp;down=mlp.dense_4h_to_h;w=down.weight.detach();b=down.bias.detach() if down.bias is not None else None
 if kind=='monomial':
  perm,sign,scale=view
  p=torch.as_tensor(perm.astype(np.int64));s=torch.as_tensor(sign,dtype=torch.float32);a=torch.as_tensor(scale,dtype=torch.float32)
  def hook(module,args,out):
   z=args[0].index_select(-1,p)* (s*a)
   wt=w.index_select(1,p)/(s*a).unsqueeze(0) if compensate else w
   return torch.nn.functional.linear(z,wt,b)
 elif kind=='hadamard':
  q=torch.as_tensor(view,dtype=torch.float32)
  def hook(module,args,out):
   z=args[0]@q.T
   return torch.nn.functional.linear(z,w@q.T,b)
 else:raise ValueError(kind)
 h=down.register_forward_hook(hook)
 try:
  with torch.no_grad():logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
 finally:h.remove()
 positions=torch.as_tensor(pos,dtype=torch.long)
 return logits[torch.arange(len(pos)),positions].detach()

def divergence(a,b,torch):
 pa=torch.softmax(a,dim=-1);pb=torch.softmax(b,dim=-1)
 loga=torch.log_softmax(a,dim=-1);logb=torch.log_softmax(b,dim=-1)
 kl=(pa*(loga-logb)).sum(-1)
 return float(kl.mean()),float(kl.max())

def npz_size(path,arrays):np.savez(path,**arrays);return Path(path).stat().st_size

def run(seed,model_dir,outdir,model=None,tok=None,torch=None):
 out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
 if model is None:model,tok,torch=load_model(model_dir)
 torch.set_num_threads(5);torch.manual_seed(seed);np.random.seed(seed)
 rows,manifest=prompts_for_seed(seed);texts=[r['text'] for r in rows]
 ids,mask,pos,ntokens,base,z,base_s=capture_base(model,tok,torch,texts)
 perms,signs,scales,rp,cp,rs,cs=make_codes(seed,D_FF)
 device=z.device;w=model.gpt_neox.layers[LAYER].mlp.dense_4h_to_h.weight.detach();bias=model.gpt_neox.layers[LAYER].mlp.dense_4h_to_h.bias.detach()
 base_local=torch.nn.functional.linear(z,w,bias)
 mono_local_diffs=[];mono_local_kl=[];t0=time.perf_counter()
 for i in range(K_MONO):
  p=torch.as_tensor(perms[i].astype(np.int64));sg=torch.as_tensor(signs[i],dtype=torch.float32);sc=torch.as_tensor(scales[i],dtype=torch.float32)
  zv=z.index_select(-1,p)*(sg*sc);wv=w.index_select(1,p)/(sg*sc).unsqueeze(0)
  y=torch.nn.functional.linear(zv,wv,bias);mono_local_diffs.append(float((y-base_local).abs().max()))
 mono_local_s=time.perf_counter()-t0
 h=hadamard(D_FF)[np.ix_(rp.astype(np.int64),cp.astype(np.int64))]
 q=h*rs.astype(np.float32)[:,None]*cs.astype(np.float32)[None,:]
 qt=torch.as_tensor(q,dtype=torch.float32);t0=time.perf_counter();dense_w=w@qt.T;dense_weight_s=time.perf_counter()-t0
 t0=time.perf_counter();zq=z@qt.T;yd=torch.nn.functional.linear(zq,dense_w,bias);dense_local_s=time.perf_counter()-t0
 dense_local_diff=float((yd-base_local).abs().max())
 # End-to-end full-vocabulary checks for one monomial, the dense orthogonal map and an uncompensated negative control.
 t0=time.perf_counter();mono_logits=transform_logits(model,torch,ids,mask,pos,'monomial',(perms[0],signs[0],scales[0]),True);mono_s=time.perf_counter()-t0
 t0=time.perf_counter();dense_logits=transform_logits(model,torch,ids,mask,pos,'hadamard',qt,True);dense_s=time.perf_counter()-t0
 t0=time.perf_counter();uncomp=transform_logits(model,torch,ids,mask,pos,'monomial',(perms[0],signs[0],scales[0]),False);uncomp_s=time.perf_counter()-t0
 def comp(x):
  delta=(x-base).abs();return {'max_abs_logit_delta':float(delta.max()),'mean_abs_logit_delta':float(delta.mean()),'top1_agreement':float((x.argmax(-1)==base.argmax(-1)).float().mean()),'mean_kl_from_base':divergence(base,x,torch)[0],'max_kl_from_base':divergence(base,x,torch)[1]}
 mono_full=comp(mono_logits);dense_full=comp(dense_logits);uncomp_full=comp(uncomp)
 common_files=['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json'];base_bytes=sum((Path(model_dir)/f).stat().st_size for f in common_files)
 meta={'task_ids':np.arange(len(TASKS),dtype=np.int16),'layer':np.array([LAYER],np.int16),'revision':np.frombuffer(REVISION.encode(),dtype='S40'),'model_sha256':np.frombuffer(MODEL_SHA.encode(),dtype='S64'),'schema':np.array([546,1],np.int32),'seed':np.array([seed],np.int64)}
 mono_bytes=npz_size(out/'monomial_views.npz',{'permutations':perms,'signs':signs,'scales':scales,**meta})
 dense_bytes=npz_size(out/'hadamard_view.npz',{'row_permutation':rp,'column_permutation':cp,'row_signs':rs,'column_signs':cs,'generator':np.frombuffer(b'sylvester_hadamard_signed_permutation_v1',dtype=np.uint8),**meta})
 (out/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 report={'experiment_id':'MA-546','seed':seed,'revision':REVISION,'model_sha256':MODEL_SHA,'prompts':len(rows),'input_tokens':ntokens,'dimensions':{'residual':512,'mlp_expansion':D_FF},'views':{'monomial_count':K_MONO,'dense_hadamard_count':1},'functional_metrics':{'compensated_monomial_local_max_abs':max(mono_local_diffs),'compensated_monomial_full_model':mono_full,'compensated_dense_local_max_abs':dense_local_diff,'compensated_dense_full_model':dense_full,'uncompensated_monomial_full_model':uncomp_full},'serialized_bytes':{'shared_model_files':base_bytes,'monomial_view_bank':mono_bytes,'dense_hadamard_view':dense_bytes,'monomial_complete_system':base_bytes+mono_bytes,'dense_complete_system':base_bytes+dense_bytes,'K_independent_models_upper':K_MONO*base_bytes},'compute':{'baseline_forward_seconds':base_s,'monomial_local_projection_16_views_seconds':mono_local_s,'monomial_full_forward_seconds':mono_s,'dense_local_transform_projection_seconds':dense_local_s,'dense_weight_compensation_seconds':dense_weight_s,'dense_full_forward_seconds':dense_s,'uncompensated_full_forward_seconds':uncomp_s,'monomial_activation_element_ops_proxy':int(K_MONO*ntokens*D_FF*2),'dense_activation_macs_proxy':int(2*ntokens*D_FF*D_FF),'dense_weight_transform_macs_proxy':int(2*512*D_FF*D_FF)},'gates':{'local_projection_delta_within_1e-5':max(max(mono_local_diffs),dense_local_diff)<=1e-5,'full_model_distribution_equivalent':mono_full['top1_agreement']==1. and dense_full['top1_agreement']==1. and max(mono_full['max_kl_from_base'],dense_full['max_kl_from_base'])<=1e-4,'uncompensated_changes_logits':uncomp_full['max_abs_logit_delta']>1e-4}}
 (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'seed':seed,'functional_metrics':report['functional_metrics'],'bytes':report['serialized_bytes'],'gates':report['gates']},indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--model-dir',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();run(a.seed,a.model_dir,a.out)
if __name__=='__main__':main()
