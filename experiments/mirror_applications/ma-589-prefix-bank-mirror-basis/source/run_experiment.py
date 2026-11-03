#!/usr/bin/env python3
"""MA-589 multi-context shared-basis KV prefix-bank compression screen."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer,GPTNeoXForCausalLM
from transformers.cache_utils import DynamicCache
REV='e93a9faa9c77e5d09219f6c868bfc7a1bd65593c';MODEL_SHA='3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd'
DATA_SHA={'train.txt':'9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f','valid.txt':'f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8','test.txt':'d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0'}
L,H,KV,D,R,RR,PFX=6,8,2,64,128,4,64;ROLE=H*KV;F=ROLE*D
METHODS=('fp16_kv','shared_mla128','mirror_resid4','native_shared_resid4')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load_tokens(p,tok):return np.asarray(tok.encode(p.read_text(encoding='utf8'),add_special_tokens=False),np.int32)
def starts(x,seed,n):return np.random.default_rng(seed).choice(len(x)-PFX-2,size=n,replace=False).tolist()
def windows(x,idx):return [x[i:i+PFX+2].astype(np.int64) for i in idx]
def getcache(m,w):
 with torch.inference_mode():return m(input_ids=torch.tensor(w[:PFX][None,:],dtype=torch.long),use_cache=True,return_dict=True).past_key_values
def cache_np(p):return [[t.detach().cpu().numpy().astype(np.float32,copy=False) for t in lay] for lay in p]
def tomat(c,l):
 k,v=c[l];return np.stack((k[0],v[0]),axis=1).transpose(2,0,1,3).reshape(k.shape[2],F)
def frommat(x):
 y=x.reshape(x.shape[0],H,KV,D).transpose(1,2,0,3);return torch.from_numpy(y[:,0][None].copy()),torch.from_numpy(y[:,1][None].copy())
def pca(x,r):
 mu=x.mean(0,dtype=np.float64).astype(np.float32);_,_,vt=np.linalg.svd((x-mu).astype(np.float32),full_matrices=False);return mu,vt[:r].T.astype(np.float32)
def encode_residual(res,basis):
 return np.stack([res[:,r]@basis[r] for r in range(ROLE)],axis=1).astype(np.float16).astype(np.float32)
def decode_residual(codes,basis):
 return np.stack([codes[:,r]@basis[r].T for r in range(ROLE)],axis=1)
def dump(p,**kw):np.savez(p,**kw);return p.stat().st_size
def score(m,q,y,past):
 t=time.perf_counter()
 with torch.inference_mode():o=m(input_ids=torch.tensor([[int(q)]],dtype=torch.long),past_key_values=DynamicCache.from_legacy_cache(past),use_cache=False,return_dict=True)
 return float(torch.nn.functional.cross_entropy(o.logits[:,-1,:].float(),torch.tensor([int(y)]))),time.perf_counter()-t
def run(seed,split,model_dir,data_dir,out):
 t_all=time.perf_counter();torch.set_num_threads(4);model_dir=Path(model_dir);data_dir=Path(data_dir);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if sha(model_dir/'model.safetensors')!=MODEL_SHA:raise ValueError('model hash mismatch')
 for n,h in DATA_SHA.items():
  if sha(data_dir/n)!=h:raise ValueError('dataset hash mismatch '+n)
 tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True);tt=time.perf_counter();m=GPTNeoXForCausalLM.from_pretrained(model_dir,local_files_only=True,torch_dtype=torch.float32).eval();load_s=time.perf_counter()-tt
 tr=load_tokens(data_dir/'train.txt',tok);ef='valid.txt' if split=='dev' else 'test.txt';ev=load_tokens(data_dir/ef,tok)
 calw=windows(tr,starts(tr,seed+100,32));evalw=windows(ev,starts(ev,seed+200,16))
 tt=time.perf_counter();cal=[cache_np(getcache(m,w)) for w in calw];prefill_s=time.perf_counter()-tt
 tt=time.perf_counter();means=[];comps=[];bases=[]
 for l in range(L):
  xx=np.concatenate([tomat(c,l) for c in cal]);mu,e=pca(xx,R);means.append(mu);comps.append(e)
  rs=xx-(((xx-mu)@e)@e.T+mu);rr=[]
  for role in range(ROLE):rr.append(pca(rs.reshape(-1,ROLE,D)[:,role,:],RR)[1])
  bases.append(np.stack(rr))
 fit_s=time.perf_counter()-tt;f16=lambda x:np.asarray(x,dtype=np.float16)
 pca_b=dump(out/'shared_prefix_basis.npz',mean=f16(means),components=f16(comps),shape=np.asarray([L,F,R],np.int32),schema=np.asarray([589,1],np.int32))
 residual_b=dump(out/'shared_residual_basis.npz',basis=f16(bases),shape=np.asarray([L,ROLE,D,RR],np.int32),schema=np.asarray([589,2],np.int32))
 state_bytes={'fp16_kv':0,'shared_mla128':pca_b,'mirror_resid4':pca_b+residual_b,'native_shared_resid4':pca_b+residual_b}
 stats={x:{'nll':[],'mse':[],'encode':[],'reconstruct':[],'query':[]} for x in METHODS};session_bytes={}
 for wi,w in enumerate(evalw):
  raw=cache_np(getcache(m,w));decoded={}
  for method in METHODS:
   tt=time.perf_counter();layers=[];zs=[];codes=[];mses=[]
   for l in range(L):
    x=tomat(raw,l)
    if method=='fp16_kv':rec=x.astype(np.float16).astype(np.float32)
    else:
     z=((x-means[l])@comps[l]).astype(np.float16).astype(np.float32);z=z.astype(np.float16).astype(np.float32);base=z@comps[l].T+means[l]
     if method=='shared_mla128':rec=base
     else:
      res=(x-base).reshape(-1,ROLE,D);c=encode_residual(res,bases[l])
      rec=base+decode_residual(c,bases[l]).reshape(x.shape);codes.append(c.astype(np.float16))
     zs.append(z.astype(np.float16))
    mses.append(float(np.mean((rec-x)**2)));layers.append(frommat(rec))
   enc_s=time.perf_counter()-tt;decoded[method]=tuple(layers);tt=time.perf_counter()
   if method=='fp16_kv':payload={'kv_fp16':np.concatenate([a.astype(np.float16).ravel() for lay in raw for a in lay]),'shape':np.asarray([L,H,KV,PFX,D],np.int32),'schema':np.asarray([589,1],np.int32)}
   else:
    payload={'latent_fp16':np.concatenate([z.ravel() for z in zs]),'shape':np.asarray([L,PFX,R],np.int32),'schema':np.asarray([589,2],np.int32)}
    if codes:payload['residual_codes_fp16']=np.concatenate([c.ravel() for c in codes])
   rebuild_s=time.perf_counter()-tt;loss,q_s=score(m,w[PFX],w[PFX+1],decoded[method]);v=stats[method];v['nll'].append(loss);v['mse'].append(float(np.mean(mses)));v['encode'].append(enc_s);v['reconstruct'].append(rebuild_s);v['query'].append(q_s)
   if wi==0:session_bytes[method]=dump(out/f'{method}_prefix_cache.npz',**payload)
 result={};front={}
 for method,v in stats.items():
  result[method]={'mean_nll':float(np.mean(v['nll'])),'std_nll':float(np.std(v['nll'],ddof=1)),'nll_delta_vs_fp16':float(np.mean(v['nll'])-np.mean(stats['fp16_kv']['nll'])),'mean_cache_reconstruction_mse':float(np.mean(v['mse'])),'session_cache_bytes':session_bytes[method],'mean_encode_seconds':float(np.mean(v['encode'])),'mean_reconstruct_seconds':float(np.mean(v['reconstruct'])),'mean_query_seconds':float(np.mean(v['query']))}
  b=state_bytes[method];front[method]={'model_state_bytes':b,'total_bytes_for_1_context':b+session_bytes[method],**{f'total_bytes_for_{n}_contexts':b+n*session_bytes[method] for n in (4,8,16)}}
 rep={'experiment_id':'MA-589','seed':seed,'split':split,'model_revision':REV,'model_sha256':MODEL_SHA,'dataset':ef,'dataset_sha256':DATA_SHA[ef],'prefix_tokens':PFX,'calibration_contexts':len(calw),'query_contexts':len(evalw),'pca_rank':R,'residual_rank':RR,'methods':result,'model_state_bytes':state_bytes,'model_state_files':{'shared_prefix_basis.npz':pca_b,'shared_residual_basis.npz':residual_b},'deployment_frontier':front,'compute':{'model_load_seconds':load_s,'calibration_prefill_seconds':prefill_s,'SVD_fit_seconds':fit_s,'total_wall_seconds':time.perf_counter()-t_all,'latent_encode_ops_per_token':L*F*R*2,'latent_decode_ops_per_cached_token':L*F*R*2,'residual_view_ops_per_cached_token':L*ROLE*D*RR*2,'optimizer_updates':0},'note':'Prefix is a text-derived Pythia KV cache, not a trained continuous Prefix-Tuning vector.'}
 (out/'metrics.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n');print(json.dumps(rep,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['dev','fresh'],required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.seed,a.split,a.model_dir,a.data_dir,a.out)
if __name__=='__main__':main()
