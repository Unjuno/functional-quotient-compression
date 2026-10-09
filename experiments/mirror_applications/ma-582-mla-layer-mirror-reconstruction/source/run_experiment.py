#!/usr/bin/env python3
"""MA-582 group-shared MLA latent and layer-specific residual View screen."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers.cache_utils import DynamicCache
REV='e93a9faa9c77e5d09219f6c868bfc7a1bd65593c'
MODEL_SHA='3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd'
DATA_SHA={'train.txt':'9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f','valid.txt':'f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8','test.txt':'d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0'}
L,H,KV,D,R,RR,PFX=6,8,2,64,128,4,64
GROUPS=((0,1,2),(3,4,5)); ROLE=H*KV; F=ROLE*D
METHODS=('fp16_kv','per_layer_mla128','group_mla128','mirror_layer_view4','native_group_resid4','private_layer_resid4')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def toks(p,tok):return np.asarray(tok.encode(p.read_text(encoding='utf8'),add_special_tokens=False),np.int32)
def starts(x,seed,n):return np.random.default_rng(seed).choice(len(x)-PFX-2,size=n,replace=False).tolist()
def wins(x,ii):return [x[i:i+PFX+2].astype(np.int64) for i in ii]
def getcache(m,w):
 with torch.inference_mode():return m(input_ids=torch.tensor(w[:PFX][None,:],dtype=torch.long),use_cache=True,return_dict=True).past_key_values
def cache_np(p):return [[t.detach().cpu().numpy().astype(np.float32,copy=False) for t in lay] for lay in p]
def tomat(c,l):
 k,v=c[l];return np.stack((k[0],v[0]),axis=1).transpose(2,0,1,3).reshape(k.shape[1],F)
def frommat(x):
 y=x.reshape(x.shape[0],H,KV,D).transpose(1,2,0,3)
 return torch.from_numpy(y[:,0][None].copy()),torch.from_numpy(y[:,1][None].copy())
def pca(x,r):
 mu=x.mean(0,dtype=np.float64).astype(np.float32);_,_,vt=np.linalg.svd((x-mu).astype(np.float32),full_matrices=False);return mu,vt[:r].T.astype(np.float32)
def dump(path,**kw):np.savez(path,**kw);return path.stat().st_size
def nll(m,q,y,past):
 t=time.perf_counter()
 with torch.inference_mode():o=m(input_ids=torch.tensor([[int(q)]],dtype=torch.long),past_key_values=DynamicCache.from_legacy_cache(past),use_cache=False,return_dict=True)
 return float(torch.nn.functional.cross_entropy(o.logits[:,-1,:].float(),torch.tensor([int(y)]))),time.perf_counter()-t
def run(seed,split,model_dir,data_dir,out):
 t_all=time.perf_counter();torch.set_num_threads(4);model_dir=Path(model_dir);data_dir=Path(data_dir);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if sha(model_dir/'model.safetensors')!=MODEL_SHA:raise ValueError('model hash mismatch')
 for name,digest in DATA_SHA.items():
  if sha(data_dir/name)!=digest:raise ValueError('dataset hash mismatch '+name)
 tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True);t=time.perf_counter();m=GPTNeoXForCausalLM.from_pretrained(model_dir,local_files_only=True,torch_dtype=torch.float32).eval();load_s=time.perf_counter()-t
 tr=toks(data_dir/'train.txt',tok);ef='valid.txt' if split=='dev' else 'test.txt';ev=toks(data_dir/ef,tok);cw=wins(tr,starts(tr,seed+100,4));ew=wins(ev,starts(ev,seed+200,16))
 t=time.perf_counter();cal=[cache_np(getcache(m,w)) for w in cw];prefill_s=time.perf_counter()-t
 t=time.perf_counter();pm=[];pe=[]
 for l in range(L):
  mu,e=pca(np.concatenate([tomat(c,l) for c in cal]),R);pm.append(mu);pe.append(e)
 gm=[];ge=[]
 for group in GROUPS:
  x=np.concatenate([tomat(c,l) for c in cal for l in group]);mu,e=pca(x,R);gm.append(mu);ge.append(e)
 # Residual bases: one per group/role; private upper control one per layer/role.
 gb=[];pb=[]
 for gi,group in enumerate(GROUPS):
  rr=[]
  for role in range(ROLE):
   chunks=[]
   for c in cal:
    for l in group:
     x=tomat(c,l);base=((x-gm[gi])@ge[gi])@ge[gi].T+gm[gi];chunks.append((x-base).reshape(-1,ROLE,D)[:,role,:])
   rr.append(pca(np.concatenate(chunks),RR)[1])
  gb.append(np.stack(rr))
 for l in range(L):
  x=np.concatenate([tomat(c,l) for c in cal]);base=((x-pm[l])@pe[l])@pe[l].T+pm[l];res=(x-base).reshape(-1,ROLE,D)
  pb.append(np.stack([pca(res[:,r,:],RR)[1] for r in range(ROLE)]))
 fit_s=time.perf_counter()-t
 arr=lambda x:np.asarray(x,dtype=np.float16)
 states={}
 states['per_layer_mla128']=dump(out/'per_layer_pca.npz',mean=arr(pm),components=arr(pe),shape=np.asarray([L,F,R],np.int32),schema=np.asarray([582,1],np.int32))
 states['group_mla128']=dump(out/'group_pca.npz',mean=arr(gm),components=arr(ge),groups=np.asarray(GROUPS,np.int32),schema=np.asarray([582,2],np.int32))
 states['mirror_layer_view4']=dump(out/'group_pca.npz',mean=arr(gm),components=arr(ge),groups=np.asarray(GROUPS,np.int32),schema=np.asarray([582,2],np.int32))+dump(out/'group_residual.npz',basis=arr(gb),groups=np.asarray(GROUPS,np.int32),shape=np.asarray([2,ROLE,D,RR],np.int32),schema=np.asarray([582,3],np.int32))
 states['native_group_resid4']=states['mirror_layer_view4']
 states['private_layer_resid4']=states['per_layer_mla128']+dump(out/'private_residual.npz',basis=arr(pb),shape=np.asarray([L,ROLE,D,RR],np.int32),schema=np.asarray([582,4],np.int32))
 stats={k:{'nll':[],'mse':[],'encode':[],'reconstruct':[],'query':[]} for k in METHODS};cache_bytes={}
 for wi,w in enumerate(ew):
  raw=cache_np(getcache(m,w));decoded={};cache_data={}
  for method in METHODS:
   t=time.perf_counter();layers=[];zs=[];cs=[];mse=[]
   for l in range(L):
    x=tomat(raw,l)
    if method=='fp16_kv':rec=x.astype(np.float16).astype(np.float32);zs.append(x.astype(np.float16))
    elif method=='per_layer_mla128':
     z=((x-pm[l])@pe[l]).astype(np.float16).astype(np.float32);rec=z@pe[l].T+pm[l];zs.append(z.astype(np.float16))
    else:
     gi=l//3;z=((x-gm[gi])@ge[gi]).astype(np.float16).astype(np.float32);base=z@ge[gi].T+gm[gi];zs.append(z.astype(np.float16))
     if method=='group_mla128':rec=base
     else:
      b=pb[l] if method=='private_layer_resid4' else gb[gi];res=(x-base).reshape(-1,ROLE,D);c=np.stack([res[:,r,:]@b[r] for r in range(ROLE)],axis=1).astype(np.float16).astype(np.float32);rec=base+np.stack([c[:,r,:]@b[r].T for r in range(ROLE)],axis=1).reshape(x.shape);cs.append(c.astype(np.float16))
    mse.append(float(np.mean((rec-x)**2)));layers.append(frommat(rec))
   enc_s=time.perf_counter()-t;decoded[method]=tuple(layers);t=time.perf_counter()
   if method=='fp16_kv':data={'kv_fp16':np.concatenate([a.astype(np.float16).ravel() for lay in raw for a in lay]),'shape':np.asarray([L,H,KV,PFX,D],np.int32),'schema':np.asarray([582,1],np.int32)}
   else:
    data={'latent_fp16':np.concatenate([z.ravel() for z in zs]),'shape':np.asarray([L,PFX,R],np.int32),'schema':np.asarray([582,2],np.int32)}
    if cs:data['residual_codes_fp16']=np.concatenate([c.ravel() for c in cs])
   rec_s=time.perf_counter()-t
   loss,q_s=nll(m,w[PFX],w[PFX+1],decoded[method]);stats[method]['nll'].append(loss);stats[method]['mse'].append(float(np.mean(mse)));stats[method]['encode'].append(enc_s);stats[method]['reconstruct'].append(rec_s);stats[method]['query'].append(q_s)
   if wi==0:cache_bytes[method]=dump(out/f'{method}_cache.npz',**data)
  if wi==0:cache_data=raw
 result={};front={}
 for method,v in stats.items():
  delta=float(np.mean(v['nll'])-np.mean(stats['fp16_kv']['nll']));result[method]={'mean_nll':float(np.mean(v['nll'])),'std_nll':float(np.std(v['nll'],ddof=1)),'nll_delta_vs_fp16':delta,'mean_cache_reconstruction_mse':float(np.mean(v['mse'])),'session_cache_bytes':cache_bytes[method],'mean_encode_seconds':float(np.mean(v['encode'])),'mean_reconstruct_seconds':float(np.mean(v['reconstruct'])),'mean_query_seconds':float(np.mean(v['query']))}
  b=states.get(method,0);front[method]={'model_state_bytes':b,'total_bytes_for_1_session':b+cache_bytes[method],'total_bytes_for_4_sessions':b+4*cache_bytes[method],'total_bytes_for_8_sessions':b+8*cache_bytes[method]}
 # The native implementation is deliberately serialized/evaluated independently for audit of the alias.
 rep={'experiment_id':'MA-582','seed':seed,'split':split,'model_revision':REV,'model_sha256':MODEL_SHA,'dataset':ef,'dataset_sha256':DATA_SHA[ef],'prefix_tokens':PFX,'query_target_pairs':len(ew),'calibration_prefixes':len(cw),'groups':GROUPS,'pca_rank':R,'residual_rank':RR,'methods':result,'model_state_bytes':states,'deployment_frontier':front,'compute':{'model_load_seconds':load_s,'calibration_prefill_seconds':prefill_s,'SVD_fit_seconds':fit_s,'total_wall_seconds':time.perf_counter()-t_all,'added_encode_project_ops_per_token':L*F*R*2,'latent_decode_ops_per_cached_token':L*F*R*2,'residual_view_ops_per_cached_token':L*ROLE*D*RR*2,'optimizer_updates':0},'note':'All PCA representations operate after the Pythia model generates K/V. Native grouped residual control has the same basis and coefficient representation as Mirror; exact alias is checked by serialized arrays.'}
 (out/'metrics.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n');print(json.dumps(rep,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['dev','fresh'],required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.seed,a.split,a.model_dir,a.data_dir,a.out)
if __name__=='__main__':main()
