#!/usr/bin/env python3
"""MA-581 shared MLA latent plus Mirror head-residual-code screen."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,GPTNeoXForCausalLM
from transformers.cache_utils import DynamicCache
REV='e93a9faa9c77e5d09219f6c868bfc7a1bd65593c';MODEL_SHA='3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd'
DATA_SHA={'train.txt':'9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f','valid.txt':'f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8','test.txt':'d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0'}
L=6;H=8;KV=2;D=64;F=H*KV*D;R=128;RR=4;PFX=64

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load_tokens(p,tok):return np.asarray(tok.encode(p.read_text(encoding='utf8'),add_special_tokens=False),np.int32)
def starts(x,seed,n):return np.random.default_rng(seed).choice(len(x)-PFX-2,size=n,replace=False).tolist()
def windows(x,inds):return [x[i:i+PFX+2].astype(np.int64) for i in inds]
def get_cache(model,w):
 inp=torch.tensor(w[:PFX][None,:],dtype=torch.long)
 with torch.inference_mode():return model(input_ids=inp,use_cache=True,return_dict=True).past_key_values
def cache_np(past):return [[t.detach().cpu().numpy().astype(np.float32,copy=False) for t in layer] for layer in past]
def to_matrix(cache,layer):
 k=cache[layer][0][0];v=cache[layer][1][0]
 return np.stack((k,v),axis=1).transpose(2,0,1,3).reshape(k.shape[1],F)
def from_matrix(x):
 y=x.reshape(x.shape[0],H,KV,D).transpose(1,2,0,3)
 return torch.from_numpy(y[:,0][None].copy()),torch.from_numpy(y[:,1][None].copy())
def pca(x,r):
 mu=x.mean(axis=0,dtype=np.float64).astype(np.float32);_,_,vt=np.linalg.svd((x-mu).astype(np.float32),full_matrices=False);e=vt[:r].T.astype(np.float32);return mu,e
def fit_bases(cal_caches):
 means=[];enc=[];shared=[];private=[]
 for l in range(L):
  x=np.concatenate([to_matrix(c,l) for c in cal_caches],axis=0);mu,e=pca(x,R);base=(x-mu)@e@e.T+mu;res=(x-base).reshape(-1,H*KV,D)
  rows=res.reshape(-1,D);_,b=pca(rows,RR);shared.append(b)
  bp=[]
  for role in range(H*KV):
   _,q=pca(res[:,role,:],RR);bp.append(q)
  means.append(mu);enc.append(e);private.append(np.stack(bp))
 return ([x.astype(np.float16).astype(np.float32) for x in means],[x.astype(np.float16).astype(np.float32) for x in enc],[x.astype(np.float16).astype(np.float32) for x in shared],[x.astype(np.float16).astype(np.float32) for x in private])
def nll(model,query,target,past):
 t=time.perf_counter()
 with torch.inference_mode():o=model(input_ids=torch.tensor([[int(query)]],dtype=torch.long),past_key_values=DynamicCache.from_legacy_cache(past),use_cache=False,return_dict=True)
 sec=time.perf_counter()-t;loss=float(torch.nn.functional.cross_entropy(o.logits[:,-1,:].float(),torch.tensor([int(target)])));return loss,sec
def serialize(path,**kw):np.savez(path,**kw);return path.stat().st_size
def run(seed,split,model_dir,data_dir,out):
 wall=time.perf_counter();torch.set_num_threads(4);model_dir=Path(model_dir);data_dir=Path(data_dir);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if sha(model_dir/'model.safetensors')!=MODEL_SHA:raise ValueError('model hash mismatch')
 for n,h in DATA_SHA.items():
  if sha(data_dir/n)!=h:raise ValueError('dataset hash mismatch: '+n)
 tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True);tm=time.perf_counter();model=GPTNeoXForCausalLM.from_pretrained(model_dir,local_files_only=True,torch_dtype=torch.float32).eval();load_s=time.perf_counter()-tm
 train=load_tokens(data_dir/'train.txt',tok);ef='valid.txt' if split=='dev' else 'test.txt';ev=load_tokens(data_dir/ef,tok)
 cwin=windows(train,starts(train,seed+100,4));ewin=windows(ev,starts(ev,seed+200,16))
 t=time.perf_counter();cal=[cache_np(get_cache(model,w)) for w in cwin];prefill_s=time.perf_counter()-t
 t=time.perf_counter();means,enc,shared,private=fit_bases(cal);svd_s=time.perf_counter()-t
 pca_state=out/'mla_pca_state.npz';pca_bytes=serialize(pca_state,pca_mean_fp16=np.stack(means).astype(np.float16),pca_components_fp16=np.stack(enc).astype(np.float16),shape=np.asarray([L,H,KV,D,R],np.int32),schema=np.array([581,1],np.int32))
 shared_state=out/'shared_residual_dictionary.npz';shared_bytes=serialize(shared_state,basis_fp16=np.stack(shared).astype(np.float16),shape=np.asarray([L,D,RR],np.int32),schema=np.array([581,2],np.int32))
 private_state=out/'private_head_residual_bases.npz';private_bytes=serialize(private_state,bases_fp16=np.stack(private).astype(np.float16),shape=np.asarray([L,H*KV,D,RR],np.int32),schema=np.array([581,3],np.int32))
 basis_bytes={'fp16_kv':0,'mla_rank128':pca_bytes,'mirror_shared_resid4':pca_bytes+shared_bytes,'native_shared_resid4':pca_bytes+shared_bytes,'private_head_resid4':pca_bytes+private_bytes}
 methods={k:{'nll':[],'rec_mse':[],'encode_s':[],'rebuild_s':[],'query_s':[]} for k in ['fp16_kv','mla_rank128','mirror_shared_resid4','native_shared_resid4','private_head_resid4']};sizes={};sample=None
 for wi,w in enumerate(ewin):
  past=get_cache(model,w);raw=cache_np(past);newpast={};session_states={}
  for method in methods:
   t=time.perf_counter();layers=[];zs=[];cs=[];mse=[]
   for l in range(L):
    x=to_matrix(raw,l);mu=means[l];e=enc[l];z=((x-mu)@e).astype(np.float16).astype(np.float32);base=z@e.T+mu
    if method=='fp16_kv':rec=x.astype(np.float16).astype(np.float32);zs.append(x.astype(np.float16));
    elif method=='mla_rank128':rec=base;zs.append(z.astype(np.float16))
    elif method in ('mirror_shared_resid4','native_shared_resid4'):
     residual=(x-base).reshape(-1,H*KV,D);c=np.einsum('tnd,dr->tnr',residual,shared[l]).astype(np.float16).astype(np.float32);rec=base+np.einsum('tnr,dr->tnd',c,shared[l]).reshape(x.shape);zs.append(z.astype(np.float16));cs.append(c.astype(np.float16))
    else:
     residual=(x-base).reshape(-1,H*KV,D);c=np.stack([residual[:,j]@private[l][j] for j in range(H*KV)],1).astype(np.float16).astype(np.float32);rec=base+np.stack([c[:,j]@private[l][j].T for j in range(H*KV)],1).reshape(x.shape);zs.append(z.astype(np.float16));cs.append(c.astype(np.float16))
    mse.append(float(np.mean((rec-x)**2)));layers.append(from_matrix(rec))
   encs=time.perf_counter()-t;newpast[method]=tuple(layers);mseval=float(np.mean(mse));t=time.perf_counter()
   if method=='fp16_kv':
    state={'kv_fp16':np.concatenate([x.detach().cpu().numpy().astype(np.float16).ravel() for layer in past for x in layer]),'shape':np.asarray([L,H,KV,PFX,D],np.int32),'schema':np.array([581,2],np.int32)}
   elif method=='mla_rank128':state={'latent_fp16':np.concatenate([z.ravel() for z in zs]),'shape':np.asarray([L,PFX,R],np.int32),'schema':np.array([581,3],np.int32)}
   else:state={'latent_fp16':np.concatenate([z.ravel() for z in zs]),'head_residual_codes_fp16':np.concatenate([c.ravel() for c in cs]),'shape':np.asarray([L,PFX,H,KV,R,RR],np.int32),'schema':np.array([581,4],np.int32)}
   rebuild_s=time.perf_counter()-t;loss,query_s=nll(model,w[PFX],w[PFX+1],newpast[method]);methods[method]['nll'].append(loss);methods[method]['rec_mse'].append(mseval);methods[method]['encode_s'].append(encs);methods[method]['rebuild_s'].append(rebuild_s);methods[method]['query_s'].append(query_s)
   if wi==0:sizes[method]=serialize(out/f'{method}_session_cache.npz',**state)
  if wi==0:sample=raw
 res={}
 for method,v in methods.items():res[method]={'mean_nll':float(np.mean(v['nll'])),'std_nll':float(np.std(v['nll'],ddof=1)),'nll_delta_vs_fp16':float(np.mean(v['nll'])-np.mean(methods['fp16_kv']['nll'])),'mean_cache_reconstruction_mse':float(np.mean(v['rec_mse'])),'session_cache_bytes':sizes[method],'mean_encode_seconds':float(np.mean(v['encode_s'])),'mean_reconstruct_seconds':float(np.mean(v['rebuild_s'])),'mean_query_seconds':float(np.mean(v['query_s']))}
 deployment={}
 for m,v in res.items():
  state_bytes=basis_bytes[m]
  deployment[m]={'model_state_bytes':state_bytes,'total_bytes_for_1_session':state_bytes+v['session_cache_bytes'],'total_bytes_for_4_sessions':state_bytes+4*v['session_cache_bytes'],'total_bytes_for_8_sessions':state_bytes+8*v['session_cache_bytes']}
 rep={'experiment_id':'MA-581','seed':seed,'split':split,'model_revision':REV,'model_sha256':MODEL_SHA,'dataset':ef,'dataset_sha256':DATA_SHA[ef],'prefix_tokens':PFX,'query_target_pairs':len(ewin),'calibration_prefixes':len(cwin),'pca_rank':R,'residual_rank':RR,'methods':res,'model_state_bytes':basis_bytes,'model_state_files':{'mla_pca_state.npz':pca_bytes,'shared_residual_dictionary.npz':shared_bytes,'private_head_residual_bases.npz':private_bytes},'deployment_frontier':deployment,'compute':{'model_load_seconds':load_s,'calibration_prefill_seconds':prefill_s,'SVD_fit_seconds':svd_s,'total_wall_seconds':time.perf_counter()-wall,'added_encode_project_ops_per_token':L*F*R*2,'latent_decode_ops_per_cached_token':L*F*R*2,'optimizer_updates':0},'note':'PCA compresses K/V states after the original model generated them; projection/reconstruction bases are charged as added model state.'}
 (out/'metrics.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n');print(json.dumps({'seed':seed,'methods':res,'basis_bytes':basis_bytes,'deployment':deployment},indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['dev','fresh'],required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.seed,a.split,a.model_dir,a.data_dir,a.out)
if __name__=='__main__':main()
