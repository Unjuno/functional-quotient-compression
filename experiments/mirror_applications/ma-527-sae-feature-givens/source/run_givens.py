#!/usr/bin/env python3
"""Run frozen MA-527 feature-space Givens view experiment."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time, zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
HFILE=Path(__file__).with_name('shared_fv_harness.py')
spec=importlib.util.spec_from_file_location('ma527_harness',HFILE); h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
SAE_SHA='85b57dfe34d19769e2df0e208e38fda8815da8493c56b703a99ff74cc610dbf4'; SAE_REV='36c2027509efad69836aa4231999c9b717848ecd'
FIT=np.arange(12); HELD=np.arange(12,16); PAIRS=np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]])
class TiedSAE(torch.nn.Module):
 def __init__(self): super().__init__()
def load_sae(path):
 if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=SAE_SHA: raise ValueError('pinned SAE hash mismatch')
 torch.serialization.add_safe_globals([TiedSAE]); sae=torch.load(path,map_location='cpu',weights_only=True)
 W=sae.encoder.detach().cpu().numpy().astype(np.float32); bias=sae.encoder_bias.detach().cpu().numpy().astype(np.float32)
 if W.shape!=(2048,512) or bias.shape!=(2048,): raise ValueError('unexpected SAE shape')
 return W,bias
def givens(c,angles):
 """Block-diagonal orthogonal pair rotations; supports torch or numpy."""
 if torch.is_tensor(c):
  pieces=[]
  for j,(a,b) in enumerate(PAIRS):
   co=torch.cos(angles[...,j]); si=torch.sin(angles[...,j]); x=c[...,a];y=c[...,b]
   pieces.extend((co*x-si*y,si*x+co*y))
  return torch.stack(pieces,dim=-1)
 pieces=[]
 for j,(a,b) in enumerate(PAIRS):
  co=np.cos(angles[...,j]);si=np.sin(angles[...,j]);x=c[...,a];y=c[...,b]
  pieces.extend((co*x-si*y,si*x+co*y))
 return np.stack(pieces,axis=-1)
def apply_gains(c,gains):
 """Apply one independent scalar gain to each fixed coordinate pair."""
 if torch.is_tensor(c): return torch.stack([c[...,a:b+1]*gains[...,j,None] for j,(a,b) in enumerate(PAIRS)],dim=-2).flatten(-2)
 return np.stack([c[...,a:b+1]*gains[...,j,None] for j,(a,b) in enumerate(PAIRS)],axis=-2).reshape(*c.shape[:-1],16)
def decode(code,Wpool): return code @ Wpool
def fit_shared(targets,Wpool,mode,updates=2000):
 """Adam full-batch fit to task FVs. mode is givens, gains, or plain."""
 torch.manual_seed(0); dev='cpu'; tar=torch.tensor(targets,dtype=torch.float32); atoms=torch.tensor(Wpool,dtype=torch.float32)
 c=torch.nn.Parameter(torch.zeros(16)); angles=torch.nn.Parameter(torch.zeros((12,8))) if mode=='givens' else None
 gains=torch.nn.Parameter(torch.ones((12,8))) if mode=='gains' else None
 params=[c]+([angles] if angles is not None else [])+([gains] if gains is not None else [])
 opt=torch.optim.Adam(params,lr=.01,weight_decay=0.);t0=time.perf_counter()
 for _ in range(updates):
  opt.zero_grad()
  if mode=='givens': codes=givens(c.expand(12,-1),angles)
  elif mode=='gains': codes=apply_gains(c.expand(12,-1),gains)
  else: codes=c.expand(12,-1)
  pred=codes@atoms; loss=torch.mean((pred-tar[FIT])**2);loss.backward();opt.step()
 return c.detach().numpy(),(angles.detach().numpy() if angles is not None else gains.detach().numpy() if gains is not None else None),time.perf_counter()-t0
def fit_codes(targets,Wpool,c,mode,updates=500):
 """Independently assign held-out m (or equal-size gains) to FV targets."""
 torch.manual_seed(0);tar=torch.tensor(targets,dtype=torch.float32);atoms=torch.tensor(Wpool,dtype=torch.float32);base=torch.tensor(c,dtype=torch.float32)
 t0=time.perf_counter()
 if mode=='givens': p=torch.nn.Parameter(torch.zeros((4,8)))
 elif mode=='gains': p=torch.nn.Parameter(torch.ones((4,8)))
 else: return np.zeros((4,8),dtype=np.float32),np.tile((base.numpy()@Wpool),(4,1)),0.
 opt=torch.optim.Adam([p],lr=.01,weight_decay=0.);t0=time.perf_counter()
 for _ in range(updates):
  opt.zero_grad()
  if mode=='givens': codes=givens(base.expand(4,-1),p)
  else: codes=apply_gains(base.expand(4,-1),p)
  loss=torch.mean((codes@atoms-tar[HELD])**2);loss.backward();opt.step()
 dec=(codes@atoms).detach().numpy()
 return p.detach().numpy(),dec,time.perf_counter()-t0

def omp16(vectors,W):
 norms=np.linalg.norm(W.astype(np.float64),axis=1); D=W/np.maximum(norms[:,None],1e-12)
 out=np.zeros_like(vectors,dtype=np.float32)
 for i,v in enumerate(vectors):
  r=v.copy();used=np.zeros(len(W),bool)
  for _ in range(16):
   cor=D@r;cor[used]=0;j=int(np.argmax(np.abs(cor))); coef=float(cor[j]/norms[j]);out[i]+=coef*W[j];r-=coef*W[j];used[j]=1
 return out
def save(path,arrays):
 np.savez(path,**arrays)
 with zipfile.ZipFile(path) as z:
  if any(x.compress_type!=zipfile.ZIP_STORED for x in z.infolist()):raise AssertionError('compressed NPZ not allowed')
 return Path(path).stat().st_size
def run(seed,model_dir,sae_path,out_dir):
 out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);print('stage=load',flush=True);model,tok,torchmod=h.load_model(model_dir)
 print('stage=extract',flush=True)
 start=time.perf_counter();vectors,manifest,support=h.extract_task_vectors(model,tok,torchmod,seed);extract_s=time.perf_counter()-start
 print('stage=load_sae',flush=True)
 pool_start=time.perf_counter();W,bias=load_sae(sae_path); acts=np.maximum(vectors@W.T+bias,0);scores=acts[FIT].sum(0);pool=np.argsort(-scores,kind='stable')[:16].astype(np.int16);Wp=W[pool];pool_s=time.perf_counter()-pool_start
 print('stage=fit',flush=True)
 print('fit=givens',flush=True);c,m,train_s=fit_shared(vectors,Wp,'givens'); print('fit=givens_done',flush=True);cg,mg,gain_train_s=fit_shared(vectors,Wp,'gains'); print('fit=gains_done',flush=True);cp,_,plain_train_s=fit_shared(vectors,Wp,'plain');print('fit=plain_done',flush=True)
 print('fit=heldout_givens',flush=True);mheld,dec_h,givens_assign_s=fit_codes(vectors,Wp,c,'givens');print('fit=heldout_gains',flush=True);gheld,dec_g,gain_assign_s=fit_codes(vectors,Wp,cg,'gains')
 d_plain=np.tile(cp@Wp,(16,1)); d_h=np.tile(c@Wp,(16,1));d_g=np.tile(cg@Wp,(16,1));d_h[HELD]=dec_h;d_g[HELD]=dec_g
 print('fit=omp_decode',flush=True);omp_start=time.perf_counter();d_omp=omp16(vectors,W);print('fit=omp_done',flush=True);omp_s=time.perf_counter()-omp_start;d_exp=vectors.copy();d_none=np.zeros_like(vectors)
 model_bytes=sum((Path(model_dir)/n).stat().st_size for n in ['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']);sae_bytes=Path(sae_path).stat().st_size
 meta={'task_ids':np.arange(16,dtype=np.int16),'model_revision':np.frombuffer(h.REVISION.encode(),dtype='S40'),'model_sha256':np.frombuffer(h.MODEL_SHA.encode(),dtype='S64'),'sae_revision':np.frombuffer(SAE_REV.encode(),dtype='S40'),'sae_sha256':np.frombuffer(SAE_SHA.encode(),dtype='S64'),'schema':np.array([527,2048,16,8],np.int32)}
 payloads={}
 payloads['explicit_fv']=save(out/'explicit_fv.npz',dict(function_vectors=vectors.astype(np.float32),**meta))
 payloads['shared_givens']=save(out/'shared_givens.npz',dict(pool_feature_ids=pool,shared_code=c.astype(np.float32),heldout_angles=mheld.astype(np.float32),fit_angles=m.astype(np.float32),**meta))
 payloads['native_pairwise_gains']=save(out/'native_gains.npz',dict(pool_feature_ids=pool,shared_code=cg.astype(np.float32),heldout_gains=gheld.astype(np.float32),fit_gains=mg.astype(np.float32),**meta))
 payloads['shared_sae_code']=save(out/'shared_code.npz',dict(pool_feature_ids=pool,shared_code=cp.astype(np.float32),**meta))
 # Global OMP indices/coefs are serialized as inference state, not decoded residuals.
 norms=np.linalg.norm(W.astype(np.float64),axis=1);D=W/np.maximum(norms[:,None],1e-12); ix=np.zeros((16,16),np.int16); val=np.zeros((16,16),np.float32)
 for i,v in enumerate(vectors):
  r=v.copy();used=np.zeros(2048,bool)
  for j in range(16):
   cc=D@r;cc[used]=0;q=int(np.argmax(abs(cc)));z=float(cc[q]/norms[q]);ix[i,j]=q;val[i,j]=z;r-=z*W[q];used[q]=1
 payloads['global_omp16']=save(out/'global_omp16.npz',dict(feature_ids=ix,coefficients=val,**meta))
 methods=[('none',d_none,0),('explicit_fv',d_exp,payloads['explicit_fv']),('shared_sae_code',d_plain,payloads['shared_sae_code']),('native_pairwise_gains',d_g,payloads['native_pairwise_gains']),('shared_givens',d_h,payloads['shared_givens']),('global_omp16',d_omp,payloads['global_omp16'])]
 rows=[]
 coding_by_method={'none':0.,'explicit_fv':0.,'shared_sae_code':pool_s+plain_train_s,'native_pairwise_gains':pool_s+gain_train_s+gain_assign_s,'shared_givens':pool_s+train_s+givens_assign_s,'global_omp16':pool_s+omp_s}
 updates_by_method={'none':0,'explicit_fv':0,'shared_sae_code':2000,'native_pairwise_gains':4000,'shared_givens':4000,'global_omp16':0}
 for name,decoded,pbytes in methods:
  print('stage=evaluate:'+name,flush=True)
  t=time.perf_counter();metrics=h.evaluate(model,tok,torchmod,vectors,decoded,manifest);inf=time.perf_counter()-t;uses=name not in ('none','explicit_fv');
  sae_fit_ops={'shared_sae_code':2000*12*16*512,'native_pairwise_gains':2000*12*16*512+500*4*16*512,'shared_givens':2000*12*16*512+500*4*16*512,'global_omp16':16*16*2048*512}.get(name,0)
  pool_ops=12*512*2048 if uses else 0
  rows.append({'seed':seed,'method':name,'payload_bytes':pbytes,'model_bytes':model_bytes,'sae_bytes':sae_bytes if uses else 0,'incremental_over_preloaded_sae_bytes':pbytes,'total_deployment_bytes':model_bytes+(sae_bytes if uses else 0)+pbytes,'explicit_fv_total_bytes':model_bytes+payloads['explicit_fv'],'support_examples':128,'support_forward_calls':support['forward_calls'],'support_input_tokens':support['input_tokens'],'candidate_input_tokens':metrics['candidate_input_tokens'],'optimizer_updates':updates_by_method[name],'extraction_seconds':extract_s,'coding_seconds':coding_by_method[name],'inference_seconds':inf,'wall_time_seconds':extract_s+coding_by_method[name]+inf,'active_compute_proxy':int(sae_fit_ops+pool_ops+(metrics['candidate_sequences']*512 if uses else 0)),'metrics':metrics})
 np.savez(out/'pool_audit.npz',pool_ids=pool,pool_scores=scores)
 (out/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 report={'experiment_id':'MA-527','seed':seed,'model_revision':h.REVISION,'model_sha256':h.MODEL_SHA,'sae_revision':SAE_REV,'sae_sha256':SAE_SHA,'model_bytes':model_bytes,'sae_bytes':sae_bytes,'pool_ids':pool.tolist(),'pool_selection_fit_tasks':FIT.tolist(),'heldout_tasks':HELD.tolist(),'fit_seconds':{'pool_selection_including_sae_encoding':pool_s,'givens_shared_fit':train_s,'native_gains_shared_fit':gain_train_s,'shared_code_fit':plain_train_s,'givens_heldout_angle_assignment':givens_assign_s,'gains_heldout_assignment':gain_assign_s,'global_omp16':omp_s},'heldout_angle_updates':500,'rows':rows}
 (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'seed':seed,'pool':pool.tolist(),'rows':[{'method':r['method'],'bytes':r['payload_bytes'],'accuracy':r['metrics']['heldout_accuracy'],'gold_logprob':r['metrics']['mean_gold_candidate_logprob']} for r in rows]},indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--model-dir',required=True);p.add_argument('--sae',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.seed,a.model_dir,a.sae,a.out)
if __name__=='__main__':main()
