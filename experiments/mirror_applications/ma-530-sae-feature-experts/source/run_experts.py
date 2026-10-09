#!/usr/bin/env python3
"""Run frozen MA-530 support-trained routed SAE feature expert screen."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];HF=Path(__file__).with_name('shared_fv_harness.py')
spec=importlib.util.spec_from_file_location('ma530_harness',HF);h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
SAE_SHA='85b57dfe34d19769e2df0e208e38fda8815da8493c56b703a99ff74cc610dbf4';SAE_REV='36c2027509efad69836aa4231999c9b717848ecd';POOL=64;K=16;TASKS=16
class TiedSAE(torch.nn.Module):
 def __init__(self):super().__init__()
def load_sae(path):
 if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=SAE_SHA:raise ValueError('SAE hash mismatch')
 torch.serialization.add_safe_globals([TiedSAE]);sae=torch.load(path,map_location='cpu',weights_only=True)
 W=sae.encoder.detach().cpu().numpy().astype(np.float32);bias=sae.encoder_bias.detach().cpu().numpy().astype(np.float32)
 if W.shape!=(2048,512) or bias.shape!=(2048,):raise ValueError('unexpected SAE shape')
 return W,bias
def residual_pool(vectors,W):
 norms=np.linalg.norm(W.astype(np.float64),axis=1);D=W/np.maximum(norms[:,None],1e-12);res=vectors.copy();used=np.zeros(len(W),bool);pool=[]
 for _ in range(POOL):
  c=np.abs(res@D.T).sum(0);c[used]=-np.inf;j=int(np.argmax(c));pool.append(j);used[j]=1
  a=(res@D[j])/max(norms[j],1e-12);res-=a[:,None]*W[j]
 return np.asarray(pool,np.int16)
def omp(vectors,atoms,k=K):
 norms=np.linalg.norm(atoms.astype(np.float64),axis=1);D=atoms/np.maximum(norms[:,None],1e-12);ids=np.zeros((len(vectors),k),np.int16);coef=np.zeros((len(vectors),k),np.float32);out=np.zeros_like(vectors,np.float32)
 for i,v in enumerate(vectors.astype(np.float32)):
  r=v.copy();used=np.zeros(len(atoms),bool)
  for j in range(k):
   c=D@r;c[used]=0;q=int(np.argmax(np.abs(c)));z=float(c[q]/norms[q]);ids[i,j]=q;coef[i,j]=z;out[i]+=z*atoms[q];r-=z*atoms[q];used[q]=1
 return ids,coef,out
def route_features(X,router):
 mu,sd,w,b=router;z=np.clip((X-mu)/sd,-10,10);logits=z@w.T+b
 return np.argmax(logits,axis=1).astype(np.int16),logits
def train_router(model,tok,torchmod,manifest):
 X=[];Y=[];n_tokens=0;calls=0;t=time.perf_counter()
 for tid,row in enumerate(manifest):
  for x,_ in row['support']:
   act,n=h.get_activation(model,tok,torchmod,h.prompt(x));X.append(act);Y.append(tid);n_tokens+=n;calls+=1
 X=np.stack(X).astype(np.float32);Y=np.asarray(Y,np.int64);mu=X.mean(0);sd=X.std(0)+1e-5;Z=np.clip((X-mu)/sd,-10,10)
 torchmod.manual_seed(0);zt=torch.tensor(Z);yt=torch.tensor(Y);w=torch.nn.Parameter(torch.zeros((TASKS,512)));b=torch.nn.Parameter(torch.zeros(TASKS));opt=torch.optim.Adam([w,b],lr=.01,weight_decay=0.)
 for _ in range(1000):
  opt.zero_grad();loss=torch.nn.functional.cross_entropy(zt@w.T+b,yt);loss.backward();opt.step()
 fit_s=time.perf_counter()-t
 return (mu.astype(np.float32),sd.astype(np.float32),w.detach().numpy().astype(np.float32),b.detach().numpy().astype(np.float32)),{'fit_seconds':fit_s,'fit_updates':1000,'support_examples':len(Y),'support_input_tokens':n_tokens,'support_forward_calls':calls,'fit_compute_proxy':1000*len(Y)*512*TASKS*2}
def encode_inputs(model,tok,torchmod,manifest,router):
 mu,sd,w,b=router;features=[];labels=[];toks=0;calls=0;t=time.perf_counter()
 for tid,row in enumerate(manifest):
  for x,y in row['evaluation']:
   act,n=h.get_activation(model,tok,torchmod,h.prompt(x));features.append(act);labels.append((tid,x,y));toks+=n;calls+=1
 X=np.stack(features).astype(np.float32);pred,_=route_features(X,router)
 return labels,pred.astype(np.int16),{'seconds':time.perf_counter()-t,'input_tokens':toks,'forward_calls':calls}
def evaluate(model,tok,torchmod,manifest,labels,preds,expert_vectors,route_mode):
 bytask=[];gold=[];margins=[];correct=0;counts=np.zeros((TASKS,TASKS),np.int64);seqs=0;tokens=0;route_correct=0
 for ix,(tid,x,y) in enumerate(labels):
  route=tid if route_mode=='oracle' else int(preds[ix]);counts[tid,route]+=1;route_correct+=int(route==tid)
  q=expert_vectors[route];queries=sorted({ans for _,ans in manifest[tid]['evaluation']})
  sc,nc,nt=h.score_candidates(model,tok,torchmod,x,queries,q);gi=queries.index(y);correct+=int(int(np.argmax(sc))==gi);gold.append(float(sc[gi]));margins.append(float(sc[gi]-np.max(np.delete(sc,gi))));seqs+=nc;tokens+=nt
 # Per-task candidate details are omitted here; avoid rescoring examples and inflating measured runtime.
 bytask=[{'task_id':i,'name':manifest[i]['name'],'queries':8} for i in range(TASKS)]
 return {'heldout_accuracy':correct/len(labels),'mean_gold_candidate_logprob':float(np.mean(gold)),'mean_gold_vs_best_negative_margin':float(np.mean(margins)),'queries':len(labels),'route_accuracy':route_correct/len(labels),'candidate_forward_calls':seqs,'candidate_sequences':seqs,'candidate_input_tokens':tokens,'route_confusion_matrix':counts.tolist(),'by_task':bytask}
def save(path,arrays):
 np.savez(path,**arrays)
 with zipfile.ZipFile(path) as z:
  if any(i.compress_type!=zipfile.ZIP_STORED for i in z.infolist()):raise AssertionError('must use uncompressed NPZ')
 return Path(path).stat().st_size
def run(seed,model_dir,sae_path,out_dir):
 out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);model,tok,torchmod=h.load_model(model_dir);t=time.perf_counter();vectors,manifest,task_support=h.extract_task_vectors(model,tok,torchmod,seed);fv_s=time.perf_counter()-t
 print('stage=router',flush=True);router,router_audit=train_router(model,tok,torchmod,manifest);labels,preds,route_audit=encode_inputs(model,tok,torchmod,manifest,router)
 W,bias=load_sae(sae_path);t=time.perf_counter();pool=residual_pool(vectors,W);pool_s=time.perf_counter()-t
 t=time.perf_counter();pi,pc,pdec=omp(vectors,W[pool]);pool_code_s=time.perf_counter()-t
 t=time.perf_counter();gi,gc,gdec=omp(vectors,W);global_code_s=time.perf_counter()-t
 model_bytes=sum((Path(model_dir)/n).stat().st_size for n in ['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']);sae_bytes=Path(sae_path).stat().st_size
 rmeta={'router_mean':router[0],'router_std':router[1],'router_weight':router[2],'router_bias':router[3],'model_revision':np.frombuffer(h.REVISION.encode(),dtype='S40'),'model_sha256':np.frombuffer(h.MODEL_SHA.encode(),dtype='S64'),'sae_revision':np.frombuffer(SAE_REV.encode(),dtype='S40'),'sae_sha256':np.frombuffer(SAE_SHA.encode(),dtype='S64'),'task_ids':np.arange(TASKS,dtype=np.int16),'schema':np.array([530,TASKS,POOL,K],np.int32)}
 payload={};payload['explicit_fv']=save(out/'learned_router_explicit_fv.npz',dict(function_vectors=vectors.astype(np.float32),**rmeta))
 payload['global_omp16']=save(out/'learned_router_global_omp16.npz',dict(feature_ids=gi.astype(np.int16),coefficients=gc,**rmeta))
 payload['shared_pool64']=save(out/'learned_router_shared_pool64.npz',dict(pool_feature_ids=pool,local_code_indices=pi.astype(np.uint8),coefficients=pc,**rmeta))
 oracle_bytes=save(out/'oracle_routed_explicit_fv.npz',dict(function_vectors=vectors.astype(np.float32),task_ids=np.arange(TASKS,dtype=np.int16),model_revision=rmeta['model_revision'],model_sha256=rmeta['model_sha256'],schema=rmeta['schema']))
 meanvec=np.tile(vectors.mean(0),(TASKS,1));none=np.zeros_like(vectors)
 mean_bytes=save(out/'shared_mean.npz',dict(mean_vector=vectors.mean(0).astype(np.float32),model_revision=rmeta['model_revision'],model_sha256=rmeta['model_sha256'],schema=np.array([530,512],np.int32)))
 methods=[('none',none,0,'oracle',False),('shared_mean',meanvec,mean_bytes,'oracle',False),('oracle_explicit_fv',vectors,oracle_bytes,'oracle',False),('learned_router_explicit_fv',vectors,payload['explicit_fv'],'learned',False),('learned_router_global_omp16',gdec,payload['global_omp16'],'learned',True),('learned_router_shared_pool64_omp16',pdec,payload['shared_pool64'],'learned',True)]
 rows=[]
 for name,decoded,pbytes,mode,uses_sae in methods:
  print('stage=evaluate:'+name,flush=True);t=time.perf_counter();m=evaluate(model,tok,torchmod,manifest,labels,preds,decoded,mode);infer_s=time.perf_counter()-t
  router_bytes=payload['explicit_fv']-save_bytes_fv(vectors,rmeta) if mode=='learned' else 0
  expert_bytes=pbytes-router_bytes if mode=='learned' else pbytes
  common_router_s=router_audit['fit_seconds']+route_audit['seconds'] if mode=='learned' else 0
  coding_s=(pool_s+pool_code_s if name=='learned_router_shared_pool64_omp16' else global_code_s if name=='learned_router_global_omp16' else 0)
  total=model_bytes+(sae_bytes if uses_sae else 0)+pbytes
  rows.append({'seed':seed,'method':name,'payload_bytes':pbytes,'router_bytes':router_bytes,'expert_bytes':expert_bytes,'model_bytes':model_bytes,'sae_bytes':sae_bytes if uses_sae else 0,'total_deployment_bytes':total,'explicit_fv_total_bytes':model_bytes+payload['explicit_fv'],'global_omp16_total_bytes':model_bytes+sae_bytes+payload['global_omp16'],'support_examples':task_support['forward_calls']//2,'support_forward_calls':task_support['forward_calls']+router_audit['support_forward_calls'],'support_input_tokens':task_support['input_tokens']+router_audit['support_input_tokens'],'query_router_tokens':route_audit['input_tokens'],'candidate_input_tokens':m['candidate_input_tokens'],'optimizer_updates':1000 if mode=='learned' else 0,'active_compute_proxy':int((router_audit['fit_compute_proxy'] if mode=='learned' else 0)+(m['queries']*512*TASKS if mode=='learned' else 0)+(16*K*512*(POOL if name=='learned_router_shared_pool64_omp16' else 2048) if uses_sae else 0)+m['candidate_sequences']*512 if name not in ('none','shared_mean') else 0),'extraction_seconds':fv_s,'router_fit_seconds':router_audit['fit_seconds'] if mode=='learned' else 0.,'expert_coding_seconds':coding_s,'router_query_seconds':route_audit['seconds'] if mode=='learned' else 0.,'inference_seconds':infer_s,'wall_time_seconds':fv_s+common_router_s+coding_s+infer_s,'metrics':m})
 np.savez(out/'router_audit.npz',predicted_experts=preds,labels=np.asarray([x[0] for x in labels],np.int16))
 np.savez(out/'pool_audit.npz',pool_ids=pool)
 (out/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 report={'experiment_id':'MA-530','seed':seed,'model_revision':h.REVISION,'model_sha256':h.MODEL_SHA,'sae_revision':SAE_REV,'sae_sha256':SAE_SHA,'model_bytes':model_bytes,'sae_bytes':sae_bytes,'route_fit':router_audit,'query_routing':route_audit,'pool_ids':pool.tolist(),'pool_selection_seconds':pool_s,'shared_omp_seconds':pool_code_s,'global_omp_seconds':global_code_s,'rows':rows}
 (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'seed':seed,'route_accuracy':m['route_accuracy'],'rows':[{'method':r['method'],'payload_bytes':r['payload_bytes'],'route_accuracy':r['metrics']['route_accuracy'],'accuracy':r['metrics']['heldout_accuracy'],'gold':r['metrics']['mean_gold_candidate_logprob']} for r in rows]},indent=2))
def save_bytes_fv(vectors,rmeta):
 # Serialized FV control's task vectors only, excluding router state.
 import tempfile,os
 fd,path=tempfile.mkstemp(suffix='.npz');os.close(fd)
 try:return save(path,dict(function_vectors=vectors.astype(np.float32),model_revision=rmeta['model_revision'],model_sha256=rmeta['model_sha256'],sae_revision=rmeta['sae_revision'],sae_sha256=rmeta['sae_sha256'],task_ids=rmeta['task_ids'],schema=rmeta['schema']))
 finally:Path(path).unlink(missing_ok=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--model-dir',required=True);p.add_argument('--sae',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.seed,a.model_dir,a.sae,a.out)
if __name__=='__main__':main()
