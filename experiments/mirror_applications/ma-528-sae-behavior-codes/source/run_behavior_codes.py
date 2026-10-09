#!/usr/bin/env python3
"""Run frozen MA-528 SAE shared-basis behavior-code screen."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];HF=Path(__file__).with_name('shared_fv_harness.py')
spec=importlib.util.spec_from_file_location('ma528_harness',HF);h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
SAE_SHA='85b57dfe34d19769e2df0e208e38fda8815da8493c56b703a99ff74cc610dbf4';SAE_REV='36c2027509efad69836aa4231999c9b717848ecd';FIT=np.arange(12);HELD=np.arange(12,16);POOL=64;K=16
class TiedSAE(torch.nn.Module):
 def __init__(self):super().__init__()
def load_sae(path):
 if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=SAE_SHA:raise ValueError('pinned SAE hash mismatch')
 torch.serialization.add_safe_globals([TiedSAE]);sae=torch.load(path,map_location='cpu',weights_only=True)
 W=sae.encoder.detach().cpu().numpy().astype(np.float32);bias=sae.encoder_bias.detach().cpu().numpy().astype(np.float32)
 if W.shape!=(2048,512) or bias.shape!=(2048,):raise ValueError('SAE shape mismatch')
 return W,bias
def aggregate_pool(vectors,W,bias):
 z=np.maximum(vectors[FIT]@W.T+bias,0);return np.argsort(-z.sum(0),kind='stable')[:POOL].astype(np.int16)
def residual_pool(vectors,W):
 norms=np.linalg.norm(W.astype(np.float64),axis=1);D=W/np.maximum(norms[:,None],1e-12);res=vectors[FIT].copy();selected=[];used=np.zeros(len(W),bool)
 for _ in range(POOL):
  # Aggregate absolute residual alignment; stable argmax favors smallest feature ID.
  scores=np.abs(res@D.T).sum(0);scores[used]=-np.inf;j=int(np.argmax(scores));selected.append(j);used[j]=True
  coeff=(res@D[j])/max(float(norms[j]),1e-12);res-=coeff[:,None]*W[j]
 return np.asarray(selected,dtype=np.int16)
def omp(vectors,atoms,k=K):
 norms=np.linalg.norm(atoms.astype(np.float64),axis=1);D=atoms/np.maximum(norms[:,None],1e-12)
 ids=np.zeros((len(vectors),k),np.int16);vals=np.zeros((len(vectors),k),np.float32);decoded=np.zeros_like(vectors,dtype=np.float32)
 for i,v in enumerate(vectors.astype(np.float32)):
  r=v.copy();used=np.zeros(len(atoms),bool)
  for j in range(k):
   corr=D@r;corr[used]=0;q=int(np.argmax(np.abs(corr)));coef=float(corr[q]/norms[q]);ids[i,j]=q;vals[i,j]=coef;decoded[i]+=coef*atoms[q];r-=coef*atoms[q];used[q]=True
 return ids,vals,decoded
def encode_pooled(vectors,W,pool):
 ids,vals,dec=omp(vectors,W[pool],K)
 return ids.astype(np.uint8),vals,dec
def save(path,arrays):
 np.savez(path,**arrays)
 with zipfile.ZipFile(path) as z:
  if any(i.compress_type!=zipfile.ZIP_STORED for i in z.infolist()):raise AssertionError('NPZ must be uncompressed')
 return Path(path).stat().st_size
def run(seed,model_dir,sae_path,out_dir):
 out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);print('stage=load',flush=True);model,tok,torchmod=h.load_model(model_dir)
 print('stage=extract',flush=True);t=time.perf_counter();vectors,manifest,support=h.extract_task_vectors(model,tok,torchmod,seed);extract_s=time.perf_counter()-t
 t=time.perf_counter();W,bias=load_sae(sae_path);pact=aggregate_pool(vectors,W,bias);activation_pool_s=time.perf_counter()-t
 t=time.perf_counter();rp=residual_pool(vectors,W);residual_pool_s=time.perf_counter()-t
 print('stage=encode',flush=True);t=time.perf_counter();rids,rvals,rdec=encode_pooled(vectors,W,rp);res_code_s=time.perf_counter()-t
 t=time.perf_counter();aids,avals,adec=encode_pooled(vectors,W,pact);act_code_s=time.perf_counter()-t
 t=time.perf_counter();gids,gvals,gdec=omp(vectors,W,K);global_code_s=time.perf_counter()-t
 model_bytes=sum((Path(model_dir)/n).stat().st_size for n in ['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']);sae_bytes=Path(sae_path).stat().st_size
 meta={'task_ids':np.arange(16,dtype=np.int16),'model_revision':np.frombuffer(h.REVISION.encode(),dtype='S40'),'model_sha256':np.frombuffer(h.MODEL_SHA.encode(),dtype='S64'),'sae_revision':np.frombuffer(SAE_REV.encode(),dtype='S40'),'sae_sha256':np.frombuffer(SAE_SHA.encode(),dtype='S64'),'schema':np.array([528,2048,POOL,K],np.int32)}
 payload={}
 payload['explicit_fv']=save(out/'explicit_fv.npz',dict(function_vectors=vectors.astype(np.float32),**meta))
 payload['global_omp16']=save(out/'global_omp16.npz',dict(feature_ids=gids.astype(np.int16),coefficients=gvals.astype(np.float32),**meta))
 payload['aggregate_pool64_omp16']=save(out/'aggregate_pool64_omp16.npz',dict(pool_feature_ids=pact,local_code_indices=aids,coefficients=avals.astype(np.float32),**meta))
 payload['residual_pool64_omp16']=save(out/'residual_pool64_omp16.npz',dict(pool_feature_ids=rp,local_code_indices=rids,coefficients=rvals.astype(np.float32),**meta))
 methods=[('none',np.zeros_like(vectors),0,0.),('explicit_fv',vectors,payload['explicit_fv'],0.),('global_omp16',gdec,payload['global_omp16'],global_code_s),('aggregate_pool64_omp16',adec,payload['aggregate_pool64_omp16'],activation_pool_s+act_code_s),('residual_pool64_omp16',rdec,payload['residual_pool64_omp16'],residual_pool_s+res_code_s)]
 rows=[]
 for name,decoded,pbytes,coding_s in methods:
  print('stage=evaluate:'+name,flush=True);t=time.perf_counter();m=h.evaluate(model,tok,torchmod,vectors,decoded,manifest);infer_s=time.perf_counter()-t;uses=name not in ('none','explicit_fv')
  ops={'global_omp16':16*16*512*2048,'aggregate_pool64_omp16':16*512*2048+16*16*512*POOL,'residual_pool64_omp16':POOL*12*512*2048+16*16*512*POOL}.get(name,0)
  rows.append({'seed':seed,'method':name,'payload_bytes':pbytes,'model_bytes':model_bytes,'sae_bytes':sae_bytes if uses else 0,'incremental_over_preloaded_sae_bytes':pbytes,'total_deployment_bytes':model_bytes+(sae_bytes if uses else 0)+pbytes,'explicit_fv_total_bytes':model_bytes+payload['explicit_fv'],'global_omp16_total_bytes':model_bytes+sae_bytes+payload['global_omp16'],'support_examples':128,'support_forward_calls':support['forward_calls'],'support_input_tokens':support['input_tokens'],'candidate_input_tokens':m['candidate_input_tokens'],'optimizer_updates':0,'extraction_seconds':extract_s,'coding_seconds':coding_s,'inference_seconds':infer_s,'wall_time_seconds':extract_s+coding_s+infer_s,'active_compute_proxy':int(ops+(K*512)+(m['candidate_sequences']*512 if uses else 0)),'metrics':m})
 np.savez(out/'pool_selection_audit.npz',aggregate_pool_ids=pact,residual_pool_ids=rp)
 (out/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 report={'experiment_id':'MA-528','seed':seed,'model_revision':h.REVISION,'model_sha256':h.MODEL_SHA,'sae_revision':SAE_REV,'sae_sha256':SAE_SHA,'model_bytes':model_bytes,'sae_bytes':sae_bytes,'aggregate_pool_ids':pact.tolist(),'residual_pool_ids':rp.tolist(),'pool_fit_tasks':FIT.tolist(),'heldout_tasks':HELD.tolist(),'timing':{'activation_pool':activation_pool_s,'residual_pool':residual_pool_s,'activation_pool_omp16':act_code_s,'residual_pool_omp16':res_code_s,'global_omp16':global_code_s},'rows':rows}
 (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'seed':seed,'residual_pool':rp.tolist(),'rows':[{'method':r['method'],'bytes':r['payload_bytes'],'accuracy':r['metrics']['heldout_accuracy'],'gold_logprob':r['metrics']['mean_gold_candidate_logprob']} for r in rows]},indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--model-dir',required=True);p.add_argument('--sae',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.seed,a.model_dir,a.sae,a.out)
if __name__=='__main__':main()
