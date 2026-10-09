#!/usr/bin/env python3
"""Train the frozen support-only transcoder and evaluate MA-533 feature Views."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];HF=Path(__file__).with_name('shared_fv_harness.py')
spec=importlib.util.spec_from_file_location('ma533_harness',HF);h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
SAE_SHA='85b57dfe34d19769e2df0e208e38fda8815da8493c56b703a99ff74cc610dbf4';SAE_REV='36c2027509efad69836aa4231999c9b717848ecd';FIT=np.arange(12);HELD=np.arange(12,16);POOL=16;K=16;WIDTH=2048
class TiedSAE(torch.nn.Module):
 def __init__(self):super().__init__()
def topk_relu(z,k=32):
 values,indices=torch.topk(torch.relu(z),k,dim=-1,sorted=False)
 sparse=torch.zeros_like(z).scatter(-1,indices,values)
 return sparse,indices
def capture_transcoder_data(model,tok,torchmod,manifest):
 xs=[];ys=[];token_count=0;calls=0;layer=model.gpt_neox.layers[3]
 for tid in FIT:
  support=manifest[int(tid)]['support']
  for x,_ in support:
   demos=[p for p in support if p[0]!=x];text=h.prompt(x,demos);ids=h.enc(tok,text);capture={}
   def xhook(mod,inputs,out):capture['x']=(out[0] if isinstance(out,tuple) else out).detach()[0].cpu().numpy().astype(np.float32)
   def yhook(mod,inputs,out):capture['y']=(out[0] if isinstance(out,tuple) else out).detach()[0].cpu().numpy().astype(np.float32)
   hx=layer.post_attention_layernorm.register_forward_hook(xhook);hy=layer.mlp.register_forward_hook(yhook)
   try:
    with torchmod.no_grad():model(input_ids=torchmod.tensor([ids],dtype=torchmod.long),use_cache=False)
   finally:hx.remove();hy.remove()
   if capture['x'].shape!=capture['y'].shape or capture['x'].shape[1]!=512:raise ValueError('transcoder capture shape mismatch')
   xs.append(capture['x']);ys.append(capture['y']);token_count+=len(ids);calls+=1
 return np.concatenate(xs),np.concatenate(ys),{'fit_task_ids':FIT.tolist(),'support_examples':calls,'support_forward_calls':calls,'input_tokens':token_count,'captured_token_positions':int(sum(len(x) for x in xs))}
def train_transcoder(X,Y,updates=1000,batch=128):
 torch.manual_seed(0);x=torch.tensor(X,dtype=torch.float32);y=torch.tensor(Y,dtype=torch.float32);enc=torch.nn.Linear(512,WIDTH);dec=torch.nn.Linear(WIDTH,512);opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=.001,weight_decay=0.);n=len(x);t=time.perf_counter()
 for _ in range(updates):
  ix=torch.randint(n,(batch,));xb=x[ix];yb=y[ix];opt.zero_grad();z=enc(xb);sparse,_=topk_relu(z,32);pred=dec(sparse);loss=torch.mean((pred-yb)**2)+.001*torch.mean(torch.abs(sparse));loss.backward();opt.step()
 elapsed=time.perf_counter()-t;D=dec.weight.detach().cpu().numpy().T.astype(np.float32).copy();db=dec.bias.detach().cpu().numpy().astype(np.float32).copy()
 with torch.no_grad():
  err=[];var=[];nnz=[]
  for start in range(0,n,512):
   z=enc(x[start:start+512]);sparse,_=topk_relu(z,32);pred=dec(sparse);err.append(torch.sum((pred-y[start:start+512])**2).item());var.append(torch.sum((y[start:start+512]-y[start:start+512].mean(0))**2).item());nnz.append(torch.count_nonzero(sparse).item())
 fvu=float(sum(err)/max(sum(var),1e-12));density=float(sum(nnz)/(n*WIDTH))
 return D,db,{'updates':updates,'batch_size':batch,'fit_seconds':elapsed,'fit_compute_proxy':updates*batch*512*WIDTH*2*3,'mlp_reconstruction_fvu':fvu,'topk_nonzero_per_token':32,'latent_density':density}
def load_sae(path):
 if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=SAE_SHA:raise ValueError('pinned SAE hash mismatch')
 torch.serialization.add_safe_globals([TiedSAE]);sae=torch.load(path,map_location='cpu',weights_only=True)
 W=sae.encoder.detach().cpu().numpy().astype(np.float32);bias=sae.encoder_bias.detach().cpu().numpy().astype(np.float32)
 if W.shape!=(2048,512) or bias.shape!=(2048,):raise ValueError('SAE shape mismatch')
 return W,bias
def residual_pool(vectors,W):
 norms=np.linalg.norm(W.astype(np.float64),axis=1);Dn=W/np.maximum(norms[:,None],1e-12);res=vectors[FIT].copy();used=np.zeros(len(W),bool);ids=[]
 for _ in range(POOL):
  c=np.abs(res@Dn.T).sum(0);c[used]=-np.inf;j=int(np.argmax(c));ids.append(j);used[j]=True;coef=(res@Dn[j])/max(float(norms[j]),1e-12);res-=coef[:,None]*W[j]
 return np.asarray(ids,np.int16)
def omp(vectors,atoms,k=K):
 norms=np.linalg.norm(atoms.astype(np.float64),axis=1);Dn=atoms/np.maximum(norms[:,None],1e-12);ids=np.zeros((len(vectors),k),np.int16);coef=np.zeros((len(vectors),k),np.float32);out=np.zeros_like(vectors,np.float32)
 for i,v in enumerate(vectors.astype(np.float32)):
  r=v.copy();used=np.zeros(len(atoms),bool)
  for j in range(k):
   cor=Dn@r;cor[used]=0;q=int(np.argmax(np.abs(cor)));c=float(cor[q]/norms[q]);ids[i,j]=q;coef[i,j]=c;out[i]+=c*atoms[q];r-=c*atoms[q];used[q]=True
 return ids,coef,out
def save(path,arrays):
 np.savez(path,**arrays)
 with zipfile.ZipFile(path) as z:
  if any(q.compress_type!=zipfile.ZIP_STORED for q in z.infolist()):raise AssertionError('payload must be uncompressed NPZ')
 return Path(path).stat().st_size
def run(seed,model_dir,sae_path,out_dir):
 out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);model,tok,torchmod=h.load_model(model_dir)
 print('stage=task_vectors',flush=True);t=time.perf_counter();vectors,manifest,task_audit=h.extract_task_vectors(model,tok,torchmod,seed);fv_s=time.perf_counter()-t
 print('stage=transcoder_capture',flush=True);t=time.perf_counter();X,Y,cap_audit=capture_transcoder_data(model,tok,torchmod,manifest);capture_s=time.perf_counter()-t
 print('stage=transcoder_fit',flush=True);D,db,train_audit=train_transcoder(X,Y);fit_s=train_audit['fit_seconds']
 print('stage=feature_codes',flush=True);pool_start=time.perf_counter();pool=residual_pool(vectors,D);pool_s=time.perf_counter()-pool_start;t=time.perf_counter();si,sc,shared=omp(vectors,D[pool]);shared_code_s=time.perf_counter()-t;t=time.perf_counter();ti,tv,trans_global=omp(vectors,D);trans_global_s=time.perf_counter()-t
 W,bias=load_sae(sae_path);t=time.perf_counter();ai,av,sae_global=omp(vectors,W);sae_code_s=time.perf_counter()-t
 model_bytes=sum((Path(model_dir)/n).stat().st_size for n in ['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']);sae_bytes=Path(sae_path).stat().st_size
 meta={'task_ids':np.arange(16,dtype=np.int16),'model_revision':np.frombuffer(h.REVISION.encode(),dtype='S40'),'model_sha256':np.frombuffer(h.MODEL_SHA.encode(),dtype='S64'),'schema':np.array([533,WIDTH,POOL,K],np.int32)}
 payload={};payload['explicit_fv']=save(out/'explicit_fv.npz',dict(function_vectors=vectors.astype(np.float32),**meta))
 payload['shared_transcoder_pool16']=save(out/'shared_transcoder_pool16.npz',dict(pool_global_ids=pool,decoder_atoms=D[pool],code_indices=si.astype(np.uint8),coefficients=sc,**meta))
 payload['global_transcoder_omp16']=save(out/'global_transcoder_omp16.npz',dict(full_decoder_dictionary=D,feature_ids=ti,coefficients=tv,**meta))
 saemeta={**meta,'sae_revision':np.frombuffer(SAE_REV.encode(),dtype='S40'),'sae_sha256':np.frombuffer(SAE_SHA.encode(),dtype='S64')}
 # The pinned checkpoint is the paid SAE dictionary object for this control; do not serialize the same matrix twice.
 payload['global_sae_omp16']=save(out/'global_sae_omp16.npz',dict(feature_ids=ai,coefficients=av,**saemeta))
 methods=[('none',np.zeros_like(vectors),0,0.,False),('explicit_fv',vectors,payload['explicit_fv'],0.,False),('shared_transcoder_pool16',shared,payload['shared_transcoder_pool16'],shared_code_s,False),('global_transcoder_omp16',trans_global,payload['global_transcoder_omp16'],trans_global_s,False),('global_sae_omp16',sae_global,payload['global_sae_omp16'],sae_code_s,True)]
 rows=[]
 for name,decoded,pbytes,coding_s,uses_sae in methods:
  print('stage=evaluate:'+name,flush=True);t=time.perf_counter();m=h.evaluate(model,tok,torchmod,vectors,decoded,manifest);inf=time.perf_counter()-t
  dict_bytes=(D[pool].nbytes+pool.nbytes if name=='shared_transcoder_pool16' else D.nbytes if name=='global_transcoder_omp16' else sae_bytes if name=='global_sae_omp16' else 0)
  ops={'shared_transcoder_pool16':16*K*512*POOL,'global_transcoder_omp16':16*K*512*WIDTH,'global_sae_omp16':16*K*512*WIDTH}.get(name,0)
  uses_trans=name in ('shared_transcoder_pool16','global_transcoder_omp16')
  code_bytes=(pbytes if name=='global_sae_omp16' else pbytes-dict_bytes if pbytes else 0)
  rows.append({'seed':seed,'method':name,'payload_bytes':pbytes,'dictionary_bytes':int(dict_bytes),'code_bytes':int(code_bytes),'model_bytes':model_bytes,'sae_bytes':sae_bytes if uses_sae else 0,'total_deployment_bytes':model_bytes+(sae_bytes if uses_sae else 0)+pbytes,'explicit_fv_total_bytes':model_bytes+payload['explicit_fv'],'support_examples':task_audit['forward_calls']//2,'support_forward_calls':task_audit['forward_calls']+(cap_audit['support_forward_calls'] if uses_trans else 0),'support_input_tokens':task_audit['input_tokens']+(cap_audit['input_tokens'] if uses_trans else 0),'transcoder_captured_token_positions':cap_audit['captured_token_positions'] if uses_trans else 0,'candidate_input_tokens':m['candidate_input_tokens'],'optimizer_updates':1000 if uses_trans else 0,'active_compute_proxy':int((train_audit['fit_compute_proxy']+POOL*12*512*WIDTH if uses_trans else 0)+ops+(m['candidate_sequences']*512 if name!='none' and name!='explicit_fv' else 0)),'extraction_seconds':fv_s,'transcoder_capture_seconds':capture_s if uses_trans else 0.,'transcoder_fit_seconds':fit_s if uses_trans else 0.,'coding_seconds':coding_s,'inference_seconds':inf,'wall_time_seconds':fv_s+(capture_s+fit_s+pool_s if uses_trans else 0.)+coding_s+inf,'metrics':m})
 np.savez(out/'pool_audit.npz',pool_ids=pool)
 np.savez(out/'transcoder_summary.npz',decoder_atoms_hash=np.frombuffer(hashlib.sha256(D.tobytes()).hexdigest().encode(),dtype='S64'),captured_tokens=np.array([cap_audit['captured_token_positions']],np.int64))
 (out/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 report={'experiment_id':'MA-533','seed':seed,'model_revision':h.REVISION,'model_sha256':h.MODEL_SHA,'sae_revision':SAE_REV,'sae_sha256':SAE_SHA,'transcoder_decoder_sha256':hashlib.sha256(D.tobytes()).hexdigest(),'decoder_shape':list(D.shape),'decoder_bytes':int(D.nbytes),'model_bytes':model_bytes,'sae_bytes':sae_bytes,'support_task_ids':FIT.tolist(),'heldout_tasks':HELD.tolist(),'capture':cap_audit,'transcoder_training':train_audit,'timings':{'task_vector_extraction':fv_s,'transcoder_activation_capture':capture_s,'transcoder_fit':fit_s,'shared_pool_selection':pool_s,'shared_code':shared_code_s,'global_transcoder_code':trans_global_s,'global_sae_code':sae_code_s},'pool_ids':pool.tolist(),'rows':rows}
 (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'seed':seed,'transcoder_fvu':train_audit['mlp_reconstruction_fvu'],'pool':pool.tolist(),'rows':[{'method':r['method'],'payload_bytes':r['payload_bytes'],'total_bytes':r['total_deployment_bytes'],'accuracy':r['metrics']['heldout_accuracy'],'gold':r['metrics']['mean_gold_candidate_logprob']} for r in rows]},indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--model-dir',required=True);p.add_argument('--sae',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.seed,a.model_dir,a.sae,a.out)
if __name__=='__main__':main()
