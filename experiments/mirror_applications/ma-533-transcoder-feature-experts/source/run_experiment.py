#!/usr/bin/env python3
"""MA-533 native transcoder reconstruction fidelity screen.

A single-layer capture/replay wrapper around the pinned SmolLM2 skip-transcoder.
The next-token language model is not rerun: the test target is the native MLP
sublayer output on the same recorded residual inputs.
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download, model_info
from datasets import load_dataset
from transcoder_ops import forward as transcoder_forward, nonoverlap_starts
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='HuggingFaceTB/SmolLM2-135M'
TC='EleutherAI/skip-transcoder-SmolLM2-135M-128x'
TC_REV='651f51421f2e1aa8fbd907e02ef421d3da55ff6d'
LAYER=8; K=128; DEVICE='cpu'; THREADS=1
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--split',choices=['dev','fresh'],required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--examples',type=int,default=32); ap.add_argument('--max-tokens',type=int,default=4096); ap.add_argument('--output',type=Path,required=True)
 a=ap.parse_args(); torch.set_num_threads(THREADS); torch.manual_seed(a.seed); np.random.seed(a.seed)
 out=a.output; out.mkdir(parents=True,exist_ok=True)
 mr=model_info(MODEL); model_rev=mr.sha
 model=AutoModelForCausalLM.from_pretrained(MODEL,revision=model_rev,torch_dtype=torch.float32).eval()
 tok=AutoTokenizer.from_pretrained(MODEL,revision=model_rev)
 # Stream a deterministic prefix of the pinned training split only. Fresh uses distinct seed offset.
 dataset_split='train' if a.split=='dev' else 'validation'
 ds=load_dataset('closji/wikitext__wikitext-2-raw-v1',split=dataset_split,streaming=True,revision='d575192455c5e98b8daed777574046264579cb09')
 texts=[]
 for row in ds:
  t=row['text'].strip()
  if len(t)>80: texts.append(t)
  if len(texts)>=a.examples*2: break
 joined='\n'.join(texts)
 enc=tok(joined,return_tensors='pt',truncation=False).input_ids[0]
 # Seeded offset into the same training token stream, enough tokens for non-overlapping sequences.
 starts=nonoverlap_starts(len(enc),a.examples,a.seed,128)
 if len(starts)<a.examples: raise RuntimeError(f'need {a.examples} nonoverlapping windows, got {len(starts)} from {len(enc)} tokens')
 chunks=torch.stack([enc[int(s):int(s)+128] for s in starts])
 block=model.model.layers[LAYER].mlp
 captured={}
 def pre(_m,args): captured['x']=args[0].detach()
 handle=block.register_forward_pre_hook(pre)
 with torch.inference_mode(): model(input_ids=chunks[:1])
 handle.remove()
 # Capture dense MLP outputs for each sequence; hook the module output.
 xs=[]; ys=[]
 def pre2(_m,args): captured['x']=args[0].detach()
 def post2(_m,args,y): xs.append(args[0].detach().cpu()); ys.append(y.detach().cpu())
 h1=block.register_forward_pre_hook(pre2); h2=block.register_forward_hook(post2)
 t0=time.perf_counter()
 with torch.inference_mode():
  for i in range(len(chunks)): model(input_ids=chunks[i:i+1])
 wall=time.perf_counter()-t0; h1.remove(); h2.remove()
 x=torch.cat(xs).reshape(-1,xs[0].shape[-1]).float(); y=torch.cat(ys).reshape(-1,ys[0].shape[-1]).float(); split_at=(len(x)//128//2)*128; x_fit,y_fit=x[:split_at],y[:split_at]; x,y=x[split_at:],y[split_at:]
 tc_path=hf_hub_download(TC,f'model.layers.{LAYER}.mlp/sae.safetensors',revision=TC_REV)
 cfg_path=hf_hub_download(TC,f'model.layers.{LAYER}.mlp/cfg.json',revision=TC_REV)
 with safe_open(tc_path,framework='pt',device='cpu') as f: w={k:f.get_tensor(k).float() for k in f.keys()}
 # Match EleutherAI/sparsify SparseCoder.forward: top-k encoder, decoder plus x @ W_skip.T.
 tc_t=time.perf_counter()
 with torch.inference_mode():
  yhat,acts,idx=transcoder_forward(x,w,K)
  skip=x @ w['W_skip'].T
 transcoder_wall=time.perf_counter()-tc_t
 dense_t=time.perf_counter()
 with torch.inference_mode(): dense_pred=block(x)
 dense_layer_wall=time.perf_counter()-dense_t
 # Baseline zero-residual estimate and rank-128 linear SVD fit on a fixed development prefix.
 zero=torch.zeros_like(y)
 fit_n=min(2048,len(x_fit)); xf=x_fit[:fit_n]; yf=y_fit[:fit_n]
 # centered linear regression, rank 128; SVD on CPU can dominate wall time and is included separately.
 svd_t=time.perf_counter(); xm=xf.mean(0); ym=yf.mean(0); xc=xf-xm; yc=yf-ym
 cov=xc.T @ yc / max(1,len(xc)-1)
 u,s,vh=torch.linalg.svd(cov,full_matrices=False); rank=128
 left=u[:,:rank]*s[:rank]; right=vh[:rank]; bias=ym-xm@left@right
 low=x@left@right+bias
 svd_wall=time.perf_counter()-svd_t
 rank_t=time.perf_counter()
 with torch.inference_mode(): low=x@left@right+bias
 rank_inference_wall=time.perf_counter()-rank_t
 skip_t=time.perf_counter()
 with torch.inference_mode(): skip=x@w['W_skip'].T
 skip_inference_wall=time.perf_counter()-skip_t
 def metrics(pred,target=None):
  target=y if target is None else target
  err=(pred-target).square().sum(-1); den=target.square().sum(-1)+1e-12
  cos=torch.nn.functional.cosine_similarity(pred,target,dim=-1)
  return {'relative_mse':float(err.sum()/den.sum()),'mean_token_relative_mse':float((err/den).mean()),'mean_cosine':float(cos.mean()),'p50_token_relative_error':float((err/den).sqrt().median()),'p95_token_relative_error':float(torch.quantile((err/den).sqrt(),.95))}
 # Development-only normalization diagnostics motivated by the checkpoint model card.
 x_rms=x_fit.square().mean().sqrt(); y_rms=y_fit.square().mean().sqrt()
 with torch.inference_mode():
  yhat_global,_,_=transcoder_forward(x/x_rms,w,K); yhat_global=yhat_global*y_rms
  x_unit=x/(x.norm(dim=-1,keepdim=True)+1e-12); y_unit=y/(y.norm(dim=-1,keepdim=True)+1e-12)
  yhat_unit,_,_=transcoder_forward(x_unit,w,K)
 head_mean=x_fit.mean(0); head_std=x_fit.std(0,unbiased=False).clamp_min(1e-6)
 xfit_std=(x_fit-head_mean)/head_std; xeval_std=(x-head_mean)/head_std
 xfit_aug=torch.cat([xfit_std,torch.ones((len(xfit_std),1))],dim=1); xeval_aug=torch.cat([xeval_std,torch.ones((len(xeval_std),1))],dim=1)
 y_norm_fit=y_fit.norm(dim=-1)
 gram=xfit_aug.T@xfit_aug; reg=torch.eye(gram.shape[0]); reg[-1,-1]=0.0
 norm_head=torch.linalg.solve(gram+1e-3*reg,xfit_aug.T@y_norm_fit)
 head_t=time.perf_counter()
 with torch.inference_mode():
  direction=yhat_unit/(yhat_unit.norm(dim=-1,keepdim=True)+1e-12)
  predicted_norm=(xeval_aug@norm_head).clamp_min(1e-8)
  yhat_direction_norm=direction*predicted_norm[:,None]
 head_inference_wall=time.perf_counter()-head_t
 # Serialize each method's exact inference state; charge its own code/state and config bytes.
 payload=out/'transcoder.safetensors'; save_file(w,str(payload))
 skip_payload=out/'skip_only.safetensors'; save_file({'W_skip':w['W_skip']},str(skip_payload))
 norm_payload=out/'global_rms_scales.safetensors'; save_file({'input_rms':x_rms.reshape(1),'output_rms':y_rms.reshape(1)},str(norm_payload))
 head_payload=out/'direction_norm_head.safetensors'; save_file({'input_mean':head_mean.contiguous(),'input_std':head_std.contiguous(),'norm_head':norm_head.contiguous()},str(head_payload))
 rank_payload=out/'rank128.safetensors'; save_file({'left':left.contiguous(),'right':right.contiguous(),'bias':bias.contiguous()},str(rank_payload))
 cfg=json.loads(Path(cfg_path).read_text()); (out/'config.json').write_text(json.dumps(cfg,sort_keys=True,separators=(',',':'))+'\n')
 cfg_bytes=(out/'config.json').stat().st_size; payload_bytes=payload.stat().st_size
 (out/'direction_norm_head.json').write_text('{"kind":"unit_l2_transcoder_plus_fit_linear_norm_head","rank":1,"input_dim":576}\n')
 (out/'global_rms_scales.json').write_text('{"kind":"fit_global_rms_normalized_transcoder","input_dim":576}\n')
 (out/'skip_only.json').write_text('{"kind":"linear_skip","input_dim":576}\n'); (out/'rank128.json').write_text('{"kind":"rank128_affine","rank":128}\n')
 skip_bytes=skip_payload.stat().st_size+(out/'skip_only.json').stat().st_size; rank_bytes=rank_payload.stat().st_size+(out/'rank128.json').stat().st_size; norm_bytes=payload_bytes+cfg_bytes+norm_payload.stat().st_size+(out/'global_rms_scales.json').stat().st_size; head_bytes=payload_bytes+cfg_bytes+head_payload.stat().st_size+(out/'direction_norm_head.json').stat().st_size
 summary={'experiment_id':'MA-533','split':a.split,'seed':a.seed,'n_sequences':len(chunks),'tokens':int(len(chunks)*128),'activation_vectors':int(x.shape[0]),'fit_vectors':int(x_fit.shape[0]),'model':MODEL,'model_revision':model_rev,'model_repo_files':{},'transcoder_repo':TC,'transcoder_revision':TC_REV,'transcoder_sha256':sha(tc_path),'transcoder_file_bytes':payload_bytes,'transcoder_payload_sha256':sha(payload),'config_bytes':cfg_bytes,'transcoder_incremental_bytes':payload_bytes+cfg_bytes,'active_nonzeros_per_vector':int(K),'optimizer_updates':0,'data':f'closji/wikitext__wikitext-2-raw-v1 {dataset_split} parquet, revision d575192455c5e98b8daed777574046264579cb09','input_tokens_sha256':hashlib.sha256(chunks.numpy().tobytes()).hexdigest(),'full_model_capture_wall_seconds':wall,'native_dense_mlp_wall_seconds':dense_layer_wall,'transcoder_encode_decode_wall_seconds':transcoder_wall,'rank128_svd_fit_wall_seconds':svd_wall,'rank128_inference_wall_seconds':rank_inference_wall,'skip_only_inference_wall_seconds':skip_inference_wall,'fit_norm_head_inference_wall_seconds':head_inference_wall,'compute_proxy':{'vectors':int(x.shape[0]),'transcoder_encoder_dense_macs':int(x.shape[0]*w['encoder.weight'].numel()),'transcoder_decoder_sparse_macs':int(x.shape[0]*K*x.shape[-1]),'transcoder_skip_macs':int(x.shape[0]*w['W_skip'].numel()),'rank128_linear_macs':int(x.shape[0]*2*rank*x.shape[-1]),'fit_norm_head_macs':int(x.shape[0]*(3*x.shape[-1]+1)),'native_mlp_linear_macs':int(x.shape[0]*sum(p.numel() for n,p in block.named_parameters() if n.endswith('weight')))},'methods':{'skip_only':metrics(skip),'transcoder_top128':metrics(yhat),'rank128_cross_covariance_svd':metrics(low),'transcoder_global_rms_normalized':metrics(yhat_global),'transcoder_token_l2_direction_only':metrics(yhat_unit,y_unit),'transcoder_direction_plus_fit_norm_head':metrics(yhat_direction_norm)},'normalization_scales':{'fit_input_rms':float(x_rms),'fit_output_rms':float(y_rms),'global_rms_incremental_bytes':norm_bytes,'global_rms_incremental_bytes':norm_bytes,'token_l2_variant_is_deployable':False,'fit_norm_head_incremental_bytes':head_bytes,'fit_norm_head_added_parameters':int(norm_head.numel()+head_mean.numel()+head_std.numel())}}
 # Hash revision manifests to enable exact provenance; actual model payload bytes are independently fetched and charged.
 from huggingface_hub import snapshot_download
 snap=snapshot_download(MODEL,revision=model_rev,allow_patterns=['*.safetensors','*.bin','*.json'])
 files=[]; model_bytes=0; other_bytes=0
 for p in Path(snap).rglob('*'):
  if p.is_file():
   model_bytes+=p.stat().st_size
   if p.name!='model.safetensors': other_bytes+=p.stat().st_size
   files.append({'path':str(p.relative_to(snap)),'bytes':p.stat().st_size,'sha256':sha(p)})
 base_state_path=Path(snap)/'model.safetensors'; removed_prefix=f'model.layers.{LAYER}.mlp.'
 with safe_open(str(base_state_path),framework='pt',device='cpu') as sf:
  keys=list(sf.keys()); base_without={k:sf.get_tensor(k) for k in keys if not k.startswith(removed_prefix)}; removed={k:sf.get_tensor(k) for k in keys if k.startswith(removed_prefix)}
 if len(removed)!=3: raise RuntimeError(f'expected 3 replaced MLP weights; got {list(removed)}')
 stripped=out/'base_without_layer8_mlp.safetensors'; save_file(base_without,str(stripped)); stripped_bytes=stripped.stat().st_size; stripped_sha=sha(stripped); stripped.unlink()
 deploy_base_bytes=stripped_bytes+other_bytes
 summary['model_base_bytes']=model_bytes; summary['model_without_layer8_mlp_bytes']=deploy_base_bytes; summary['removed_native_mlp_tensors']={k:{'shape':list(v.shape),'dtype':str(v.dtype),'bytes':v.numel()*v.element_size()} for k,v in removed.items()}; summary['model_without_layer8_mlp_safetensors_bytes']=stripped_bytes; summary['model_without_layer8_mlp_safetensors_sha256']=stripped_sha; summary['model_files']=files
 summary['transcoder_standalone_total_bytes']=deploy_base_bytes+payload_bytes+cfg_bytes
 summary['method_incremental_bytes']={'native_mlp':0,'skip_only':skip_bytes,'transcoder_top128':payload_bytes+cfg_bytes,'rank128_cross_covariance_svd':rank_bytes,'transcoder_global_rms_normalized':norm_bytes,'transcoder_direction_plus_fit_norm_head':head_bytes}
 summary['method_standalone_bytes']={'native_mlp':model_bytes,'skip_only':deploy_base_bytes+skip_bytes,'transcoder_top128':deploy_base_bytes+payload_bytes+cfg_bytes,'rank128_cross_covariance_svd':deploy_base_bytes+rank_bytes,'transcoder_global_rms_normalized':deploy_base_bytes+norm_bytes,'transcoder_direction_plus_fit_norm_head':deploy_base_bytes+head_bytes}
 summary['method_payload_sha256']={'skip_only':sha(skip_payload),'transcoder_top128':sha(payload),'rank128_cross_covariance_svd':sha(rank_payload),'global_rms_scales':sha(norm_payload),'fit_norm_head':sha(head_payload)}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
