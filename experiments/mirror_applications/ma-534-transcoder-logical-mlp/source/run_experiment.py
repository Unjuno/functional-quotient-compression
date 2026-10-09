#!/usr/bin/env python3
"""MA-534 role-conditioned sparse transcoder bank screen."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download,snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM,AutoTokenizer
from role_ops import givens_view,sparse_decode,fit_centroids,assign_roles
MODEL='HuggingFaceTB/SmolLM2-135M'; MODEL_REV='93efa2f097d58c2a74874c7e644dbc9b0cee75a2'; TC_SHA='07f811a32b7c0deb5cb15ad3f1e8529bb38265768eafba0713b3c208b4bc02e5'; TC='EleutherAI/skip-transcoder-SmolLM2-135M-128x'; TC_REV='651f51421f2e1aa8fbd907e02ef421d3da55ff6d'; DATA='closji/wikitext__wikitext-2-raw-v1'; DATA_REV='d575192455c5e98b8daed777574046264579cb09'; LAYER=8; D=576; BANK=32; K=8; FULL_K=128; ROLES=4

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def nonoverlap_starts(n, count, seed, window=128):
 blocks=np.random.default_rng(seed).choice(n//window,size=min(count,n//window),replace=False)
 return sorted((blocks*window).astype(int).tolist())

def capture():
 return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['dev','fresh'],required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed);out=args.output;out.mkdir(parents=True,exist_ok=True)
 if args.split=='fresh': raise SystemExit('Fresh split remains sealed until the development gate passes and a frozen-state loader amendment is committed.')
 model_rev=MODEL_REV;model=AutoModelForCausalLM.from_pretrained(MODEL,revision=model_rev,torch_dtype=torch.float32).eval();tok=AutoTokenizer.from_pretrained(MODEL,revision=model_rev)
 ds_split='train' if args.split=='dev' else 'validation';ds=load_dataset(DATA,split=ds_split,streaming=True,revision=DATA_REV);texts=[]
 for row in ds:
  t=row['text'].strip()
  if len(t)>80:texts.append(t)
  if len(texts)>=128:break
 ids=tok('\n'.join(texts),return_tensors='pt').input_ids[0];starts=nonoverlap_starts(len(ids),32,args.seed)
 if len(starts)<32:raise RuntimeError(f'need 32 aligned blocks; got {len(starts)}')
 seq=torch.stack([ids[s:s+128] for s in starts]);block=model.model.layers[LAYER].mlp;xs=[];ys=[]
 def prehook(_m,a): pass
 def posthook(_m,a,y): xs.append(a[0].detach().cpu());ys.append(y.detach().cpu())
 h1=block.register_forward_pre_hook(prehook);h2=block.register_forward_hook(posthook);t0=time.perf_counter()
 with torch.inference_mode():
  for i in range(len(seq)):model(input_ids=seq[i:i+1])
 capture_wall=time.perf_counter()-t0;h1.remove();h2.remove()
 x=torch.cat(xs).reshape(-1,D).float();y=torch.cat(ys).reshape(-1,D).float();split=len(x)//2;xfit,yfit=x[:split],y[:split];xeval,yeval=x[split:],y[split:]
 centroids,role_fit=fit_centroids(xfit,ROLES,args.seed);role_eval=assign_roles(xeval,centroids)
 tcfile=hf_hub_download(TC,f'model.layers.{LAYER}.mlp/sae.safetensors',revision=TC_REV)
 cfgfile=hf_hub_download(TC,f'model.layers.{LAYER}.mlp/cfg.json',revision=TC_REV)
 if sha(tcfile)!=TC_SHA: raise RuntimeError('pinned transcoder checkpoint hash mismatch')
 with safe_open(tcfile,framework='pt',device='cpu') as f:w={k:f.get_tensor(k).float() for k in f.keys()}
 # Feature bank selected only on fit tokens; accumulate positive activations without a huge dense cache.
 score=torch.zeros(w['encoder.weight'].shape[0])
 with torch.inference_mode():
  for st in range(0,len(xfit),128):
   xx=xfit[st:st+128];p=xx@w['encoder.weight'].T+w['encoder.bias'];v,i=p.topk(FULL_K,dim=-1);v=v.clamp_min(0);score.scatter_add_(0,i.reshape(-1),v.reshape(-1))
 bank=score.topk(BANK).indices
 enc=w['encoder.weight'][bank].contiguous(); eb=w['encoder.bias'][bank].contiguous();dec=w['W_dec'][bank].contiguous();bias=w['b_dec'].contiguous();skip=w['W_skip'].contiguous()
 with torch.inference_mode(): cfit=torch.relu(xfit@enc.T+eb);ceval=torch.relu(xeval@enc.T+eb)
 # Role-conditioned hard supports are fit-only and shared across evaluation samples.
 supports=[]
 for r in range(ROLES):
  mask=role_fit==r; supports.append(cfit[mask].mean(0).topk(K).indices)
 supports=torch.stack(supports)
 def role_predict(code,xx,roles,kind,params=None):
  pred=torch.empty_like(xx);active=[]
  for r in range(ROLES):
   m=roles==r
   if not m.any():continue
   c=code[m]
   if kind=='shared': z=c
   elif kind=='gate': z=c*torch.zeros_like(c).scatter(1,supports[r].expand(len(c),-1),1.0)
   elif kind=='diag': z=c*params[r].exp()
   elif kind=='mirror': z=givens_view(c,params[r])
   else:raise ValueError(kind)
   yp,ii,v=sparse_decode(z,dec,bias,skip,xx[m],K);pred[m]=yp;active.append((ii,v))
  return pred,active
 # Fixed tied and fit-selected sparse support controls.
 with torch.inference_mode():
  shared,_=role_predict(ceval,xeval,role_eval,'shared')
  gated,_=role_predict(ceval,xeval,role_eval,'gate')
 # Fit role-specific diagonal and Givens controls with same batches/updates.
 diag=torch.nn.Parameter(torch.zeros(ROLES,BANK));opt_d=torch.optim.Adam([diag],lr=.01)
 angles=torch.nn.Parameter(torch.zeros(ROLES,BANK//2));opt_m=torch.optim.Adam([angles],lr=.01)
 rng=np.random.default_rng(args.seed+1);updates=200;batch=256
 for step in range(updates):
  ids_b=torch.as_tensor(rng.choice(len(xfit),size=batch,replace=True),dtype=torch.long);xb=xfit[ids_b];cb=cfit[ids_b];yb=yfit[ids_b];rb=role_fit[ids_b]
  opt_d.zero_grad();pd,_=role_predict(cb,xb,rb,'diag',diag);ld=(pd-yb).square().mean();ld.backward();opt_d.step()
  opt_m.zero_grad();pm,_=role_predict(cb,xb,rb,'mirror',angles);lm=(pm-yb).square().mean();lm.backward();opt_m.step()
 with torch.inference_mode():pd_eval,_=role_predict(ceval,xeval,role_eval,'diag',diag);pm_eval,active=role_predict(ceval,xeval,role_eval,'mirror',angles)
 # Full transcoder reference on eval and native MLP metrics.
 fullpred=[]
 with torch.inference_mode():
  for st in range(0,len(xeval),64):
   xx=xeval[st:st+64];p=xx@w['encoder.weight'].T+w['encoder.bias'];v,i=p.topk(FULL_K,dim=-1);v=v.clamp_min(0);o=(v.unsqueeze(-1)*w['W_dec'][i]).sum(1)+w['b_dec']+xx@w['W_skip'].T;fullpred.append(o)
 full=torch.cat(fullpred)
 def metric(pred,target):
  rows=[]
  for r in range(ROLES):
   m=role_eval==r
   if not m.any(): rows.append({'role':r,'n':0,'relative_mse':None,'cosine':None});continue
   err=(pred[m]-target[m]).square().sum();den=target[m].square().sum()+1e-12;cos=torch.nn.functional.cosine_similarity(pred[m],target[m],dim=-1).mean();rows.append({'role':r,'n':int(m.sum()),'relative_mse':float(err/den),'cosine':float(cos)})
  present=[z for z in rows if z['n']]
  return {'role_balanced_relative_mse':float(np.mean([z['relative_mse'] for z in present])),'worst_role_relative_mse':float(max(z['relative_mse'] for z in present)),'mean_cosine':float(np.mean([z['cosine'] for z in present])),'per_role':rows}
 metrics={'native_mlp':metric(yeval,yeval),'full_transcoder_top128':metric(full,yeval),'shared_bank_top8':metric(shared,yeval),'role_sparse_gate':metric(gated,yeval),'role_diagonal_gain':metric(pd_eval,yeval),'role_givens_mirror':metric(pm_eval,yeval)}
 # Actual replacement base bytes, with native layer-8 MLP weights removed.
 model_dir=Path(snapshot_download(MODEL,revision=model_rev,allow_patterns=['*.safetensors','*.bin','*.json','*.txt'])); mf=model_dir/'model.safetensors';other=sum(p.stat().st_size for p in model_dir.rglob('*') if p.is_file() and p.name!='model.safetensors')
 with safe_open(str(mf),framework='pt',device='cpu') as f:
  base_state={k:f.get_tensor(k) for k in f.keys() if not k.startswith(f'model.layers.{LAYER}.mlp.')}
  native_mlp={k:f.get_tensor(k) for k in f.keys() if k.startswith(f'model.layers.{LAYER}.mlp.')}
 basefile=out/'base_without_mlp.safetensors';save_file(base_state,str(basefile));base_safetensors_bytes=basefile.stat().st_size;base_bytes=base_safetensors_bytes+other;base_hash=sha(basefile)
 # The physical transcoder bank is only 32 selected atoms plus the shared skip path.
 common={'encoder_weight':enc,'encoder_bias':eb,'decoder_weight':dec,'decoder_bias':bias,'skip_weight':skip,'bank_indices':bank.int()}
 role_common={**common,'role_centroids':centroids}
 method_states={'shared_bank_top8':common,'role_sparse_gate':{**role_common,'role_supports':supports.int()},'role_diagonal_gain':{**role_common,'diagonal_gains':diag.detach()},'role_givens_mirror':{**role_common,'givens_angles':angles.detach()}}
 payloads={};standalone={}
 configs={'shared_bank_top8':{'kind':'shared_sparse_bank','k':K},'role_sparse_gate':{'kind':'per_role_sparse_gate','k':K,'roles':ROLES},'role_diagonal_gain':{'kind':'role_diagonal_gain','k':K,'roles':ROLES},'role_givens_mirror':{'kind':'role_givens_mirror','k':K,'roles':ROLES}}
 for name,state in method_states.items():
  pf=out/f'{name}.safetensors';save_file({k:v.contiguous() for k,v in state.items()},str(pf));cf=out/f'{name}.json';cf.write_text(json.dumps(configs[name],sort_keys=True,separators=(',',':'))+'\n');payloads[name]={'bytes':pf.stat().st_size,'sha256':sha(pf),'config_bytes':cf.stat().st_size};standalone[name]=base_bytes+payloads[name]['bytes']+payloads[name]['config_bytes']
 # Actual native model and 4x independent MLP upper payloads.
 full_base_bytes=sum(p.stat().st_size for p in model_dir.rglob('*') if p.is_file())
 role_mlp={f'role{r}.{k}':v.clone().contiguous() for r in range(ROLES) for k,v in native_mlp.items()}; independent_file=out/'four_full_mlps.safetensors';save_file(role_mlp,str(independent_file));independent_extra_bytes=independent_file.stat().st_size;independent_hash=sha(independent_file);independent_bytes=base_bytes+independent_extra_bytes;independent_file.unlink();basefile.unlink()
 # Compute proxies per token: selected bank encoder + skip + active decoder; Mirror adds 16 rotations.
 enc_macs=D*BANK;skip_macs=D*D;dec_macs=K*D;full_encoder_macs=D*73728;full_decoder_macs=FULL_K*D; native_macs=sum(p.numel() for n,p in block.named_parameters() if n.endswith('weight'))
 summary={'experiment_id':'MA-534','split':args.split,'seed':args.seed,'model':MODEL,'model_revision':model_rev,'transcoder_revision':TC_REV,'transcoder_file_sha256':sha(tcfile),'dataset':DATA,'dataset_revision':DATA_REV,'dataset_split':ds_split,'tokens_sha256':hashlib.sha256(seq.numpy().tobytes()).hexdigest(),'fit_tokens':len(xfit),'eval_tokens':len(xeval),'role_counts':torch.bincount(role_eval,minlength=ROLES).tolist(),'updates':updates,'bank_indices':bank.tolist(),'role_supports':supports.tolist(),'gates':{'shared_bank_top8':metrics['shared_bank_top8'],'role_sparse_gate':metrics['role_sparse_gate'],'role_diagonal_gain':metrics['role_diagonal_gain'],'role_givens_mirror':metrics['role_givens_mirror']},'metrics':metrics,'model_full_bytes':full_base_bytes,'model_without_native_mlp_bytes':base_bytes,'base_without_mlp_safetensors_bytes':base_safetensors_bytes,'base_without_mlp_sha256':base_hash,'independent_mlp_extra_bytes':independent_extra_bytes,'independent_mlp_sha256':independent_hash,'full_transcoder_incremental_bytes':341363524,'full_transcoder_standalone_bytes':base_bytes+341363524,'full_transcoder_file_sha256':TC_SHA,'method_payloads':payloads,'method_standalone_bytes':standalone,'four_independent_mlp_standalone_bytes':independent_bytes,'compute_macs_per_token':{'selected_bank_encoder':enc_macs,'skip':skip_macs,'top8_decoder':dec_macs,'givens_rotation_scalar_ops_proxy':BANK,'full_transcoder_encoder':full_encoder_macs,'full_transcoder_decoder':full_decoder_macs,'native_mlp_linear':native_macs},'wall_seconds':{'full_model_capture':capture_wall},'optimizer_updates':{'role_diagonal_gain':updates,'role_givens_mirror':updates},'active_features_per_token':K}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
