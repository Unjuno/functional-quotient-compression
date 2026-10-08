"""MA-760 predeclared mechanism screen; no fresh data path exists in this program."""
import csv, hashlib, json, os, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'source'/'artifacts'; OUT.mkdir(parents=True,exist_ok=True)
MODEL='hf-internal-testing/tiny-stable-diffusion-pipe'; REV='3ee6c9f225f088ad5d35b624b6514b091e6a4849'
SEEDS=[7601,7602]; RHOS=[0.1,0.4,1.0]; RANKS=[1,2,4,8]; N_TRAIN=96; N_CAL=32; N_PRIV=32; N_EVAL=128
METHODS=['no_control','hard_shared','independent','additive','mirror','mirror_private']
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)

def sha(b):return hashlib.sha256(b).hexdigest()
def conv(x,k):return F.conv2d(x,k,padding=1)
def make_x(seed,n):
 g=torch.Generator().manual_seed(seed); z=torch.randn(n,3,8,8,generator=g)
 smooth=F.avg_pool2d(z,3,1,1); yy,xx=torch.meshgrid(torch.linspace(-1,1,8),torch.linspace(-1,1,8),indexing='ij')
 coords=torch.stack([xx,yy,torch.sin(3*xx+yy)]).unsqueeze(0).expand(n,-1,-1,-1)
 return (0.55*smooth+0.25*z+0.2*coords).contiguous()
def fit_kernel(x,y,alpha=1e-6):
 # Solve output-channel regressions from a convolutional design matrix.
 a=F.unfold(x,3,padding=1).transpose(1,2).reshape(-1,27).double()
 b=y.permute(0,2,3,1).reshape(-1,4).double()
 gram=a.T@a+alpha*torch.eye(a.shape[1],dtype=torch.float64)
 w=torch.linalg.solve(gram,a.T@b)
 return w.T.reshape(4,3,3,3).float()
def gen_tasks(seed,rho):
 g=torch.Generator().manual_seed(seed+int(rho*10000))
 base=torch.randn(4,3,3,3,generator=g)*0.10
 shared=torch.randn(3,4,3,3,3,generator=g)*0.07
 private=torch.randn(8,4,3,3,3,generator=g)*0.07
 coeff=torch.randn(8,3,generator=g)
 ks=[]
 for i in range(8):ks.append(base+rho*(torch.einsum('r,roihw->oihw',coeff[i],shared)+private[i]))
 return torch.stack(ks),base

def fit_code(x,y,mean,basis):
 n=len(basis); pred=torch.stack([conv(x,b) for b in basis],dim=-1) # N,O,H,W,R
 target=(y-conv(x,mean)).unsqueeze(-1)
 a=pred.permute(0,1,2,3,4).reshape(-1,n).double(); b=target.reshape(-1,1).double()
 return torch.linalg.lstsq(a,b).solution[:,0].float()
def top_sparse_residual(x,y,pred_code,k=8):
 residual_target=y-pred_code
 full=fit_kernel(x,residual_target)
 flat=full.flatten(); ids=torch.topk(flat.abs(),k).indices; vals=flat[ids]
 sparse=torch.zeros_like(flat);sparse[ids]=vals
 return sparse.reshape_as(full),ids

def apply_view(x,mean,basis,code,res=None,ids=None):
 y=conv(x,mean)
 for a,b in zip(code,basis):y=y+float(a)*conv(x,b)
 if res is not None:y=y+conv(x,res)
 return y

def serialized(obj,path):
 torch.save(obj,path); data=Path(path).read_bytes(); return len(data),sha(data)

def main():
 t0=time.perf_counter()
 pipe=StableDiffusionPipeline.from_pretrained(MODEL,revision=REV,safety_checker=None,requires_safety_checker=False)
 unet=pipe.unet.eval()
 for p in unet.parameters():p.requires_grad_(False)
 base_path=OUT/'tiny_unet_state.pt'
 base_bytes,base_sha=serialized({'model':MODEL,'revision':REV,'unet_state':unet.state_dict(),'config':dict(unet.config)},base_path)
 # Actual denoiser integration and isolated CPU call calibration.
 g=torch.Generator().manual_seed(7600); latent=torch.randn(2,4,8,8,generator=g);ctx=torch.zeros(2,2,unet.config.cross_attention_dim);ts=torch.full((2,),10,dtype=torch.long)
 with torch.inference_mode():
  _=unet(latent,ts,encoder_hidden_states=ctx).sample
  tbench=time.perf_counter()
  for _ in range(6): base_out=unet(latent,ts,encoder_hidden_states=ctx).sample
 unet_forward_s=(time.perf_counter()-tbench)/6
 results=[]; details=[]; whole=[]
 for seed in SEEDS:
  for rho in RHOS:
   true_k,_=gen_tasks(seed,rho)
   train_x=[]; train_y=[]; fitted=[]
   for c in range(6):
    x=make_x(seed*1000+int(rho*1000)+c,N_TRAIN);y=conv(x,true_k[c])
    train_x.append(x);train_y.append(y);fitted.append(fit_kernel(x,y))
   fitted=torch.stack(fitted)
   mean=fitted.mean(0)
   centered=(fitted-mean).reshape(6,-1).double()
   _,_,vh=torch.linalg.svd(centered,full_matrices=False)
   full_basis=torch.zeros(8,4,3,3,3)
   full_basis[:len(vh)]=vh.reshape(len(vh),4,3,3,3)
   hard=fit_kernel(torch.cat(train_x),torch.cat(train_y))
   # Fit/test held-out condition IDs 6,7. Separate code and private adaptation examples.
   per_condition={};direct_kernels=list(fitted)
   for c in range(8):
    if c<6:
     code_full=(full_basis.reshape(8,-1)*(fitted[c]-mean).flatten()).sum(1)
     for rank in RANKS:
      basis=full_basis[:rank]
      recon=mean+torch.einsum('r,r...->...',code_full[:rank],basis)
      residual=fitted[c]-recon
      flat=residual.flatten();ids=torch.topk(flat.abs(),8).indices;vals=flat[ids]
      per_condition[(c,rank)]=(code_full[:rank],ids,vals)
    else:
     xcal=make_x(seed*100000+int(rho*10000)+c*10+1,N_CAL)
     xpriv=make_x(seed*100000+int(rho*10000)+c*10+2,N_PRIV)
     ycal=conv(xcal,true_k[c]);ypriv=conv(xpriv,true_k[c])
     fullfit=fit_kernel(torch.cat([xcal,xpriv]),torch.cat([ycal,ypriv]))
     direct_kernels.append(fullfit)
     for rank in RANKS:
      basis=full_basis[:rank]
      code=fit_code(xcal,ycal,mean,basis)
      pred=apply_view(xpriv,mean,basis,code)
      resid,ids=top_sparse_residual(xpriv,ypriv,pred)
      per_condition[(c,rank)]=(code,ids,resid.flatten()[ids])
     per_condition[(c,'full')]=(fullfit,None,None)
   # Libraries saved per method, with the common frozen UNet charged separately once per deployment.
   # Controller payload is K=8; each per-condition payload is also serialized to audit marginal code state.
   for method in METHODS:
    ranks=RANKS if method in ('additive','mirror','mirror_private') else [0]
    for rank in ranks:
     lib={'format':'MA760-control-library-v1','method':method,'rank':rank,'rho':rho,'seed':seed,'base_model':MODEL,'base_revision':REV,'base_unet_sha256':base_sha,'base_unet_bytes':base_bytes,'metadata':{'condition_count':8,'input_channels':3,'output_channels':4,'kernel':'3x3'}}
     if method=='independent':lib['kernels']=torch.stack(direct_kernels)
     elif method=='hard_shared':lib['kernel']=hard
     elif method in ('additive','mirror','mirror_private'):
      lib['mean']=mean;lib['basis']=full_basis[:rank].clone();lib['codes']=torch.zeros(8,rank)
      for c in range(6):lib['codes'][c]=per_condition[(c,rank)][0]
      lib['codes'][6]=per_condition[(6,rank)][0];lib['codes'][7]=per_condition[(7,rank)][0]
      if method=='mirror_private':
       lib['private_indices']=torch.zeros(8,8,dtype=torch.long);lib['private_values']=torch.zeros(8,8)
       for c in range(8):
        ids,vals=per_condition[(c,rank)][1:]
        lib['private_indices'][c]=ids;lib['private_values'][c]=vals
     elif method=='no_control':pass
     path=OUT/f'seed{seed}_rho{rho}_{method}_r{rank}_K8.pt'
     b,digest=serialized(lib,path);whole.append({'seed':seed,'rho':rho,'method':method,'rank':rank,'artifact':str(path.relative_to(ROOT)),'adapter_payload_bytes':b,'adapter_sha256':digest,'base_unet_bytes':base_bytes,'whole_library_bytes':base_bytes+b,'base_unet_sha256':base_sha})
     # Record actual standalone condition payload size (code, index, residual or kernel) for held-out conditions.
     payloads={}
     for c in [6,7]:
      if method=='independent':obj={'kernel':direct_kernels[c]}
      elif method=='hard_shared':obj={'kernel':hard}
      elif method in ('additive','mirror','mirror_private'):
       code,ids,vals=per_condition[(c,rank)];obj={'code':code}
       if method=='mirror_private':obj.update({'private_indices':ids,'private_values':vals})
      else:obj={}
      obj={'condition_id':c,'state':obj}
      cp=OUT/f'tmp_condition_seed{seed}_rho{rho}_{method}_r{rank}_c{c}.pt';cb,ch=serialized(obj,cp);payloads[c]=(cb,ch)
     for c in [6,7]:
      xtest=make_x(seed*1000000+int(rho*100000)+c,N_EVAL);ytrue=conv(xtest,true_k[c])
      def predict(qx):
       if method=='no_control':return torch.zeros_like(conv(qx,true_k[c]))
       if method=='hard_shared':return conv(qx,hard)
       if method=='independent':return conv(qx,direct_kernels[c])
       code,ids,vals=per_condition[(c,rank)];res=None
       if method=='mirror_private':
        res=torch.zeros(108);res[ids]=vals;res=res.reshape(4,3,3,3)
       return apply_view(qx,mean,full_basis[:rank],code,res)
      ypred=predict(xtest)
      query_start=time.perf_counter()
      for _ in range(25):_ = predict(xtest)
      query_wall=(time.perf_counter()-query_start)/25
      mse=float((ypred-ytrue).square().mean());mse_true=float(ytrue.square().mean());norm=mse/max(mse_true,1e-12)
      # Denoiser integration: base epsilon is shared; control residual is added at UNet output.
      # One actual UNet call per seed/rho/condition is cached via a deterministic representative latent.
      lgen=torch.Generator().manual_seed(seed*100+int(rho*10)+c);z=torch.randn(1,4,8,8,generator=lgen);ec=torch.zeros(1,2,unet.config.cross_attention_dim);tt=torch.tensor([10])
      with torch.inference_mode():eps=unet(z,tt,encoder_hidden_states=ec).sample
      xint=make_x(seed+c+1,1);target_eps=eps+conv(xint,true_k[c]);int_res=predict(xint)
      candidate_eps=eps+int_res
      # The primary held-out residual metric is evaluated on the full synthetic batch above.
      # The actual UNet invocation verifies correct tensor interface and finite adapter composition.
      integration_finite=bool(torch.isfinite(candidate_eps).all() and torch.isfinite(target_eps).all())
      results.append({'condition':f'rho{rho}_id{c}','world_or_seed':seed,'rho':rho,'condition_id':c,'method':method,'rank':rank,'adapter_payload_bytes':b,'whole_library_payload_bytes':base_bytes+b,'marginal_condition_payload_bytes':payloads[c][0],'train_tokens_or_examples':N_TRAIN*6+N_CAL+N_PRIV,'optimizer_updates':0,'active_compute_proxy':{'independent':108*64,'hard_shared':108*64,'additive':rank*108*64,'mirror':rank*108*64,'mirror_private':rank*108*64+8*64,'no_control':0}[method],'wall_time_s':query_wall,'query_throughput_examples_s':N_EVAL/max(query_wall,1e-12),'primary_metric':'heldout_residual_mse','primary_value':mse,'normalized_residual_mse':norm,'target_residual_mse':mse_true,'integration_finite':integration_finite,'condition_payload_sha256':payloads[c][1]})
   details.append({'seed':seed,'rho':rho,'training_condition_kernels':fitted.tolist(),'shared_mean':mean.tolist(),'shared_basis':full_basis.tolist()})
 # Keep only persistent library artifacts; tmp standalone condition files are retained as actual serialized audits.
 (ROOT/'source'/'development_raw.json').write_text(json.dumps({'experiment_id':'MA-760','base_model':MODEL,'revision':REV,'base_unet_bytes':base_bytes,'base_unet_sha256':base_sha,'unet_forward_s_batch2':unet_forward_s,'results':results,'libraries':whole,'fresh_accessed':False,'wall_total_s':time.perf_counter()-t0},indent=2)+'\n')
 (ROOT/'source'/'development_summary.json').write_text(json.dumps({'rows':len(results),'seed_count':len(SEEDS),'rhos':RHOS,'rank_sweep':RANKS,'fresh_accessed':False,'max_rows_per_method':len(results)//len(METHODS)},indent=2)+'\n')
 print(json.dumps({'rows':len(results),'base_unet_bytes':base_bytes,'base_unet_sha256':base_sha,'unet_forward_s_batch2':unet_forward_s,'wall_total_s':time.perf_counter()-t0,'fresh_accessed':False},indent=2))

if __name__=='__main__':main()
