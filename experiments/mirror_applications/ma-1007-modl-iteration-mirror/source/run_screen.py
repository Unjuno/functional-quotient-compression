#!/usr/bin/env python3
"""Development, replay and sealed-fresh runner for MA-1007."""
import argparse,copy,hashlib,io,json,pathlib,time
import numpy as np
import torch
import torch.nn.functional as F
from modl_torch import MoDLNet,make_case,rel_measurement_residual

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def train(model,cases,steps,only=None):
 for p in model.parameters():p.requires_grad_(only is None)
 if only=='angles':model.angles.requires_grad_(True)
 if only=='lambdas':model.raw_lambda.requires_grad_(True)
 opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.003);hist=[];t=time.perf_counter();model.train()
 for i in range(steps):
  q=cases[i%len(cases)];x,cg=model(q);loss=(x.abs()-q['target']).square().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],2.0);opt.step();hist.append(float(loss.detach()))
 return {'updates':steps,'last_mse':hist[-1],'wall_seconds':time.perf_counter()-t,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
def global_ssim(x,y):
 x=x[None,None];y=y[None,None];mu_x=F.avg_pool2d(x,7,1,3);mu_y=F.avg_pool2d(y,7,1,3)
 vx=F.avg_pool2d(x*x,7,1,3)-mu_x.square();vy=F.avg_pool2d(y*y,7,1,3)-mu_y.square();cov=F.avg_pool2d(x*y,7,1,3)-mu_x*mu_y;c1=.01**2;c2=.03**2
 return float((((2*mu_x*mu_y+c1)*(2*cov+c2))/((mu_x.square()+mu_y.square()+c1)*(vx+vy+c2))).mean())
@torch.no_grad()
def evaluate(model,cases):
 model.eval();nmse=[];psnr=[];ssim=[];dc=[];cg=[];times=[]
 for q in cases:
  t=time.perf_counter();x,cr=model(q);times.append(time.perf_counter()-t);target=q['target'];mag=x.abs();mse=(mag-target).square().mean()
  nmse.append(float((mag-target).square().sum()/(target.square().sum()+1e-12)));psnr.append(float(20*torch.log10(target.max()/torch.sqrt(mse+1e-12))));ssim.append(global_ssim(mag,target));dc.append(rel_measurement_residual(x,q));cg.append(float(np.mean(cr)))
 return {'nmse':float(np.mean(nmse)),'psnr_db':float(np.mean(psnr)),'global_ssim':float(np.mean(ssim)),'measured_residual':float(np.mean(dc)),'cg_relative_residual':float(np.mean(cg)),'mean_forward_s':float(np.mean(times)),'n':len(cases)}
def payload(model,name):
 if name in ('independent_frozen','independent_ft'):
  obj={'method':name,'weights':{k:v.detach().cpu() for k,v in model.state_dict().items()},'operator':'multi-coil Cartesian FFT; CG=5; steps=3','shape':[32,32],'coils':4}
 else:
  obj={'method':name,'shared_denoiser':{k:v.detach().cpu() for k,v in model.denoiser.state_dict().items()},'lambda':model.raw_lambda.detach().cpu(),'operator':'multi-coil Cartesian FFT; CG=5; steps=3','shape':[32,32],'coils':4}
  if name=='mirror':obj['angles']=model.angles.detach().cpu()
 f=io.BytesIO();torch.save(obj,f);b=f.getvalue();return len(b),hashlib.sha256(b).hexdigest()
def build_models(base_ind,base_tied):
 models={'independent_frozen':copy.deepcopy(base_ind),'hard_tied':copy.deepcopy(base_tied)}
 m=MoDLNet('step_lambda');m.denoiser.load_state_dict(base_tied.denoiser.state_dict());m.raw_lambda.data.copy_(base_tied.raw_lambda.detach().expand(3));models['scalar_step']=m
 m=MoDLNet('mirror');m.denoiser.load_state_dict(base_tied.denoiser.state_dict());m.raw_lambda.data.copy_(base_tied.raw_lambda.detach());models['mirror']=m
 return models
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['dev','replay','fresh'],default='dev');ap.add_argument('--out',required=True);ap.add_argument('--lock',default='');a=ap.parse_args();torch.set_num_threads(1);torch.manual_seed(107001)
 if a.stage=='dev':
  train_a=[make_case(107001+i,'A') for i in range(24)];support=[make_case(107101+i,'B') for i in range(8)];dev=[make_case(107201+i,'B') for i in range(16)]
  torch.manual_seed(107001);base_ind=MoDLNet('independent');pre_ind=train(base_ind,train_a,24);base_ind=copy.deepcopy(base_ind)
  torch.manual_seed(107001);base_tied=MoDLNet('tied');pre_tied=train(base_tied,train_a,24);base_tied=copy.deepcopy(base_tied)
  models=build_models(base_ind,base_tied);adapt={}
  adapt['independent_frozen']={'updates':0,'wall_seconds':0,'trainable_parameters':0};adapt['hard_tied']={'updates':0,'wall_seconds':0,'trainable_parameters':0}
  adapt['scalar_step']=train(models['scalar_step'],support,24,'lambdas');adapt['mirror']=train(models['mirror'],support,24,'angles')
  upper=copy.deepcopy(base_ind);adapt['independent_ft']=train(upper,support,24);models['independent_ft']=upper
  metrics={k:evaluate(m,dev) for k,m in models.items()};payloads={k:payload(m,k) for k,m in models.items()}
  out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);cp=out.with_name('DEV_MODELS.pt')
  torch.save({'base_independent':base_ind.state_dict(),'base_tied':base_tied.state_dict(),'models':{k:m.state_dict() for k,m in models.items()}},cp)
  lock={'stage':'development','pretraining':{'independent':pre_ind,'tied':pre_tied},'adaptation':adapt,'metrics':metrics,'payloads':payloads,'parameters':{k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},'fresh_seeds':[107301,107307,107313],'fresh_per_seed':8,'fresh_opened':False,'checkpoint':cp.name,'checkpoint_sha256':sha(cp)}
  out.write_text(json.dumps(lock,indent=2)+'\n');print(json.dumps(lock,indent=2))
 else:
  if not a.lock:raise SystemExit('--lock required')
  lock=json.loads(pathlib.Path(a.lock).read_text())
  if lock.get('fresh_opened') is not False:raise SystemExit('lock indicates fresh already opened')
  cp=pathlib.Path(a.lock).with_name(lock['checkpoint'])
  if sha(cp)!=lock['checkpoint_sha256']:raise SystemExit('checkpoint hash mismatch')
  state=torch.load(cp,map_location='cpu',weights_only=False);ind=MoDLNet('independent');ind.load_state_dict(state['base_independent']);tied=MoDLNet('tied');tied.load_state_dict(state['base_tied']);models=build_models(ind,tied)
  models['independent_ft']=MoDLNet('independent')
  for k,m in models.items():m.load_state_dict(state['models'][k])
  if a.stage=='replay':cases=[make_case(107201+i,'B') for i in range(16)]
  else:cases=[make_case(seed*100+j,'B') for seed in lock['fresh_seeds'] for j in range(lock['fresh_per_seed'])]
  metrics={k:evaluate(m,cases) for k,m in models.items()}
  if a.stage=='replay':
   diff={k:{metric:abs(metrics[k][metric]-lock['metrics'][k][metric]) for metric in ('nmse','psnr_db','global_ssim','measured_residual','cg_relative_residual')} for k in models};result={'stage':'dev_replay','lock_sha256':sha(a.lock),'checkpoint_sha256':lock['checkpoint_sha256'],'metrics':metrics,'differences':diff,'max_abs_difference':max(v for dd in diff.values() for v in dd.values()),'fresh_opened':False}
  else:result={'stage':'fresh_audit','lock_sha256':sha(a.lock),'checkpoint_sha256':lock['checkpoint_sha256'],'fresh_seeds':lock['fresh_seeds'],'metrics':metrics,'weights_or_codes_changed_after_lock':False}
  pathlib.Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
