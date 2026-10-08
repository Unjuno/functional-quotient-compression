#!/usr/bin/env python3
"""Synthetic composable control-adapter screen for MA-753."""
import argparse,csv,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';METHODS=('independent_adapters','mirror_givens','basis_mix2','film_controls','hard_shared');VARIANTS=('aligned','independent')
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def teacher(seed_id,variant):
 r=np.random.default_rng(seed_id+ (0 if variant=='aligned' else 100000));p=r.normal(0,1/math.sqrt(8),(8,8)).astype('float32')
 if variant=='aligned':
  angles=r.uniform(-1,1,4); mats=np.stack([rotate(p,a) for a in angles])
 else:mats=r.normal(0,1/math.sqrt(8),(4,8,8)).astype('float32')
 return mats.astype('float32')
def rotate(p,theta):
 z=p.copy();c=np.cos(theta);s=np.sin(theta)
 for j in range(0,8,2):a=p[j].copy();b=p[j+1].copy();z[j]=c*a-s*b;z[j+1]=s*a+c*b
 return z
def samples(world,variant,split,n):
 code={'train':101,'dev':211,'audit':401}[split];v=0 if variant=='aligned' else 10000;r=np.random.default_rng(world*1000003+v+code);x=r.normal(size=(n,8)).astype('float32');c=r.integers(0,4,size=n);return x,c
class Adapter(torch.nn.Module):
 def __init__(self,method):
  super().__init__();self.method=method
  if method=='independent_adapters':self.delta=torch.nn.Parameter(torch.randn(4,8,8)*.1)
  elif method=='mirror_givens':self.shared=torch.nn.Parameter(torch.randn(8,8)*.1);self.angles=torch.nn.Parameter(torch.zeros(4))
  elif method=='basis_mix2':self.bases=torch.nn.Parameter(torch.randn(2,8,8)*.1);self.codes=torch.nn.Parameter(torch.randn(4,2)*.1)
  elif method=='film_controls':self.shared=torch.nn.Parameter(torch.randn(8,8)*.1);self.scale=torch.nn.Parameter(torch.zeros(4,8));self.bias=torch.nn.Parameter(torch.zeros(4,8))
  else:self.shared=torch.nn.Parameter(torch.randn(8,8)*.1)
 def matrices(self):
  m=self.method
  if m=='independent_adapters':return self.delta
  if m=='mirror_givens':return torch.stack([torch.as_tensor(rotate(self.shared.detach().numpy(),0))]) if False else self._mirror_mats()
  if m=='basis_mix2':return torch.einsum('cr,rij->cij',self.codes,self.bases)
  if m=='film_controls':return self.shared[None,:,:]*(1+self.scale[:,:,None])
  return self.shared[None,:,:].expand(4,-1,-1)
 def _mirror_mats(self):
  outs=[]
  for a in self.angles:
   rows=self.shared;z=rows.clone();c=torch.cos(a);s=torch.sin(a)
   for j in range(0,8,2):z[j]=c*rows[j]-s*rows[j+1];z[j+1]=s*rows[j]+c*rows[j+1]
   outs.append(z)
  return torch.stack(outs)
 def forward(self,x,c):
  mats=self.matrices();sel=mats[c]
  y=torch.bmm(sel,x.unsqueeze(-1)).squeeze(-1)
  if self.method=='film_controls':y=y+self.bias[c]
  return y
def mse(a,b):return float(np.mean((a-b)**2))
def eval_data(model,mats,world,variant,split='dev',n=1024):
 x,c=samples(world,variant,split,n);target=np.einsum('nij,nj->ni',mats[c],x).astype('float32')
 with torch.no_grad():pred=model(torch.as_tensor(x),torch.as_tensor(c,dtype=torch.long)).numpy()
 return mse(pred,target)
def fit(method,world,variant,modelseed,updates,development):
 seed(modelseed);mats=teacher(world,variant);model=Adapter(method);opt=torch.optim.Adam(model.parameters(),lr=.03);x,c=samples(world,variant,'train',4096);y=np.einsum('nij,nj->ni',mats[c],x).astype('float32');rng=np.random.default_rng(world*1009+modelseed+(0 if variant=='aligned' else 191));best=1e99;beststep=0;state=None;curve=[];start=time.perf_counter()
 for step in range(1,updates+1):
  ix=rng.integers(0,len(x),256);xb=torch.as_tensor(x[ix]);cb=torch.as_tensor(c[ix],dtype=torch.long);yb=torch.as_tensor(y[ix]);loss=torch.nn.functional.mse_loss(model(xb,cb),yb);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  if development and step%25==0:
   score=eval_data(model,mats,world,variant,'dev',1024);curve.append((step,score))
   if score<best:best=score;beststep=step;state={k:v.detach().clone() for k,v in model.state_dict().items()}
 if development and state:model.load_state_dict(state)
 return model,beststep,best,time.perf_counter()-start,curve,mats
def payload(model,path):
 sd={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()};flat=torch.cat([v.reshape(-1).float() for k,v in sorted(sd.items())]);layout=';'.join(f'{k}:{list(v.shape)}' for k,v in sorted(sd.items()));save_file({'p':flat},str(path),metadata={'schema':'MA753-V1','method':model.method,'layout':layout});return path.stat().st_size,hashlib.sha256(path.read_bytes()).hexdigest()
def latency(model):
 x=torch.randn(1,8);c=torch.tensor([0])
 with torch.no_grad():
  for _ in range(40):model(x,c)
  t=time.perf_counter()
  for _ in range(500):model(x,c)
 return 1000*(time.perf_counter()-t)/500
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','audit'],required=True);a=ap.parse_args();ART.mkdir(exist_ok=True);worlds=[2,5] if a.phase=='development' else [11,17,23];modelseeds=[29,41,53];rows=[];curves=[]
 if a.phase=='development':
  selected={}
  for method in METHODS:
   fits=[(w,v,s,*fit(method,w,v,s,1000,True)) for w in worlds for v in VARIANTS for s in modelseeds];steps=sorted(set(z[4] for z in fits));mean={k:np.mean([next(q for st,q in z[7] if st==k) for z in fits]) for k in steps};choice=min(steps,key=lambda k:mean[k]);selected[method]={'updates':choice,'mean_dev_mse':float(mean[choice])}
   for w,v,s,model,bstep,best,wall,curve,mats in fits:
    n,sha=payload(model,ART/f'dev_{method}_{v}_w{w}_s{s}.safetensors');curves.extend({'method':method,'variant':v,'world':w,'model_seed':s,'updates':st,'dev_mse':q} for st,q in curve);rows.append({'method':method,'variant':v,'world':w,'model_seed':s,'selected_updates':choice,'best_dev_mse':best,'payload_bytes':n,'payload_sha256':sha,'train_wall_seconds':wall,'cpu_batch1_latency_ms':latency(model)})
  (ROOT/'source/DEVELOPMENT_FREEZE.json').write_text(json.dumps({'selected':selected,'selection':'mean single-control dev MSE pooled over two development worlds, both teacher variants and three model seeds','dev_metrics_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()},indent=2)+'\n')
  with (ART/'development_curve.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
 else:
  selected=json.loads((ROOT/'source/DEVELOPMENT_FREEZE.json').read_text())['selected']
  for method in METHODS:
   for world in worlds:
    for variant in VARIANTS:
     mats=teacher(world,variant);xq=np.random.default_rng(world*1000003+(0 if variant=='aligned' else 10000)+401).normal(size=(2048,8)).astype('float32')
     for s in modelseeds:
      model,_,_,wall,_,_=fit(method,world,variant,s,selected[method]['updates'],False);n,sha=payload(model,ART/f'audit_{method}_{variant}_w{world}_s{s}.safetensors');single=[]
      with torch.no_grad():
       for c in range(4):single.append(model(torch.as_tensor(xq),torch.full((len(xq),),c,dtype=torch.long)).numpy())
      singles=np.stack(single);single_mse=[mse(singles[c],np.einsum('ij,nj->ni',mats[c],xq)) for c in range(4)]
      for i in range(4):
       for j in range(i+1,4):
        pred=singles[i]+singles[j];target=np.einsum('ij,nj->ni',mats[i]+mats[j],xq);rows.append({'method':method,'variant':variant,'world':world,'model_seed':s,'controls':f'{i}+{j}','single_mse_mean':float(np.mean(single_mse)),'composition_mse':mse(pred,target),'composition_linearity_error':mse(pred,singles[i]+singles[j]),'payload_bytes':n,'payload_sha256':sha,'train_wall_seconds':wall,'cpu_batch1_latency_ms':latency(model),'selected_updates':selected[method]['updates']})
 out=ART/f'metrics_{a.phase}.csv'
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(out)
if __name__=='__main__':main()
