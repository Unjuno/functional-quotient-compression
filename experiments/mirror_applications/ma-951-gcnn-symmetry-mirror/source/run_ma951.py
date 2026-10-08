#!/usr/bin/env python3
"""Finite C4 operator screen for task-specific symmetry exceptions (MA-951)."""
import argparse,csv,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';METHODS=('equivariant_only','mirror_exception','basis2_exception','independent_maps','hard_shared_residual');VARIANTS=('aligned','independent')
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def group_matrix(kernel):
 return np.array([[kernel[(i-j)%4] for j in range(4)] for i in range(4)],dtype='float32')
def rotate(M,theta):
 c=np.cos(theta);s=np.sin(theta);z=M.copy()
 for i,j in [(0,1),(2,3)]:a=M[i].copy();b=M[j].copy();z[i]=c*a-s*b;z[j]=s*a+c*b
 return z
def teacher(world,variant):
 r=np.random.default_rng(world+(0 if variant=='aligned' else 100000));C=group_matrix(r.normal(0,.45,4).astype('float32'))
 if variant=='aligned':
  B=r.normal(0,.45,(4,4)).astype('float32');angles=r.uniform(-1,1,4);D=np.stack([rotate(B,t) for t in angles])
 else:D=r.normal(0,.45,(4,4,4)).astype('float32')
 return C,D
def samples(world,variant,split,task,n):
 code={'train':101,'dev':211,'audit':401}[split];v=0 if variant=='aligned' else 10000;r=np.random.default_rng(world*1000003+v+code+task*101);return r.normal(size=(n,4)).astype('float32')
class Model(torch.nn.Module):
 def __init__(self,method):
  super().__init__();self.method=method
  if method=='equivariant_only':self.kernel=torch.nn.Parameter(torch.randn(4)*.1)
  elif method=='mirror_exception':self.kernel=torch.nn.Parameter(torch.randn(4)*.1);self.shared=torch.nn.Parameter(torch.randn(4,4)*.1);self.angles=torch.nn.Parameter(torch.zeros(4))
  elif method=='basis2_exception':self.kernel=torch.nn.Parameter(torch.randn(4)*.1);self.bases=torch.nn.Parameter(torch.randn(2,4,4)*.1);self.codes=torch.nn.Parameter(torch.randn(4,2)*.1)
  elif method=='independent_maps':self.maps_param=torch.nn.Parameter(torch.randn(4,4,4)*.1)
  elif method=='hard_shared_residual':self.kernel=torch.nn.Parameter(torch.randn(4)*.1);self.shared=torch.nn.Parameter(torch.randn(4,4)*.1)
 def matrices(self):
  if self.method=='independent_maps':return self.maps_param
  idx=torch.tensor([[(i-j)%4 for j in range(4)] for i in range(4)],device=(self.kernel.device if hasattr(self,'kernel') else self.maps_param.device))
  C=self.kernel[idx] if hasattr(self,'kernel') else None
  m=self.method
  if m=='equivariant_only':return C[None].expand(4,-1,-1)
  if m=='mirror_exception':
   outs=[]
   for a in self.angles:
    B=self.shared;z=B.clone();c=torch.cos(a);s=torch.sin(a)
    for i,j in [(0,1),(2,3)]:z[i]=c*B[i]-s*B[j];z[j]=s*B[i]+c*B[j]
    outs.append(z)
   R=torch.stack(outs)
  elif m=='basis2_exception':R=torch.einsum('tr,rij->tij',self.codes,self.bases)
  else:R=self.shared[None].expand(4,-1,-1)
  return C[None]+R
 def forward(self,x,t):
  M=self.matrices()[t];return torch.bmm(M,x.unsqueeze(-1)).squeeze(-1)
def mse(a,b):return float(np.mean((a-b)**2))
def score(model,C,D,world,variant,split,n=1024):
 vals=[]
 for t in range(4):
  x=samples(world,variant,split,t,n);target=x@(C+D[t]).T
  with torch.no_grad():pred=model(torch.as_tensor(x),torch.full((n,),t,dtype=torch.long)).numpy()
  vals.append((mse(pred,target),float(np.mean(np.argmax(pred,1)==np.argmax(target,1)))))
 return vals
def fit(method,world,variant,modelseed,updates,development):
 seed(modelseed);C,D=teacher(world,variant);model=Model(method);opt=torch.optim.Adam(model.parameters(),lr=.02);xs=[samples(world,variant,'train',t,2048) for t in range(4)];ys=[x@(C+D[t]).T for t,x in enumerate(xs)];rng=np.random.default_rng(world*991+modelseed+(0 if variant=='aligned' else 401));best=1e99;beststep=0;state=None;curve=[];start=time.perf_counter()
 for step in range(1,updates+1):
  t=int(rng.integers(0,4));ix=rng.integers(0,len(xs[t]),256);xb=torch.as_tensor(xs[t][ix]);yb=torch.as_tensor(ys[t][ix]);loss=torch.nn.functional.mse_loss(model(xb,torch.full((len(ix),),t,dtype=torch.long)),yb);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  if development and step%25==0:
   sc=score(model,C,D,world,variant,'dev');v=float(np.mean([q[0] for q in sc]));curve.append((step,v))
   if v<best:best=v;beststep=step;state={k:v.detach().clone() for k,v in model.state_dict().items()}
 if development and state:model.load_state_dict(state)
 return model,beststep,best,time.perf_counter()-start,curve,C,D
def payload(model,path):
 sd={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()};flat=torch.cat([v.reshape(-1).float() for k,v in sorted(sd.items())]);layout=';'.join(f'{k}:{list(v.shape)}' for k,v in sorted(sd.items()));save_file({'p':flat},str(path),metadata={'schema':'MA951-V1','method':model.method,'layout':layout});return path.stat().st_size,hashlib.sha256(path.read_bytes()).hexdigest()
def equiv_error(model,world,variant,n=512):
 C,D=teacher(world,variant);errs=[];x=np.random.default_rng(world*7+3).normal(size=(n,4)).astype('float32')
 with torch.no_grad():
  for t in range(4):
   ti=torch.full((n,),t,dtype=torch.long);base=model(torch.as_tensor(x),ti).numpy()
   for k in range(4):
    xr=np.roll(x,k,axis=1);pred=model(torch.as_tensor(xr),ti).numpy();want=np.roll(base,k,axis=1);errs.append(mse(pred,want))
 return float(np.mean(errs))
def latency(model):
 x=torch.randn(1,4);t=torch.tensor([0])
 with torch.no_grad():
  for _ in range(30):model(x,t)
  start=time.perf_counter()
  for _ in range(500):model(x,t)
 return 1000*(time.perf_counter()-start)/500
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','audit'],required=True);a=p.parse_args();ART.mkdir(exist_ok=True);worlds=[7,13] if a.phase=='development' else [19,29,43];seeds=[31,47,59];rows=[]
 if a.phase=='development':
  selected={};curves=[]
  for method in METHODS:
   fs=[(w,v,s,*fit(method,w,v,s,1000,True)) for w in worlds for v in VARIANTS for s in seeds];steps=sorted(set(z[4] for z in fs));means={k:np.mean([next(q for st,q in z[7] if st==k) for z in fs]) for k in steps};chosen=min(steps,key=lambda k:means[k]);selected[method]={'updates':chosen,'mean_dev_mse':float(means[chosen])}
   for w,v,s,model,bstep,best,wall,curve,C,D in fs:
    n,sha=payload(model,ART/f'dev_{method}_{v}_w{w}_s{s}.safetensors');curves.extend({'method':method,'variant':v,'world':w,'model_seed':s,'updates':st,'dev_mse':q} for st,q in curve);rows.append({'method':method,'variant':v,'world':w,'model_seed':s,'selected_updates':chosen,'best_dev_mse':best,'payload_bytes':n,'payload_sha256':sha,'train_wall_seconds':wall,'cpu_batch1_latency_ms':latency(model)})
  (ROOT/'source/DEVELOPMENT_FREEZE.json').write_text(json.dumps({'selected':selected,'selection':'minimum pooled single-task dev MSE over teacher variants/worlds/seeds','dev_metrics_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()},indent=2)+'\n')
  with (ART/'development_curve.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
 else:
  selected=json.loads((ROOT/'source/DEVELOPMENT_FREEZE.json').read_text())['selected']
  for method in METHODS:
   for world in worlds:
    for variant in VARIANTS:
     for s in seeds:
      model,_,_,wall,_,C,D=fit(method,world,variant,s,selected[method]['updates'],False);n,sha=payload(model,ART/f'audit_{method}_{variant}_w{world}_s{s}.safetensors');scores=score(model,C,D,world,variant,'audit',2048);eq=equiv_error(model,world,variant);macs={'equivariant_only':16,'mirror_exception':40,'basis2_exception':52,'independent_maps':16,'hard_shared_residual':32}[method];params={'equivariant_only':4,'mirror_exception':24,'basis2_exception':44,'independent_maps':64,'hard_shared_residual':20}[method]
      for t,(q,acc) in enumerate(scores):rows.append({'method':method,'variant':variant,'world':world,'model_seed':s,'task_id':t,'query_mse':q,'argmax_accuracy':acc,'c4_equivariance_error':eq,'payload_bytes':n,'payload_sha256':sha,'train_wall_seconds':wall,'cpu_batch1_latency_ms':latency(model),'selected_updates':selected[method]['updates'],'parameter_count':params,'analytic_active_macs_per_example':macs})
 out=ART/f'metrics_{a.phase}.csv'
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(out)
if __name__=='__main__':main()
