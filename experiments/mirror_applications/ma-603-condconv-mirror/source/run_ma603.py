#!/usr/bin/env python3
"""Synthetic input-conditioned dynamic-weight benchmark for MA-603."""
import argparse,csv,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; METHODS=('mirror_givens','condconv4','film8','full_dynamic8','static_linear')
def seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def world(s):
 r=np.random.default_rng(s);return r.normal(0,1/math.sqrt(8),(8,8)).astype('float32'),r.normal(0,.05,8).astype('float32')
def data(w,split,n):return np.random.default_rng(w*1000003+{'train':101,'dev':211,'query':401}[split]).normal(size=(n,8)).astype('float32')
def teacher(x,W,b):
 t=.8*np.tanh(x[:,0])+.6*np.tanh(x[:,1]);h=x@W.T+b;o=h.copy();c=np.cos(t);s=np.sin(t)
 for j in range(0,8,2):o[:,j]=c*h[:,j]-s*h[:,j+1];o[:,j+1]=s*h[:,j]+c*h[:,j+1]
 return o.astype('float32')
class Net(torch.nn.Module):
 def __init__(self,m):
  super().__init__();self.method=m
  if m=='mirror_givens':self.W=torch.nn.Parameter(torch.randn(8,8)*.2);self.b=torch.nn.Parameter(torch.zeros(8));self.c=torch.nn.Sequential(torch.nn.Linear(2,16),torch.nn.Tanh(),torch.nn.Linear(16,1))
  elif m=='condconv4':self.W=torch.nn.Parameter(torch.randn(4,8,8)*.2);self.b=torch.nn.Parameter(torch.zeros(4,8));self.g=torch.nn.Sequential(torch.nn.Linear(2,16),torch.nn.Tanh(),torch.nn.Linear(16,4))
  elif m=='film8':self.W=torch.nn.Parameter(torch.randn(8,8)*.2);self.b=torch.nn.Parameter(torch.zeros(8));self.c=torch.nn.Sequential(torch.nn.Linear(2,16),torch.nn.Tanh(),torch.nn.Linear(16,16))
  elif m=='full_dynamic8':self.W=torch.nn.Parameter(torch.randn(8,8)*.2);self.b=torch.nn.Parameter(torch.zeros(8));self.c=torch.nn.Sequential(torch.nn.Linear(2,16),torch.nn.Tanh(),torch.nn.Linear(16,72))
  else:self.W=torch.nn.Parameter(torch.randn(8,8)*.2);self.b=torch.nn.Parameter(torch.zeros(8))
 def forward(self,x):
  u=x[:,:2];m=self.method
  if m=='mirror_givens':
   h=x@self.W.T+self.b;t=self.c(u).squeeze(-1);c=torch.cos(t);s=torch.sin(t);o=h.clone()
   for j in range(0,8,2):o[:,j]=c*h[:,j]-s*h[:,j+1];o[:,j+1]=s*h[:,j]+c*h[:,j+1]
   return o
  if m=='condconv4':return torch.einsum('bk,bko->bo',torch.softmax(self.g(u),-1),torch.einsum('koi,bi->bko',self.W,x)+self.b[None])
  h=x@self.W.T+self.b
  if m=='film8':z=self.c(u);return h*(1+z[:,:8])+z[:,8:]
  if m=='full_dynamic8':z=self.c(u);return h+torch.bmm(z[:,:64].view(-1,8,8),x[:,:,None]).squeeze(-1)+z[:,64:]
  return h
def mse(net,x,y):
 with torch.no_grad():return float(((net(torch.as_tensor(x))-torch.as_tensor(y))**2).mean())
def fit(m,w,s,n,dev):
 seed(s);W,b=world(w);net=Net(m);opt=torch.optim.Adam(net.parameters(),lr=.003);x=data(w,'train',4096);y=teacher(x,W,b);dx=data(w,'dev',2048);dy=teacher(dx,W,b);rng=np.random.default_rng(w*997+s);best=1e99;state=None;stepbest=0;curve=[];t=time.perf_counter()
 for step in range(1,n+1):
  ix=rng.integers(len(x),size=256);xb=torch.as_tensor(x[ix]);yb=torch.as_tensor(y[ix]);loss=torch.nn.functional.mse_loss(net(xb),yb);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  if dev and step%50==0:
   v=mse(net,dx,dy);curve.append((step,v))
   if v<best:best=v;stepbest=step;state={k:v.detach().clone() for k,v in net.state_dict().items()}
 if dev and state:net.load_state_dict(state)
 return net,stepbest,best,time.perf_counter()-t,curve
def payload(net,p):
 sd={k:v.detach().cpu().contiguous() for k,v in net.state_dict().items()};flat=torch.cat([v.reshape(-1).float() for k,v in sorted(sd.items())]);layout=';'.join(f'{k}:{list(v.shape)}' for k,v in sorted(sd.items()));save_file({'p':flat},str(p),metadata={'schema':'MA603-V1','method':net.method,'layout':layout});return p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest()
def lat(net):
 x=torch.randn(1,8)
 with torch.no_grad():
  for _ in range(50):net(x)
  t=time.perf_counter()
  for _ in range(500):net(x)
 return 1000*(time.perf_counter()-t)/500
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','audit'],required=True);a=p.parse_args();ART.mkdir(exist_ok=True);ws=[3,5] if a.phase=='development' else [11,17,23];ss=[31,47,59];rows=[]
 if a.phase=='development':
  selected={};curves=[]
  for m in METHODS:
   fs=[(w,s,*fit(m,w,s,1500,True)) for w in ws for s in ss];steps=sorted(set(q[3] for q in fs));means={k:np.mean([next(v for st,v in q[6] if st==k) for q in fs]) for k in steps};choice=min(steps,key=lambda k:means[k]);selected[m]={'updates':choice,'mean_dev_mse':float(means[choice])}
   for w,s,net,bstep,best,wall,curve in fs:
    n,sha=payload(net,ART/f'dev_{m}_w{w}_s{s}.safetensors');curves.extend({'method':m,'world_seed':w,'model_seed':s,'updates':st,'dev_mse':v} for st,v in curve);rows.append({'method':m,'world_seed':w,'model_seed':s,'selected_updates':choice,'best_dev_mse':best,'train_wall_seconds':wall,'payload_bytes':n,'payload_sha256':sha,'cpu_batch1_latency_ms':lat(net)})
  blob={'selected':selected,'selection':'mean dev MSE over two development worlds and three init seeds','metrics_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()};(ROOT/'source/DEVELOPMENT_FREEZE.json').write_text(json.dumps(blob,indent=2)+'\n')
  with (ART/'development_curve.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
 else:
  selected=json.loads((ROOT/'source/DEVELOPMENT_FREEZE.json').read_text())['selected']
  for m in METHODS:
   for w in ws:
    W,b=world(w);xq=data(w,'query',8192);yq=teacher(xq,W,b)
    for s in ss:
     net,_,_,wall,_=fit(m,w,s,selected[m]['updates'],False);q=mse(net,xq,yq);n,sha=payload(net,ART/f'audit_{m}_w{w}_s{s}.safetensors');rows.append({'method':m,'world_seed':w,'model_seed':s,'selected_updates':selected[m]['updates'],'train_wall_seconds':wall,'payload_bytes':n,'payload_sha256':sha,'cpu_batch1_latency_ms':lat(net),'query_mse':q})
 out=ART/f'metrics_{a.phase}.csv'
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(out)
if __name__=='__main__':main()
