#!/usr/bin/env python3
import argparse,hashlib,io,json,math,random,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';METHODS=['shared','mirror_time','linear_time','discrete_depth'];DEV=[42500,42501];FRESH=[42510,42511,42512];SEEDS=[0,1,2];LRS=[.003,.01];D=2;UPDATES,BATCH=400,256;OMEGA=.7;STEPS=16
def fixseed(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def rot(a):
 c=a.cos();s=a.sin();return torch.stack([torch.stack([c,-s]),torch.stack([s,c])])
def world(w):
 g=torch.Generator().manual_seed(w);q=rot(torch.rand((),generator=g)*math.pi);a0=q@torch.diag(torch.tensor([-.4,-1.1]))@q.T
 return a0
def true_A(a0,t):
 t=torch.as_tensor(t)
 if t.ndim==0:q=rot(OMEGA*t);return q@a0@q.T
 theta=OMEGA*t;c=theta.cos();s=theta.sin();q=torch.stack([torch.stack([c,-s],-1),torch.stack([s,c],-1)],-2);return q@a0.expand(len(t),-1,-1)@q.transpose(-1,-2)
def data(w,n=4096):
 g=torch.Generator().manual_seed(w+222);a0=world(w);x=torch.randn(n,D,generator=g);t=torch.rand(n,generator=g);dx=torch.bmm(true_A(a0,t),x.unsqueeze(-1)).squeeze(-1);return x,t,dx,a0
class Model(nn.Module):
 def __init__(self,m):
  super().__init__();self.m=m
  if m=='discrete_depth':self.a=nn.Parameter(torch.randn(4,D,D)*.1)
  else:self.a=nn.Parameter(torch.randn(D,D)*.1)
  if m=='mirror_time':self.omega=nn.Parameter(torch.tensor(.2))
  if m=='linear_time':self.b=nn.Parameter(torch.zeros(D,D))
 def matrix(self,t):
  if self.m=='discrete_depth':return self.a[(t.clamp(0,.9999)*4).long()]
  if self.m=='mirror_time':
   theta=self.omega*t;c=theta.cos();s=theta.sin();q=torch.stack([torch.stack([c,-s],-1),torch.stack([s,c],-1)],-2);return q@self.a.expand(len(t),-1,-1)@q.transpose(-1,-2)
  if self.m=='linear_time':return self.a[None]+t[:,None,None]*self.b[None]
  return self.a[None].expand(len(t),-1,-1)
 def forward(self,x,t):return torch.bmm(self.matrix(t),x.unsqueeze(-1)).squeeze(-1)
def field(model,x,t):return model(x,torch.full((len(x),),t,dtype=x.dtype))
def integrate(fun,x0,horizon):
 dt=horizon/STEPS;x=x0
 for i in range(STEPS):
  t=i*dt;k1=fun(x,t);k2=fun(x+dt*k1/2,t+dt/2);k3=fun(x+dt*k2/2,t+dt/2);k4=fun(x+dt*k3,t+dt);x=x+dt*(k1+2*k2+2*k3+k4)/6
 return x
def serialize(m):
 b=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},b,_use_new_zipfile_serialization=True);return b.getvalue()
def run(m,w,s,lr,split):
 fixseed(w*100+s);x,t,y,a0=data(w);model=Model(m);opt=torch.optim.AdamW(model.parameters(),lr=lr);st=time.perf_counter()
 for _ in range(UPDATES):
  ix=torch.randint(len(x),(BATCH,));loss=((model(x[ix],t[ix])-y[ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
 wall=time.perf_counter()-st
 g=torch.Generator().manual_seed(w+500);x0=torch.randn(128,D,generator=g)
 def teacher(z,tt):return z@true_A(a0,tt).T
 def pred(z,tt):return field(model,z,tt)
 with torch.no_grad():
  endpoints={};errs={}
  for h in (1.,2.):
   target=integrate(teacher,x0,h);out=integrate(pred,x0,h);endpoints[str(int(h))]=((out-target).square().mean().sqrt()/(target.square().mean().sqrt()+1e-9)).item()
  vg=torch.Generator().manual_seed(w+1000);valx=torch.randn(512,D,generator=vg);valt=torch.rand(512,generator=vg);valy=torch.bmm(true_A(a0,valt),valx.unsqueeze(-1)).squeeze(-1);mask=valt<=1;deriv=((model(valx[mask],valt[mask])-valy[mask]).square().mean().sqrt()/(valy[mask].square().mean().sqrt()+1e-9)).item()
  stiff=[]
  for tt in torch.linspace(0,2,17):
   mats=model.matrix(tt.repeat(1))[0];ev=torch.linalg.eigvals(mats).abs();stiff.append((ev.max()/ev.min().clamp_min(1e-9)).item())
 raw=serialize(model);name=f'{split}_{w}_{s}_{m}_lr{lr:g}.pt';(OUT/'payloads'/name).write_bytes(raw);cl=Model(m);cl.load_state_dict(torch.load(io.BytesIO(raw),weights_only=True));rb=serialize(cl);assert raw==rb
 return {'split':split,'world':w,'seed':s,'method':m,'lr':lr,'payload_file':name,'serialized_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'roundtrip_sha256':hashlib.sha256(rb).hexdigest(),'examples':UPDATES*BATCH,'updates':UPDATES,'active_mac_proxy_per_state':4,'solver_nfe_rk4_16':4*STEPS,'wall_time_s':round(wall,6),'validation_derivative_nrmse':deriv,'endpoint_nrmse_h1':endpoints['1'],'endpoint_nrmse_h2':endpoints['2'],'stiffness_ratio_max_0_2':max(stiff)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);(OUT/'payloads').mkdir(exist_ok=True);rows=[];sel={}
 if a.phase=='dev':
  for m in METHODS:
   sc={lr:[] for lr in LRS}
   for lr in LRS:
    for w in DEV:
     for s in SEEDS:r=run(m,w,s,lr,'development');rows.append(r);sc[lr].append(r['validation_derivative_nrmse'])
   sel[m]=min(LRS,key=lambda lr:sum(sc[lr])/len(sc[lr]))
  (OUT/'development_selection.json').write_text(json.dumps(sel,indent=2)+'\n');mode='w'
 else:
  sel=json.loads((OUT/'development_selection.json').read_text())
  for m in METHODS:
   for w in FRESH:
    for s in SEEDS:rows.append(run(m,w,s,sel[m],'fresh'))
  mode='a'
 with (OUT/'runs.jsonl').open(mode) as f:
  for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
 print(json.dumps({'phase':a.phase,'rows':len(rows),'selection':sel},indent=2))
if __name__=='__main__':main()
