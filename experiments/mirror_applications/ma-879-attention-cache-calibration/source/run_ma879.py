#!/usr/bin/env python3
"""Synthetic causal KV-cache translation screen (MA-879)."""
import argparse,csv,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';METHODS=('cachebridge_weighted_ridge','mirror_raw','mirror_attention','basis2_attention','shared_attention','independent_attention');VARIANTS=('aligned','independent');L=8;D=4
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def rot_right(A,t):
 c=np.cos(t);s=np.sin(t);z=A.copy()
 for i,j in [(0,1),(2,3)]:a=A[:,i].copy();b=A[:,j].copy();z[:,i]=c*a-s*b;z[:,j]=s*a+c*b
 return z
def teacher(w,v):
 r=np.random.default_rng(w+(0 if v=='aligned' else 100000));
 if v=='aligned':
  bk=r.normal(0,.5,(D,D)).astype('float32');bv=r.normal(0,.5,(D,D)).astype('float32');ak=r.uniform(-1,1,4);av=r.uniform(-1,1,4);K=np.stack([rot_right(bk,a) for a in ak]);V=np.stack([rot_right(bv,a) for a in av])
 else:K=r.normal(0,.5,(4,D,D)).astype('float32');V=r.normal(0,.5,(4,D,D)).astype('float32')
 return K,V
def batch(w,v,split,task,n):
 code={'calibration':101,'dev':211,'query':401}[split];variant=0 if v=='aligned' else 10000;r=np.random.default_rng(w*1000003+variant+code+task*97);sk=r.normal(size=(n,L,D)).astype('float32');sv=r.normal(size=(n,L,D)).astype('float32');q=r.normal(size=(n,L,D)).astype('float32');return sk,sv,q
def attend(q,k,v):
 scores=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(D);mask=torch.triu(torch.ones(L,L,dtype=torch.bool),diagonal=1);scores=scores.masked_fill(mask,-1e9);p=torch.softmax(scores,dim=-1);return p,torch.matmul(p,v)
class Translator(torch.nn.Module):
 def __init__(self,m):
  super().__init__();self.method=m
  if m in ('cachebridge_weighted_ridge','independent_attention'):self.k=torch.nn.Parameter(torch.randn(4,D,D)*.1);self.v=torch.nn.Parameter(torch.randn(4,D,D)*.1)
  elif m in ('mirror_raw','mirror_attention'):self.k0=torch.nn.Parameter(torch.randn(D,D)*.1);self.v0=torch.nn.Parameter(torch.randn(D,D)*.1);self.ak=torch.nn.Parameter(torch.zeros(4));self.av=torch.nn.Parameter(torch.zeros(4))
  elif m=='basis2_attention':self.kb=torch.nn.Parameter(torch.randn(2,D,D)*.1);self.vb=torch.nn.Parameter(torch.randn(2,D,D)*.1);self.kc=torch.nn.Parameter(torch.randn(4,2)*.1);self.vc=torch.nn.Parameter(torch.randn(4,2)*.1)
  else:self.k0=torch.nn.Parameter(torch.randn(D,D)*.1);self.v0=torch.nn.Parameter(torch.randn(D,D)*.1)
 def maps(self):
  m=self.method
  if m in ('cachebridge_weighted_ridge','independent_attention'):return self.k,self.v
  if m in ('mirror_raw','mirror_attention'):return torch.stack([self._rot(self.k0,a) for a in self.ak]),torch.stack([self._rot(self.v0,a) for a in self.av])
  if m=='basis2_attention':return torch.einsum('tr,rij->tij',self.kc,self.kb),torch.einsum('tr,rij->tij',self.vc,self.vb)
  return self.k0[None].expand(4,-1,-1),self.v0[None].expand(4,-1,-1)
 def _rot(self,A,a):
  z=A.clone();c=torch.cos(a);s=torch.sin(a)
  for i,j in [(0,1),(2,3)]:z[:,i]=c*A[:,i]-s*A[:,j];z[:,j]=s*A[:,i]+c*A[:,j]
  return z
 def forward(self,sk,sv,t):
  K,V=self.maps();return torch.bmm(sk,K[t]),torch.bmm(sv,V[t])
def targets(sk,sv,q,K,V,t):
 kt=torch.as_tensor(K[t]);vt=torch.as_tensor(V[t]);k=sk@kt;v=sv@vt;p,o=attend(q,k,v);return k,v,p,o
def losses(model,sk,sv,q,K,V,t):
 pk,pv=model(sk,sv,t);tk,tv,tp,to=targets(sk,sv,q,K,V,t);pp,po=attend(q,pk,pv);cache=.5*((pk-tk).square().mean()+(pv-tv).square().mean());out=(po-to).square().mean();kl=(tp*(torch.log(tp.clamp_min(1e-9))-torch.log(pp.clamp_min(1e-9)))).sum(-1).mean();return cache,out,kl,tp,pp,po,to
def ridge(X,Y,w,lam=1e-4):
 x=X.reshape(-1,D);y=Y.reshape(-1,D);wt=w.reshape(-1).clamp_min(1e-5);xt=x.T*wt;return torch.linalg.solve(xt@x+lam*torch.eye(D),xt@y)
def fit_ridge(model,w,variant,seedid):
 K,V=teacher(w,variant)
 with torch.no_grad():
  for t in range(4):
   sk,sv,q=batch(w,variant,'calibration',t,256);sk=torch.tensor(sk);sv=torch.tensor(sv);q=torch.tensor(q);tk,tv,tp,_=targets(sk,sv,q,K,V,t);weights=tp.sum(1)+1e-3;model.k[t].copy_(ridge(sk,tk,weights));model.v[t].copy_(ridge(sv,tv,weights))
 return model
def eval_one(model,w,variant,t,split,n):
 K,V=teacher(w,variant);sk,sv,q=batch(w,variant,split,t,n);sk=torch.tensor(sk);sv=torch.tensor(sv);q=torch.tensor(q);idx=torch.full((n,),t,dtype=torch.long)
 with torch.no_grad():cache,out,kl,tp,pp,po,to=losses(model,sk,sv,q,K,V,idx);target=tp.argmax(-1);nll=-torch.log(pp.gather(-1,target.unsqueeze(-1)).clamp_min(1e-9)).mean();return {'cache_mse':float(cache),'attention_output_mse':float(out),'attention_kl':float(kl),'target_argmax_nll':float(nll)}
def fit(method,w,variant,s,n,dev):
 seed(s);model=Translator(method)
 if method=='cachebridge_weighted_ridge':
  start=time.perf_counter();model=fit_ridge(model,w,variant,s);return model,0,0.,time.perf_counter()-start,[]
 opt=torch.optim.Adam(model.parameters(),lr=.01);rng=np.random.default_rng(w*811+s+(0 if variant=='aligned' else 731));curve=[];best=1e99;bstep=0;state=None;start=time.perf_counter()
 data={t:(batch(w,variant,'calibration',t,256)) for t in range(4)}
 for step in range(1,n+1):
  t=int(rng.integers(0,4));sk,sv,q=data[t];ix=rng.integers(0,len(sk),32);sk=torch.tensor(sk[ix]);sv=torch.tensor(sv[ix]);q=torch.tensor(q[ix]);idx=torch.full((len(ix),),t,dtype=torch.long);cache,out,kl,*_=losses(model,sk,sv,q, *teacher(w,variant),idx)
  loss=cache if method=='mirror_raw' else out+kl
  opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  if dev and step%25==0:
   sc=np.mean([eval_one(model,w,variant,t,'dev',64)['attention_output_mse'] for t in range(4)]);curve.append((step,sc))
   if sc<best:best=sc;bstep=step;state={k:v.detach().clone() for k,v in model.state_dict().items()}
 if dev and state:model.load_state_dict(state)
 return model,bstep,best,time.perf_counter()-start,curve
def payload(model,p):
 sd={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()};flat=torch.cat([x.reshape(-1).float() for _,x in sorted(sd.items())]);layout=';'.join(f'{k}:{list(x.shape)}' for k,x in sorted(sd.items()));save_file({'p':flat},str(p),metadata={'schema':'MA879-V1','method':model.method,'layout':layout});return p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest()
def latency(model):
 sk=torch.randn(1,L,D);sv=torch.randn(1,L,D);t=torch.tensor([0])
 with torch.no_grad():
  for _ in range(30):model(sk,sv,t)
  st=time.perf_counter()
  for _ in range(300):model(sk,sv,t)
 return 1000*(time.perf_counter()-st)/300
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','audit'],required=True);a=ap.parse_args();ART.mkdir(exist_ok=True);worlds=[3,7] if a.phase=='development' else [11,17,23];seeds=[31,47,59];rows=[]
 if a.phase=='development':
  selected={};curves=[]
  for m in METHODS:
   fs=[]
   for w in worlds:
    for v in VARIANTS:
     for s in seeds:
      model,step,best,wall,curve=fit(m,w,v,s,500,m!='cachebridge_weighted_ridge');fs.append((w,v,s,model,step,best,wall,curve))
   if m=='cachebridge_weighted_ridge':chosen=0;mean=0
   else:
    steps=sorted(set(x[4] for x in fs));means={k:np.mean([next(q for st,q in x[7] if st==k) for x in fs]) for k in steps};chosen=min(steps,key=lambda k:means[k]);mean=means[chosen]
   selected[m]={'updates':chosen,'mean_dev_attention_mse':float(mean)}
   for w,v,s,model,step,best,wall,curve in fs:
    n,sha=payload(model,ART/f'dev_{m}_{v}_w{w}_s{s}.safetensors');curves.extend({'method':m,'variant':v,'world':w,'model_seed':s,'updates':st,'attention_mse':q} for st,q in curve);rows.append({'method':m,'variant':v,'world':w,'model_seed':s,'selected_updates':chosen,'best_dev_mse':best,'payload_bytes':n,'payload_sha256':sha,'train_wall_seconds':wall,'cpu_batch1_latency_ms':latency(model)})
  (ROOT/'source/DEVELOPMENT_FREEZE.json').write_text(json.dumps({'selected':selected,'dev_metrics_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()},indent=2)+'\n')
  with (ART/'development_curve.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
 else:
  selected=json.loads((ROOT/'source/DEVELOPMENT_FREEZE.json').read_text())['selected']
  for m in METHODS:
   for w in worlds:
    for v in VARIANTS:
     for s in seeds:
      model,_,_,wall,_=fit(m,w,v,s,selected[m]['updates'],False);n,sha=payload(model,ART/f'audit_{m}_{v}_w{w}_s{s}.safetensors')
      for t in range(4):
       met=eval_one(model,w,v,t,'query',128);params=sum(x.numel() for x in model.parameters());rows.append({'method':m,'variant':v,'world':w,'model_seed':s,'task_id':t,**met,'payload_bytes':n,'payload_sha256':sha,'train_wall_seconds':wall,'cpu_batch1_latency_ms':latency(model),'selected_updates':selected[m]['updates'],'parameter_count':params})
 out=ART/f'metrics_{a.phase}.csv'
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(out)
if __name__=='__main__':main()
