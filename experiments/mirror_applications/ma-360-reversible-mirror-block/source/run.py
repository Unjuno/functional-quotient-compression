#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';SEEDS=(36001,36002,36011,36012);D=32;H=D//2;DEPTH=8;BATCH=64;UPDATES=300

class RevStep(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,F,G,scale):
  a,b=x.chunk(2,-1);ap=a+scale*torch.tanh(b@F);bp=b+scale*torch.tanh(ap@G);y=torch.cat([ap,bp],-1);ctx.save_for_backward(y,F,G,scale);return y
 @staticmethod
 def backward(ctx,dy):
  y,F,G,scale=ctx.saved_tensors;ap,bp=y.chunk(2,-1)
  with torch.no_grad():
   b=bp-scale*torch.tanh(ap@G);a=ap-scale*torch.tanh(b@F)
  with torch.enable_grad():
   aa=a.detach().requires_grad_(True);bb=b.detach().requires_grad_(True);ff=F.detach().requires_grad_(True);gg=G.detach().requires_grad_(True);ss=scale.detach().requires_grad_(True)
   aa2=aa+ss*torch.tanh(bb@ff);bb2=bb+ss*torch.tanh(aa2@gg);yy=torch.cat([aa2,bb2],-1)
   grads=torch.autograd.grad(yy,(aa,bb,ff,gg,ss),dy)
  return torch.cat([grads[0],grads[1]],-1),grads[2],grads[3],grads[4]

class Coupling(nn.Module):
 def __init__(self,kind,seed):
  super().__init__();g=torch.Generator().manual_seed(seed)
  if kind=='untied':
   self.f=nn.ParameterList([nn.Parameter(torch.randn(H,H,generator=g)*.08) for _ in range(DEPTH)]);self.g=nn.ParameterList([nn.Parameter(torch.randn(H,H,generator=g)*.08) for _ in range(DEPTH)])
  else:
   self.f=nn.Parameter(torch.randn(H,H,generator=g)*.08);self.g=nn.Parameter(torch.randn(H,H,generator=g)*.08)
  self.kind=kind;self.m=nn.Parameter(torch.zeros(DEPTH)) if kind in ('mirror','direct') else None
 def matrices(self,i):return (self.f[i],self.g[i]) if self.kind=='untied' else (self.f,self.g)
 def step(self,x,i):
  a,b=x.chunk(2,-1);f,g=self.matrices(i);scale=torch.tanh(self.m[i]) if self.m is not None else 1.
  if self.kind=='reversible':return RevStep.apply(x,f,g,torch.as_tensor(scale,dtype=x.dtype,device=x.device))
  ap=a+scale*torch.tanh(b@f);bp=b+scale*torch.tanh(ap@g);return torch.cat([ap,bp],-1)
 def forward(self,x,use_ckpt=False):
  for i in range(DEPTH):
   x=self.step(x,i)
  return x

def world(seed):
 torch.manual_seed(seed);return torch.randn(2048,D),torch.randn(2048,D)
def target(x,seed):
 torch.manual_seed(seed+800);w=Coupling('untied',seed+800);return w(x)
def payload(model):
 b=io.BytesIO();state={k:v.detach().cpu().numpy() for k,v in model.state_dict().items()};np.savez_compressed(b,**state);return b.getvalue()
def saved_bytes(kind,seed,x):
 model=Coupling(kind,seed);track={'current':0,'peak':0};param_ptrs={p.untyped_storage().data_ptr() for p in model.parameters()}
 def pack(t):
  n=0 if t.untyped_storage().data_ptr() in param_ptrs else t.numel()*t.element_size();track['current']+=n;track['peak']=max(track['peak'],track['current']);return (t,n)
 def unpack(z):
  t,n=z;track['current']-=n;return t
 y=target(x,seed);opt=torch.optim.AdamW(model.parameters(),lr=1e-3);start=time.perf_counter()
 with torch.autograd.graph.saved_tensors_hooks(pack,unpack):
  out=model(x[:BATCH],use_ckpt=(kind=='reversible'));loss=((out-y[:BATCH])**2).mean();loss.backward()
 return track['peak'],time.perf_counter()-start
def train(seed,kind):
 x,y=world(seed);m=Coupling(kind,seed);opt=torch.optim.AdamW(m.parameters(),lr=1e-3);start=time.perf_counter()
 for step in range(UPDATES):
  idx=torch.randint(len(x),(BATCH,));opt.zero_grad();out=m(x[idx],use_ckpt=(kind=='reversible'));loss=((out-y[idx])**2).mean();loss.backward();opt.step()
 elapsed=time.perf_counter()-start
 with torch.no_grad():pred=m(x);mse=float(((pred-y)**2).mean())
 p=payload(m);mem,onepass=saved_bytes(kind,seed,x)
 return {'seed':seed,'method':kind,'payload_bytes':len(p),'payload_sha256':hashlib.sha256(p).hexdigest(),'task_mse':mse,'saved_activation_bytes_one_batch':mem,'examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'train_MAC_proxy':UPDATES*BATCH*DEPTH*2*H*H,'train_wall_seconds':elapsed,'onepass_forward_backward_seconds':onepass}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');a=ap.parse_args();rows=[]
 for seed in SEEDS[:2] if a.dev_only else SEEDS:
  for kind in ('untied','tied','direct','mirror','reversible'):
   r=train(seed,kind);rows.append(r);print(json.dumps(r,sort_keys=True))
 p=OUT/('development.csv' if a.dev_only else 'results.csv');OUT.mkdir(exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
