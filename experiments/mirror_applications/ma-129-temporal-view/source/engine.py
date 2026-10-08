"""Synthetic speculative-branch verifier for MA-129."""
import io,math,time
from dataclasses import dataclass
import torch
from torch import nn

BRANCHES,D,NTRAIN,NTEST=4,8,512,256
METHODS=('tied','ptp_rank2','mirror_view','mtp','sequential')
ANGLES=torch.tensor([-.5,-.17,.17,.5])

def rot(t):
 c,s=torch.cos(t),torch.sin(t);r=torch.eye(D);r[0,0],r[0,1]=c,-s;r[1,0],r[1,1]=s,c;return r

def world(seed,aligned):
 g=torch.Generator().manual_seed(seed);x=torch.randn(NTRAIN,D,generator=g);xt=torch.randn(NTEST,D,generator=g)
 if aligned:
  base=torch.randn(D,generator=g);ws=torch.stack([rot(a)@base for a in ANGLES])
 else:ws=torch.randn(BRANCHES,D,generator=g)
 y=torch.einsum('sd,nd->ns',ws,x)>0;yt=torch.einsum('sd,nd->ns',ws,xt)>0
 return x,y.float(),xt,yt.float()

class Verifier(nn.Module):
 def __init__(self,method):
  super().__init__();self.method=method
  if method=='mtp':self.weight=nn.Parameter(torch.randn(BRANCHES,D)*.05)
  else:self.weight=nn.Parameter(torch.randn(D)*.05)
  if method=='mirror_view':self.angle=nn.Parameter(torch.zeros(BRANCHES))
  if method=='ptp_rank2':self.basis=nn.Parameter(torch.randn(2,D)*.02);self.code=nn.Parameter(torch.zeros(BRANCHES,2))
 def forward(self,x):
  if self.method=='mtp':return x@self.weight.T
  if self.method=='mirror_view':
   xs=torch.stack([x@rot(a) for a in self.angle],dim=1)
   return torch.einsum('nsd,d->ns',xs,self.weight)
  if self.method=='ptp_rank2':
   ws=self.weight[None]+self.code@self.basis
   return torch.einsum('sd,nd->ns',ws,x)
  if self.method=='sequential':
   return torch.stack([x@self.weight for _ in range(BRANCHES)],dim=1)
  return (x@self.weight)[:,None].expand(-1,BRANCHES)

def payload(m):
 b=io.BytesIO();torch.save({'method':m.method,'state_dict':m.state_dict()},b);return b.getvalue()

@dataclass
class Result:
 method:str;seed:int;condition:str;nll:float;path_acc:float;bytes:int;updates:int;examples:int;macs:int;train_wall:float;infer_wall:float;branches_s:float

def train_one(method,seed,condition,updates=500,lr=.02):
 torch.manual_seed(seed+129);x,y,xt,yt=world(seed,condition=='aligned');m=Verifier(method);opt=torch.optim.Adam(m.parameters(),lr=lr);start=time.perf_counter()
 for _ in range(updates):
  logits=m(x);loss=nn.functional.binary_cross_entropy_with_logits(logits,y);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 train_wall=time.perf_counter()-start
 with torch.no_grad():
  logits=m(xt);nll=float(nn.functional.binary_cross_entropy_with_logits(logits,yt));pred=logits>0;path=float((pred==(yt>0.5)).all(dim=1).float().mean())
  m(xt);st=time.perf_counter()
  for _ in range(200):m(xt)
  infer_wall=time.perf_counter()-st
 macs=updates*NTRAIN*BRANCHES*D*2
 if method=='mirror_view':macs+=updates*3*2*BRANCHES*D**3
 elif method=='ptp_rank2':macs+=updates*BRANCHES*D*2
 return Result(method,seed,condition,nll,path,len(payload(m)),updates,updates*NTRAIN,int(macs),train_wall,infer_wall,200*NTEST*BRANCHES/infer_wall)

def run(seed,condition,updates=500,lr=.02):return [train_one(m,seed,condition,updates,lr) for m in METHODS]
