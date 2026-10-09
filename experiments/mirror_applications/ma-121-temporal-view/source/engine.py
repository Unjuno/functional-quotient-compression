"""Synthetic packet-slot predictor for MA-121."""
import io,math,time
from dataclasses import dataclass
import torch
from torch import nn

SLOTS,D,VOCAB,NTRAIN,NTEST=4,8,8,256,128
METHODS=('tied','ptp_rank2','mirror_phase','mtp','cached_ar')
ANGLES=torch.tensor([-0.45,-0.15,0.15,0.45])

def rot(t):
 c,s=torch.cos(t),torch.sin(t);r=torch.eye(D);r[0,0],r[0,1]=c,-s;r[1,0],r[1,1]=s,c;return r

def make_world(seed,aligned):
 g=torch.Generator().manual_seed(seed);x=torch.randn(NTRAIN,D,generator=g);xt=torch.randn(NTEST,D,generator=g)
 if aligned:
  base=torch.randn(VOCAB,D,generator=g)/math.sqrt(D)
  teachers=torch.stack([base@rot(t) for t in ANGLES])
 else:teachers=torch.randn(SLOTS,VOCAB,D,generator=g)/math.sqrt(D)
 logits=torch.einsum('svd,nd->nsv',teachers,x);logits_test=torch.einsum('svd,nd->nsv',teachers,xt)
 y=torch.argmax(logits*2.5,dim=-1)
 yt=torch.argmax(logits_test,dim=-1)
 return x,y,xt,yt

class PacketModel(nn.Module):
 def __init__(self,method):
  super().__init__();self.method=method
  if method=='mtp':self.weight=nn.Parameter(torch.randn(SLOTS,VOCAB,D)*.05)
  else:self.weight=nn.Parameter(torch.randn(VOCAB,D)*.05)
  if method=='mirror_phase':self.angle=nn.Parameter(torch.zeros(SLOTS))
  if method=='ptp_rank2':
   self.basis=nn.Parameter(torch.randn(2,VOCAB,D)*.02);self.code=nn.Parameter(torch.zeros(SLOTS,2))
  if method=='cached_ar':
   self.slot=nn.Parameter(torch.randn(SLOTS,D)*.02);self.token=nn.Parameter(torch.randn(VOCAB,D)*.02)
 def forward(self,x):
  if self.method=='mtp':return torch.einsum('svd,nd->nsv',self.weight,x)
  if self.method=='cached_ar':return self.generate(x)
  if self.method=='mirror_phase':
   rotated=torch.stack([x@rot(a) for a in self.angle],dim=1)
   return torch.einsum('vd,nsd->nsv',self.weight,rotated)
  if self.method=='ptp_rank2':
   ws=self.weight[None]+torch.einsum('sr,rvd->svd',self.code,self.basis)
   return torch.einsum('svd,nd->nsv',ws,x)
  return torch.einsum('vd,nd->nv',self.weight,x)[:,None,:].expand(-1,SLOTS,-1)
 def teacher_forced(self,x,y):
  if self.method!='cached_ar':return self(x)
  outs=[]
  for s in range(SLOTS):
   h=x+self.slot[s]
   if s: h=h+self.token[y[:,s-1]]
   outs.append(h@self.weight.T)
  return torch.stack(outs,dim=1)
 def generate(self,x):
  if self.method!='cached_ar':return self(x)
  outs=[];prev=None
  for s in range(SLOTS):
   h=x+self.slot[s]
   if prev is not None:h=h+self.token[prev]
   logits=h@self.weight.T;prev=logits.argmax(-1);outs.append(logits)
  return torch.stack(outs,dim=1)

def payload(m):
 b=io.BytesIO();torch.save({'method':m.method,'state_dict':m.state_dict()},b);return b.getvalue()

@dataclass
class Result:
 method:str;seed:int;condition:str;nll:float;joint:float;bytes:int;updates:int;examples:int;macs:int;wall:float;inference_wall:float;throughput:float

def train_one(method,seed,condition,updates=500,lr=.02):
 torch.manual_seed(seed+121);aligned=condition=='aligned';x,y,xt,yt=make_world(seed,aligned);m=PacketModel(method);opt=torch.optim.Adam(m.parameters(),lr=lr);start=time.perf_counter()
 for _ in range(updates):
  logits=m.teacher_forced(x,y);loss=nn.functional.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1));opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 wall=time.perf_counter()-start
 with torch.no_grad():
  logits=m.generate(xt);nll=float(nn.functional.cross_entropy(logits.reshape(-1,VOCAB),yt.reshape(-1)));pred=logits.argmax(-1);joint=float((pred==yt).all(dim=-1).float().mean())
  m.generate(xt);infer_start=time.perf_counter()
  for _ in range(10):m.generate(xt)
  inference_wall=time.perf_counter()-infer_start
 macs=updates*NTRAIN*SLOTS*VOCAB*D*2
 if method=='mirror_phase':macs+=updates*3*2*SLOTS*D**3
 elif method=='ptp_rank2':macs+=updates*NTRAIN*SLOTS*VOCAB*D*2
 return Result(method,seed,condition,nll,joint,len(payload(m)),updates,updates*NTRAIN*SLOTS,int(macs),wall,inference_wall,10*NTEST*SLOTS/inference_wall)

def run(seed,condition,updates=500,lr=.02):return [train_one(m,seed,condition,updates,lr) for m in METHODS]
