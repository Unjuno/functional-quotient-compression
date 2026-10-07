"""Layer-group tying and Mirror-view screen for MA-086."""
import io,math,time
from dataclasses import dataclass
import torch
from torch import nn

DEPTH,WIDTH,N=8,8,256
METHODS=('tie2','lora2','mirror2','tie4','mirror4','untied')

def rot(t):
 c,s=torch.cos(t),torch.sin(t);r=torch.eye(WIDTH);r[0,0],r[0,1]=c,-s;r[1,0],r[1,1]=s,c;return r

def make_world(seed,aligned):
 g=torch.Generator().manual_seed(seed);x=torch.randn(N,WIDTH,generator=g);xt=torch.randn(128,WIDTH,generator=g)
 if aligned:
  bases=torch.randn(4,WIDTH,WIDTH,generator=g)/math.sqrt(WIDTH); mats=[]
  for d in range(DEPTH):
   local=d%2;theta=torch.tensor((-.35 if local==0 else .35))
   mats.append(rot(theta)@bases[d//2]@rot(-theta))
  mats=torch.stack(mats)
 else:mats=torch.randn(DEPTH,WIDTH,WIDTH,generator=g)/math.sqrt(WIDTH)
 return x,torch.einsum('dij,nj->dni',mats,x),xt,torch.einsum('dij,nj->dni',mats,xt)

class GroupModel(nn.Module):
 def __init__(self,method):
  super().__init__();self.method=method
  k=2 if method.endswith('2') else 4
  self.group=k;groups=DEPTH//k
  if method=='untied':self.weight=nn.Parameter(torch.empty(DEPTH,WIDTH,WIDTH));nn.init.normal_(self.weight,std=1/math.sqrt(WIDTH))
  else:self.weight=nn.Parameter(torch.empty(groups,WIDTH,WIDTH));nn.init.normal_(self.weight,std=1/math.sqrt(WIDTH))
  if method.startswith('lora'):
   self.a=nn.Parameter(torch.randn(DEPTH,WIDTH,1)*.05);self.b=nn.Parameter(torch.zeros(DEPTH,1,WIDTH))
  if method.startswith('mirror'):self.angle=nn.Parameter(torch.zeros(DEPTH))
 def matrices(self):
  if self.method=='untied':return self.weight
  group=self.weight.repeat_interleave(self.group,dim=0)
  if self.method.startswith('lora'):return group+torch.bmm(self.a,self.b)
  if self.method.startswith('mirror'):return torch.stack([rot(t)@w@rot(-t) for t,w in zip(self.angle,group)])
  return group
 def forward(self,x):return torch.einsum('dij,nj->dni',self.matrices(),x)

def payload(m):
 b=io.BytesIO();torch.save({'method':m.method,'state_dict':m.state_dict()},b);return b.getvalue()

@dataclass
class Result:
 method:str;seed:int;condition:str;mse:float;layermax:float;bytes:int;updates:int;examples:int;macs:int;wall:float;throughput:float

def train_one(method,seed,condition,updates=600,lr=.01):
 torch.manual_seed(seed+862);x,y,xt,yt=make_world(seed,condition=='aligned');m=GroupModel(method);o=torch.optim.Adam(m.parameters(),lr=lr);t=time.perf_counter()
 for _ in range(updates):
  loss=(m(x)-y).square().mean();o.zero_grad(set_to_none=True);loss.backward();o.step()
 wall=time.perf_counter()-t
 with torch.no_grad():
  d=m(xt)-yt;v=d.square().mean(dim=(1,2)); mse=float(d.square().mean());lm=float(v.max())
 macs=updates*N*DEPTH*WIDTH*WIDTH*3
 if method.startswith('mirror'):macs+=updates*3*2*DEPTH*WIDTH**3
 elif method.startswith('lora'):macs+=updates*3*2*DEPTH*WIDTH
 return Result(method,seed,condition,mse,lm,len(payload(m)),updates,updates*N,int(macs),wall,updates*N/wall)

def run(seed,condition,updates=600,lr=.01):return [train_one(m,seed,condition,updates,lr) for m in METHODS]
