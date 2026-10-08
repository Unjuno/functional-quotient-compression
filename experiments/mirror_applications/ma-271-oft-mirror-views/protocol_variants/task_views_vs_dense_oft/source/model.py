from __future__ import annotations
import io,json,torch
from torch import nn
METHODS=('independent','oft','lowrank','mirror')
T,D,O,R=4,16,16,2

def givens(x,angles):
 y=x
 for i in range(8):
  a,b=y[...,2*i],y[...,2*i+1];c,s=torch.cos(angles[...,i]),torch.sin(angles[...,i]);z=y.clone();z[...,2*i]=c*a-s*b;z[...,2*i+1]=s*a+c*b;y=z
 return y
class TaskViews(nn.Module):
 def __init__(self,method,seed=0):
  super().__init__();self.method=method;self.seed=int(seed);self.base=nn.Parameter(torch.randn(D,O)*.04)
  if method=='independent':self.weight=nn.Parameter(torch.randn(T,D,O)*.04)
  if method=='oft':self.raw=nn.Parameter(torch.zeros(T,O,O))
  if method=='lowrank':self.left=nn.Parameter(torch.randn(T,D,R)*.01);self.right=nn.Parameter(torch.randn(T,R,O)*.01)
  if method=='mirror':self.angles=nn.Parameter(torch.zeros(T,8))
 def mats(self):
  if self.method=='independent':return self.weight
  if self.method=='oft':
   skew=self.raw-self.raw.transpose(-1,-2);eye=torch.eye(O,device=skew.device).expand(T,-1,-1);return torch.linalg.solve(eye+skew,eye-skew)
  if self.method=='mirror':
   mats=[]
   for t in range(T):
    m=torch.eye(O)
    for i in range(8):
     a=self.angles[t,i];c,s=torch.cos(a),torch.sin(a);r=torch.eye(O);r[2*i,2*i]=c;r[2*i,2*i+1]=-s;r[2*i+1,2*i]=s;r[2*i+1,2*i+1]=c;m=r@m
    mats.append(m)
   return torch.stack(mats)
  return None
 def forward(self,x,t):
  if self.method=='lowrank':return x@self.base+torch.einsum('bd,bdr,bro->bo',x,self.left[t],self.right[t])
  mats=self.mats()
  if mats is not None:return torch.einsum('bd,do,boq->bq',x,self.base,mats[t])
  return x@self.base
 def payload(self):
  cfg={'method':self.method,'seed':self.seed,'tasks':T,'D':D,'O':O,'rank':R,'oft_transform':'Cayley(skew raw)','mirror':'8 adjacent-plane Givens' if self.method=='mirror' else None}
  sd=self.state_dict()
  if self.method=='oft':
   # Native dense OFT transform is the direct inference state.
   sd={k:v for k,v in sd.items() if k!='raw'};sd['effective_task_transforms']=self.mats().detach()
  b=io.BytesIO();torch.save({'state_dict':sd,'config':cfg},b);return b.getvalue()+json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()
 def payload_bytes(self):return len(self.payload())
def mac_proxy(method,n):
 per=2*D*O
 if method=='oft':per+=T*O*O*O
 if method=='mirror':per+=T*8
 if method=='lowrank':per+=T*R*(D+O)
 if method=='independent':per+=T*D*O
 return n*per
