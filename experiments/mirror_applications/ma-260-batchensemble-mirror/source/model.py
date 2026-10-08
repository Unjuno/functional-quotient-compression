from __future__ import annotations
import io,json,torch
from torch import nn
METHODS=('independent','shared','batchensemble','mirror')
M,D,H,C=4,16,16,4

def rotate(x,a):
 y=x
 for i in range(8):
  u,v=y[...,2*i],y[...,2*i+1];c,s=torch.cos(a[...,i]),torch.sin(a[...,i]);z=y.clone();z[...,2*i]=c*u-s*v;z[...,2*i+1]=s*u+c*v;y=z
 return y

class Ensemble(nn.Module):
 def __init__(self,method,seed=0):
  super().__init__();self.method=method;self.seed=int(seed)
  if method=='independent':self.w1=nn.Parameter(torch.randn(M,D,H)*.15);self.b1=nn.Parameter(torch.zeros(M,H));self.w2=nn.Parameter(torch.randn(M,H,C)*.15);self.b2=nn.Parameter(torch.zeros(M,C))
  else:
   self.w1=nn.Parameter(torch.randn(D,H)*.15);self.b1=nn.Parameter(torch.zeros(H));self.w2=nn.Parameter(torch.randn(H,C)*.15);self.b2=nn.Parameter(torch.zeros(C))
  if method=='batchensemble':self.r=nn.Parameter(torch.ones(M,D));self.s=nn.Parameter(torch.ones(M,H));self.u=nn.Parameter(torch.ones(M,H));self.v=nn.Parameter(torch.ones(M,C))
  if method=='mirror':self.angles=nn.Parameter(torch.zeros(M,8))
 def forward_members(self,x):
  if self.method=='independent':return torch.stack([torch.relu(x@self.w1[k]+self.b1[k])@self.w2[k]+self.b2[k] for k in range(M)],dim=1)
  if self.method=='batchensemble':
   out=[]
   for k in range(M):
    h=torch.relu((x*self.r[k])@self.w1*self.s[k]+self.b1);out.append((h*self.u[k])@self.w2*self.v[k]+self.b2)
   return torch.stack(out,dim=1)
  base=torch.relu(x@self.w1+self.b1)
  if self.method=='mirror':return torch.stack([rotate(base,self.angles[k])@self.w2+self.b2 for k in range(M)],dim=1)
  z=base@self.w2+self.b2
  return z[:,None,:].expand(-1,M,-1)
 def payload(self):
  cfg={'method':self.method,'seed':self.seed,'members':M,'input':D,'hidden':H,'classes':C,'view':'8 hidden Givens planes' if self.method=='mirror' else None}
  b=io.BytesIO();torch.save({'state_dict':self.state_dict(),'config':cfg},b);return b.getvalue()+json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()
 def payload_bytes(self):return len(self.payload())

def mac_proxy(method,n):
 params=M*(D*H+H*C) if method=='independent' else D*H+H*C
 if method=='batchensemble':params+=M*(D+2*H+C)
 if method=='mirror':params+=8*M
 return 2*n*params
