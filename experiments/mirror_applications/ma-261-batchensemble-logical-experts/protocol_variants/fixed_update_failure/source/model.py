from __future__ import annotations
import io,json,torch
from torch import nn
METHODS=('independent','tied','batchensemble','mirror')
E,D,O=2,16,8

def rotate(x,angle):
 y=x.clone();c,s=torch.cos(angle),torch.sin(angle)
 for i in range(4):
  a,b=x[...,2*i],x[...,2*i+1];y[...,2*i]=c*a-s*b;y[...,2*i+1]=s*a+c*b
 return y

class Experts(nn.Module):
 def __init__(self,method,router_seed=0):
  super().__init__();self.method=method;self.router_seed=int(router_seed)
  if method=='independent':self.w=nn.Parameter(torch.randn(E,D,O)*.04);self.b=nn.Parameter(torch.zeros(E,O))
  else:self.w=nn.Parameter(torch.randn(D,O)*.04);self.b=nn.Parameter(torch.zeros(O))
  if method=='batchensemble':self.r=nn.Parameter(torch.ones(E,D));self.s=nn.Parameter(torch.ones(E,O))
  if method=='mirror':self.angle=nn.Parameter(torch.zeros(E))
 def forward(self,x,e):
  if self.method=='independent':return torch.einsum('bd,bdo->bo',x,self.w[e])+self.b[e]
  if self.method=='batchensemble':return (x*self.r[e])@self.w*self.s[e]+self.b
  if self.method=='mirror':
   z=x.clone();z[:,:8]=rotate(x[:,:8],self.angle[e]);z[:,8:]=rotate(x[:,8:],self.angle[e]);return z@self.w+self.b
  return x@self.w+self.b
 def payload(self):
  cfg={'method':self.method,'router_seed':self.router_seed,'experts':E,'input':D,'output':O,'router':'first-half vs second-half oracle'}
  b=io.BytesIO();torch.save({'state_dict':self.state_dict(),'config':cfg},b);return b.getvalue()+json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()
 def payload_bytes(self):return len(self.payload())

def mac_proxy(method,n):
 p=E*D*O if method=='independent' else D*O
 if method=='batchensemble':p+=E*(D+O)
 if method=='mirror':p+=E*4
 return int(2*n*p)
