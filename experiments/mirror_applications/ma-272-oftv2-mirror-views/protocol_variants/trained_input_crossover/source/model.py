from __future__ import annotations
import io,json,torch
from torch import nn
T,D,O=4,16,16
METHODS=('independent','shared','oftv2','mirror')
def givens(x,a):
 y=x
 for i in range(8):
  p,q=y[...,2*i],y[...,2*i+1];c,s=torch.cos(a[...,i]),torch.sin(a[...,i]);z=y.clone();z[...,2*i]=c*p-s*q;z[...,2*i+1]=s*p+c*q;y=z
 return y
class Views(nn.Module):
 def __init__(self,method,seed=0):
  super().__init__();self.method=method;self.seed=seed
  if method=='independent':self.w=nn.Parameter(torch.randn(T,D,O)*.04)
  else:self.w=nn.Parameter(torch.randn(D,O)*.04)
  if method=='oftv2':self.q=nn.Parameter(torch.zeros(T,D,D))
  if method=='mirror':self.a=nn.Parameter(torch.zeros(T,8))
 def orth(self):
  skew=self.q-self.q.transpose(-1,-2);eye=torch.eye(D).expand(T,-1,-1);return torch.linalg.solve(eye+skew,eye-skew)
 def forward(self,x,t):
  if self.method=='independent':return torch.einsum('bd,bdo->bo',x,self.w[t])
  if self.method=='oftv2':z=torch.einsum('bd,bdh->bh',x,self.orth()[t]);return z@self.w
  if self.method=='mirror':return givens(x,self.a[t])@self.w
  return x@self.w
 def payload(self):
  cfg={'method':self.method,'seed':self.seed,'tasks':T,'d':D,'insertion':'input activation before shared map'}
  sd=self.state_dict()
  if self.method=='oftv2':sd={'w':self.w.detach(),'effective_input_transforms':self.orth().detach()}
  b=io.BytesIO();torch.save({'state_dict':sd,'config':cfg},b);return b.getvalue()+json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()
 def payload_bytes(self):return len(self.payload())
def mac(method,n):return n*(2*D*O+(T*D*D*D if method=='oftv2' else T*8*D if method=='mirror' else T*D*O if method=='independent' else 0))
