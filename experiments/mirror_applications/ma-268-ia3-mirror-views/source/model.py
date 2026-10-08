from __future__ import annotations
import io,json,torch
from torch import nn
METHODS=('independent','shared','ia3','mirror')
T,D,H,O=4,16,16,4

def rotate4(x,a):
 y=x.clone()
 for i in range(4):
  u,v=x[...,2*i],x[...,2*i+1];c,s=torch.cos(a[...,i]),torch.sin(a[...,i]);y[...,2*i]=c*u-s*v;y[...,2*i+1]=s*u+c*v
 return y
class TaskNet(nn.Module):
 def __init__(self,method,seed=0):
  super().__init__();self.method=method;self.seed=int(seed)
  if method=='independent':self.w1=nn.Parameter(torch.randn(T,D,H)*.08);self.b1=nn.Parameter(torch.zeros(T,H));self.w2=nn.Parameter(torch.randn(T,H,O)*.08);self.b2=nn.Parameter(torch.zeros(T,O))
  else:self.w1=nn.Parameter(torch.randn(D,H)*.08);self.b1=nn.Parameter(torch.zeros(H));self.w2=nn.Parameter(torch.randn(H,O)*.08);self.b2=nn.Parameter(torch.zeros(O))
  if method=='ia3':self.scale=nn.Parameter(torch.ones(T,H))
  if method=='mirror':self.angle=nn.Parameter(torch.zeros(T,4))
 def forward(self,x,t):
  if self.method=='independent':
   h=torch.tanh(torch.einsum('bd,bdh->bh',x,self.w1[t])+self.b1[t]);return torch.einsum('bh,bho->bo',h,self.w2[t])+self.b2[t]
  h=torch.tanh(x@self.w1+self.b1)
  if self.method=='ia3':h=h*self.scale[t]
  if self.method=='mirror':h=rotate4(h,self.angle[t])
  return h@self.w2+self.b2
 def payload(self):
  cfg={'method':self.method,'seed':self.seed,'tasks':T,'input':D,'hidden':H,'output':O,'mirror':'four paired rotations' if self.method=='mirror' else None}
  b=io.BytesIO();torch.save({'state_dict':self.state_dict(),'config':cfg},b);return b.getvalue()+json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()
 def payload_bytes(self):return len(self.payload())
def mac_proxy(method,n):
 p=T*(D*H+H*O) if method=='independent' else D*H+H*O
 if method=='ia3':p+=T*H
 if method=='mirror':p+=T*4*H
 return 2*n*p
