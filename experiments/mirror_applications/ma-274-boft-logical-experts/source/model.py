from __future__ import annotations
import io,json,torch
from torch import nn
METHODS=('independent','tied','ia3','rankone','boft','mirror')
E,D,H,O=4,16,16,16

def butterfly(x,angles):
 y=x
 for stage in range(4):
  bit=1<<stage
  for i in range(D):
   if i&bit:continue
   j=i|bit;a,b=y[...,i],y[...,j];idx=(i//(2*bit))*bit+(i%bit);c,s=torch.cos(angles[...,stage,idx]),torch.sin(angles[...,stage,idx]);z=y.clone();z[...,i]=c*a-s*b;z[...,j]=s*a+c*b;y=z
 return y
class RoutedFFN(nn.Module):
 def __init__(self,method,seed=0):
  super().__init__();self.method=method;self.seed=int(seed)
  n=E if method=='independent' else 1
  self.w1=nn.Parameter(torch.randn(n,D,H)*.06);self.b1=nn.Parameter(torch.zeros(n,H));self.w2=nn.Parameter(torch.randn(n,H,O)*.06);self.b2=nn.Parameter(torch.zeros(n,O))
  if method=='ia3':self.gain=nn.Parameter(torch.ones(E,H))
  if method=='rankone':self.r=nn.Parameter(torch.ones(E,D));self.s=nn.Parameter(torch.ones(E,H));self.u=nn.Parameter(torch.ones(E,H));self.v=nn.Parameter(torch.ones(E,O))
  if method=='boft':self.angles=nn.Parameter(torch.zeros(E,4,8))
  if method=='mirror':self.atom=nn.Parameter(torch.zeros(4,8));self.code=nn.Parameter(torch.ones(E))
 def forward(self,x,e):
  if self.method=='independent':
   h=torch.tanh(torch.einsum('bd,bdh->bh',x,self.w1[e])+self.b1[e]);return torch.einsum('bh,bho->bo',h,self.w2[e])+self.b2[e]
  if self.method=='rankone':
   h=torch.tanh((x*self.r[e])@self.w1[0]*self.s[e]+self.b1[0]);return (h*self.u[e])@self.w2[0]*self.v[e]+self.b2[0]
  h=torch.tanh(x@self.w1[0]+self.b1[0])
  if self.method=='ia3':h=h*self.gain[e]
  elif self.method=='boft':h=butterfly(h,self.angles[e])
  elif self.method=='mirror':h=butterfly(h,self.atom[None]*self.code[e,None,None])
  return h@self.w2[0]+self.b2[0]
 def payload(self):
  cfg={'method':self.method,'seed':self.seed,'experts':E,'d':D,'hidden':H,'out':O,'oracle_router':'external expert IDs supplied','boft':'4-stage butterfly with 8 angles/stage','mirror':'shared 4x8 angle atom plus scalar expert codes'}
  b=io.BytesIO();torch.save({'state_dict':self.state_dict(),'config':cfg},b);return b.getvalue()+json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()
 def payload_bytes(self):return len(self.payload())
def mac_proxy(method,n):
 base=2*(D*H+H*O)
 if method=='independent':base*=E
 elif method=='ia3':base+=E*H
 elif method=='rankone':base+=E*(D+2*H+O)
 elif method=='boft':base+=E*4*8*H
 elif method=='mirror':base+=4*8*H+E*H
 return int(n*base)
