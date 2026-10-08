from __future__ import annotations
import io,json,torch
from torch import nn
METHODS=('independent','shared','boft','lowrank','mirror')
T,D,O,R=4,16,16,2

def butterfly(x,angles):
 y=x
 for stage in range(4):
  bit=1<<stage
  for i in range(D):
   if i&bit: continue
   j=i|bit;a,b=y[...,i],y[...,j];c,s=torch.cos(angles[...,stage,i//(2*bit)*bit + (i%bit)]),torch.sin(angles[...,stage,i//(2*bit)*bit + (i%bit)])
   z=y.clone();z[...,i]=c*a-s*b;z[...,j]=s*a+c*b;y=z
 return y
class Bank(nn.Module):
 def __init__(self,method,seed=0):
  super().__init__();self.method=method;self.seed=seed
  if method=='independent':self.w=nn.Parameter(torch.randn(T,D,O)*.04)
  else:self.w=nn.Parameter(torch.randn(D,O)*.04)
  if method=='boft':self.angles=nn.Parameter(torch.zeros(T,4,8))
  if method=='mirror':self.atom=nn.Parameter(torch.zeros(4,8));self.code=nn.Parameter(torch.ones(T))
  if method=='lowrank':self.a=nn.Parameter(torch.randn(T,D,R)*.01);self.b=nn.Parameter(torch.randn(T,R,O)*.01)
 def forward(self,x,t):
  if self.method=='independent':return torch.einsum('bd,bdo->bo',x,self.w[t])
  base=x@self.w
  if self.method=='boft':return torch.einsum('bd,bdo->bo',butterfly(x,self.angles[t]),self.w.unsqueeze(0).expand(len(x),-1,-1))
  if self.method=='mirror':
   aa=self.atom.unsqueeze(0)*self.code[t,None,None];return butterfly(x,aa)@self.w
  if self.method=='lowrank':return base+torch.einsum('bd,bdr,bro->bo',x,self.a[t],self.b[t])
  return base
 def payload(self):
  cfg={'method':self.method,'seed':self.seed,'tasks':T,'D':D,'O':O,'boft':'4 binary butterfly stages, 8 Givens per stage','mirror':'shared 4x8 angle atom plus scalar task codes'}
  b=io.BytesIO();torch.save({'state_dict':self.state_dict(),'config':cfg},b);return b.getvalue()+json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()
 def payload_bytes(self):return len(self.payload())
def mac(method,n):
 p=D*O*(T if method=='independent' else 1)
 if method=='boft':p+=T*4*8*D
 if method=='mirror':p+=4*8*D+T*D
 if method=='lowrank':p+=T*R*(D+O)
 return n*p
