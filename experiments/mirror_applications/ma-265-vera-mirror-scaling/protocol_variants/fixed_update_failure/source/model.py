from __future__ import annotations
import io,json,torch
from torch import nn
T,D,O,R=4,16,8,4
METHODS=('independent','shared','vera','mirror')
def orthogonal_code(z,angle):
 y=z.clone();c,s=torch.cos(angle),torch.sin(angle)
 for i in range(R//2):
  a,b=z[...,2*i],z[...,2*i+1];y[...,2*i]=c*a-s*b;y[...,2*i+1]=s*a+c*b
 return y
class AdapterBank(nn.Module):
 def __init__(self,method,basis_seed=0):
  super().__init__();self.method=method;self.basis_seed=int(basis_seed)
  g=torch.Generator().manual_seed(basis_seed);a=torch.randn(D,R,generator=g)/D**.5;b=torch.randn(R,O,generator=g)/R**.5
  self.register_buffer('A',a);self.register_buffer('B',b)
  self.base=nn.Parameter(torch.zeros(D,O))
  if method=='independent':self.delta=nn.Parameter(torch.randn(T,D,O)*.01)
  if method=='vera':self.left=nn.Parameter(torch.ones(T,R));self.right=nn.Parameter(torch.ones(T,O))
  if method=='mirror':self.angles=nn.Parameter(torch.zeros(T))
 def forward(self,x,task):
  base=x@self.base
  if self.method=='independent':return base+torch.einsum('bd,bdo->bo',x,self.delta[task])
  if self.method=='shared':return base
  z=x@self.A
  if self.method=='vera':return base+torch.einsum('br,bro,bo->bo',z[:,None,:],torch.ones_like(z[:,None,:]),self.B.unsqueeze(0).expand(len(x),-1,-1))*0 if False else base+((z*self.left[task])@self.B)*self.right[task]
  if self.method=='mirror':return base+(orthogonal_code(z,self.angles[task])@self.B)
  raise ValueError(self.method)
 def payload(self):
  cfg={'method':self.method,'basis_seed':self.basis_seed,'tasks':T,'input':D,'output':O,'rank':R,'mirror':'task-wise same-angle Givens pairs' if self.method=='mirror' else None}
  sd={k:v for k,v in self.state_dict().items() if k not in ('A','B')};sd['A']=self.A;sd['B']=self.B
  b=io.BytesIO();torch.save({'state_dict':sd,'config':cfg},b);return b.getvalue()+json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()
 def payload_bytes(self):return len(self.payload())
def mac_proxy(method,n):
 p=D*O+2*D*R if method in ('vera','mirror') else D*O
 if method=='independent':p+=T*D*O
 if method=='vera':p+=T*(R+O)
 if method=='mirror':p+=T
 return 2*n*p
