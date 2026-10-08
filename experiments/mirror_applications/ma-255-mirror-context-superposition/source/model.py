from __future__ import annotations
import io,json,torch
from torch import nn
METHODS=('independent','shared','psp','lowrank','mirror_diag','mirror_orthogonal')
T,D,O,RANK=8,16,16,2

def givens(x,angles):
 y=x
 for i in range(8):
  a,b=y[...,2*i],y[...,2*i+1];c,s=torch.cos(angles[...,i]),torch.sin(angles[...,i]);z=y.clone()
  z[...,2*i]=c*a-s*b;z[...,2*i+1]=s*a+c*b;y=z
 return y

class TaskFamilyModel(nn.Module):
 def __init__(self,method,context_seed=0):
  super().__init__();self.method=method;self.context_seed=int(context_seed)
  self.weight=nn.Parameter(torch.randn(T,D,O)*.05 if method=='independent' else torch.randn(D,O)*.05);self.bias=nn.Parameter(torch.zeros(O))
  if method=='psp':
   g=torch.Generator().manual_seed(context_seed);self.register_buffer('context',(torch.randint(0,2,(T,D,O),generator=g,dtype=torch.int64)*2-1).float())
  if method=='lowrank':self.left=nn.Parameter(torch.randn(T,D,RANK)*.01);self.right=nn.Parameter(torch.randn(T,RANK,O)*.01)
  if method=='mirror_diag':self.gain=nn.Parameter(torch.ones(T,O))
  if method=='mirror_orthogonal':self.angles=nn.Parameter(torch.zeros(T,8))
 def forward(self,x,task):
  if self.method=='independent':return torch.einsum('bd,bdo->bo',x,self.weight[task])+self.bias
  if self.method=='psp':
   # Bind task tensors with fixed +/-1 contexts, sum into one physical tensor,
   # then unbind using the same context (cross-task interference remains).
   # Here weight is the one physical superposed tensor, trained directly.
   # Decoding sign-unbinds each task address; interference is explicit.
   wb=self.weight.unsqueeze(0)*self.context[task]/T
   return torch.einsum('bd,bdo->bo',x,wb)+self.bias
  base=x@self.weight
  if self.method=='shared':return base+self.bias
  if self.method=='lowrank':return base+torch.einsum('bd,bdr,bro->bo',x,self.left[task],self.right[task])+self.bias
  if self.method=='mirror_diag':return base*self.gain[task]+self.bias
  if self.method=='mirror_orthogonal':return givens(base,self.angles[task])+self.bias
  raise ValueError(self.method)
 def payload(self):
  cfg={'method':self.method,'context_seed':self.context_seed,'T':T,'D':D,'O':O,'rank':RANK,'view':'8 adjacent Givens planes' if self.method=='mirror_orthogonal' else None}
  b=io.BytesIO();torch.save({'state_dict':{k:v for k,v in self.state_dict().items() if k!='context'},'config':cfg},b);return b.getvalue()+json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()
 def payload_bytes(self):return len(self.payload())

def mac_proxy(method,examples):
 per=(T*D*O if method=='independent' else D*O)
 if method=='lowrank':per+=T*(D*RANK+RANK*O)
 if method=='psp':per+=T*D*O
 if method=='mirror_orthogonal':per+=8*O
 if method=='mirror_diag':per+=O
 return int(2*examples*per)
