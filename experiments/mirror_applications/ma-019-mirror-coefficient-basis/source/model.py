from __future__ import annotations
import io,json,torch
from torch import nn
N,D,O,R=8,16,12,1
METHODS=['full_moe','tied','rank1','rank2_generic','mirror_angle']
def role_of(x):return (x[:,0]>=0).long()*4+(x[:,1]>=0).long()*2+(x[:,2]>=0).long()
class ExpertBasis(nn.Module):
 def __init__(self,method,seed=0):
  super().__init__();self.method=method;g=torch.Generator().manual_seed(seed)
  if method=='full_moe':self.w=nn.Parameter(torch.randn(N,D,O,generator=g)*.07)
  elif method=='tied':self.w=nn.Parameter(torch.randn(D,O,generator=g)*.07)
  elif method=='rank1':self.w=nn.Parameter(torch.randn(D,O,generator=g)*.07);self.u=nn.Parameter(torch.randn(N,D,R,generator=g)*.04);self.v=nn.Parameter(torch.zeros(N,R,O))
  else:
   self.basis=nn.Parameter(torch.randn(2,D,O,generator=g)*.07)
   if method=='rank2_generic':self.coeff=nn.Parameter(torch.randn(N,2,generator=g)*.2)
   elif method=='mirror_angle':self.angle=nn.Parameter(torch.zeros(N))
 def matrices(self):
  if self.method=='full_moe':return self.w
  if self.method=='tied':return self.w.expand(N,-1,-1)
  if self.method=='rank1':return self.w[None]+torch.einsum('ndr,nro->ndo',self.u,self.v)
  if self.method=='rank2_generic':return torch.einsum('nk,kdo->ndo',self.coeff,self.basis)
  c,s=torch.cos(self.angle),torch.sin(self.angle);return torch.einsum('n,do->ndo',c,self.basis[0])+torch.einsum('n,do->ndo',s,self.basis[1])
 def forward(self,x):return torch.einsum('bd,bdo->bo',x,self.matrices()[role_of(x)])
 def serialized_payload_bytes(self):
  b=io.BytesIO();torch.save({'state_dict':self.state_dict(),'config':json.dumps({'method':self.method,'n':N,'d':D,'o':O},sort_keys=True)},b);return len(b.getvalue())
def compute_proxy(method,examples):
 per=(D*O if method in ('full_moe','tied') else D*O+(D+O)*R if method=='rank1' else 2*D*O+2)
 return per*examples
