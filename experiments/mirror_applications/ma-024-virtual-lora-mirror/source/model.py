from __future__ import annotations
import io,json,torch
from torch import nn
N,D,O,R=8,16,12,2
METHODS=['lora_bank','shared_lora','generic_basis','hyper_coeff','mirror_lora']
def task_id(x):return (x[:,0]>=0).long()*4+(x[:,1]>=0).long()*2+(x[:,2]>=0).long()
class VirtualLoRA(nn.Module):
 def __init__(self,method,seed=0):
  super().__init__();self.method=method;g=torch.Generator().manual_seed(seed)
  if method=='lora_bank':self.a=nn.Parameter(torch.randn(N,D,R,generator=g)*.08);self.b=nn.Parameter(torch.zeros(N,R,O))
  elif method=='shared_lora':self.a=nn.Parameter(torch.randn(D,R,generator=g)*.08);self.b=nn.Parameter(torch.zeros(R,O))
  elif method=='generic_basis':
   self.a=nn.Parameter(torch.randn(2,D,R,generator=g)*.08);self.b=nn.Parameter(torch.zeros(2,R,O));self.coef=nn.Parameter(torch.randn(N,2,generator=g)*.1)
  elif method=='hyper_coeff':
   self.a=nn.Parameter(torch.randn(2,D,R,generator=g)*.08);self.b=nn.Parameter(torch.zeros(2,R,O));self.embed=nn.Parameter(torch.randn(N,2,generator=g)*.1);self.gen=nn.Linear(2,2);nn.init.zeros_(self.gen.weight);nn.init.zeros_(self.gen.bias)
  elif method=='mirror_lora':
   self.a=nn.Parameter(torch.randn(D,R,generator=g)*.08);self.b=nn.Parameter(torch.zeros(R,O));self.angle=nn.Parameter(torch.zeros(N))
 def matrices(self):
  if self.method=='lora_bank':return torch.einsum('ndr,nro->ndo',self.a,self.b)
  if self.method=='shared_lora':return (self.a@self.b).expand(N,-1,-1)
  if self.method in ('generic_basis','hyper_coeff'):
   bases=torch.einsum('kdr,kro->kdo',self.a,self.b)
   c=self.coef if self.method=='generic_basis' else self.gen(self.embed)
   return torch.einsum('nk,kdo->ndo',c,bases)
  c,s=torch.cos(self.angle),torch.sin(self.angle);rot=torch.stack((torch.stack((c,-s),dim=-1),torch.stack((s,c),dim=-1)),dim=1)
  return torch.einsum('dr,nrs,so->ndo',self.a,rot,self.b)
 def forward(self,x):return torch.einsum('bd,bdo->bo',x,self.matrices()[task_id(x)])
 def serialized_payload_bytes(self):
  b=io.BytesIO();torch.save({'state_dict':self.state_dict(),'config':json.dumps({'method':self.method,'n':N,'d':D,'o':O,'rank':R},sort_keys=True)},b);return len(b.getvalue())
def compute_proxy(method,examples):
  per= D*R+R*O if method in ('lora_bank','shared_lora') else 2*(D*R+R*O) if method in ('generic_basis','hyper_coeff') else 2*(D*R+R*O)
  return examples*per
