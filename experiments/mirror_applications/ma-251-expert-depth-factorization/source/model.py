from __future__ import annotations
import io,json,math
import torch
from torch import nn
METHODS=('untied','tied','expert_tied','expert_tied_depth_lora','depth_tied','expert_view','depth_view','factorized_mirror','cartesian_mirror')

def rotate(x,angles):
 p=x.reshape(*x.shape[:-1],x.shape[-1]//2,2);c,s=torch.cos(angles),torch.sin(angles);a,b=p[...,0],p[...,1]
 return torch.stack((c*a-s*b,s*a+c*b),-1).flatten(-2)

def factorized_input(x,e,l,ae,al):
 a=x.clone();b=x.clone();a[...,:8]=rotate(x[...,:8],ae[e]);b[...,8:]=rotate(x[...,8:],al[l]);return a+b-x

class FactorizedExpertDepth(nn.Module):
 def __init__(self,method,address_seed=0,experts=4,depth=4,d=16,out=12,rank=1):
  super().__init__();self.method=method;self.address_seed=int(address_seed);self.experts=experts;self.depth=depth;self.d=d;self.out=out;self.rank=rank
  if method=='untied':self.weight=nn.Parameter(torch.randn(experts,depth,d,out)*.07);self.bias=nn.Parameter(torch.zeros(experts,depth,out))
  elif method=='expert_tied':self.weight=nn.Parameter(torch.randn(experts,d,out)*.07);self.bias=nn.Parameter(torch.zeros(experts,out))
  elif method=='depth_tied':self.weight=nn.Parameter(torch.randn(depth,d,out)*.07);self.bias=nn.Parameter(torch.zeros(depth,out))
  else:
   self.weight=nn.Parameter(torch.randn(d,out)*.07);self.bias=nn.Parameter(torch.zeros(out))
  if method=='expert_tied_depth_lora':
   self.weight=nn.Parameter(torch.randn(experts,d,out)*.07);self.bias=nn.Parameter(torch.zeros(experts,out));self.lora_a=nn.Parameter(torch.zeros(depth,out,rank));self.lora_b=nn.Parameter(torch.randn(depth,rank,d)*.02)
  if method in ('expert_view','factorized_mirror'):self.expert_angles=nn.Parameter(torch.zeros(experts,4))
  if method in ('depth_view','factorized_mirror'):self.depth_angles=nn.Parameter(torch.zeros(depth,4))
  if method=='cartesian_mirror':self.pair_angles=nn.Parameter(torch.zeros(experts,depth,8))
 def forward(self,x,e,l):
  if self.method=='untied':return torch.einsum('bd,bdo->bo',x,self.weight[e,l])+self.bias[e,l]
  if self.method=='expert_tied':return torch.einsum('bd,bdo->bo',x,self.weight[e])+self.bias[e]
  if self.method=='depth_tied':return torch.einsum('bd,bdo->bo',x,self.weight[l])+self.bias[l]
  if self.method=='expert_tied_depth_lora':
   base=torch.einsum('bd,bdo->bo',x,self.weight[e])+self.bias[e]
   return base+torch.einsum('bd,brd,bro->bo',x,self.lora_b[l],self.lora_a[l])
  h=x
  if self.method=='expert_view':
   h=x.clone();h[:,:8]=rotate(x[:,:8],self.expert_angles[e])
  elif self.method=='depth_view':
   h=x.clone();h[:,8:]=rotate(x[:,8:],self.depth_angles[l])
  elif self.method=='factorized_mirror':h=factorized_input(x,e,l,self.expert_angles,self.depth_angles)
  elif self.method=='cartesian_mirror':h=rotate(x,self.pair_angles[e,l])
  return h@self.weight+self.bias
 def serialize(self):
  cfg={'method':self.method,'address_seed':self.address_seed,'experts':self.experts,'depth':self.depth,'d':self.d,'out':self.out,'rank':self.rank,'factorization':'expert rotates first 8 dims; depth rotates last 8 dims'}
  b=io.BytesIO();torch.save({'state_dict':self.state_dict(),'config':cfg},b);return b.getvalue()+json.dumps(cfg,sort_keys=True).encode()
 def serialized_payload_bytes(self):return len(self.serialize())
def compute_proxy(method,examples,experts=4,depth=4,d=16,out=12):
 factors=experts*depth if method=='untied' else (experts if method in ('expert_tied','expert_tied_depth_lora') else depth if method=='depth_tied' else 1)
 mac=examples*d*out*factors
 if method in ('expert_view','depth_view','factorized_mirror','cartesian_mirror'):mac+=examples*d*(experts+depth if method=='factorized_mirror' else experts*depth if method=='cartesian_mirror' else 1)
 if method=='expert_tied_depth_lora':mac+=examples*(d+out)*depth
 return int(mac*3)
