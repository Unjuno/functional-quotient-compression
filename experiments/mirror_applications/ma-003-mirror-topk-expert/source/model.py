from __future__ import annotations
import io,json,math
import torch
from torch import nn
METHODS=('full_moe','lowrank_router_full_moe','tied_expert','scalar_gate','lowrank_expert','mirror')
def givens(x,angles):
 p=x.reshape(*x.shape[:-1],x.shape[-1]//2,2);c,s=torch.cos(angles),torch.sin(angles);a,b=p[...,0],p[...,1]
 return torch.stack((c*a-s*b,s*a+c*b),-1).flatten(-2)
class TopKExperts(nn.Module):
 def __init__(self,method,address_seed=0,experts=4,d=16,out=12,rank=1,router_rank=2):
  super().__init__();self.method=method;self.address_seed=int(address_seed);self.experts=experts;self.d=d;self.out=out;self.rank=rank;self.router_rank=router_rank
  if method=='lowrank_router_full_moe':self.router_a=nn.Linear(d,router_rank,bias=False);self.router_b=nn.Linear(router_rank,experts)
  else:self.router=nn.Linear(d,experts)
  if method=='full_moe':self.weight=nn.Parameter(torch.randn(experts,d,out)*.07);self.bias=nn.Parameter(torch.zeros(experts,out))
  else:
   self.weight=nn.Parameter(torch.randn(d,out)*.07);self.bias=nn.Parameter(torch.zeros(out))
   if method=='scalar_gate':self.gate=nn.Parameter(torch.ones(experts))
   if method=='lowrank_expert':self.a=nn.Parameter(torch.zeros(experts,out,rank));self.b=nn.Parameter(torch.randn(experts,rank,d)*.02)
   if method=='mirror':self.angles=nn.Parameter(torch.zeros(experts,d//2))
 def route_logits(self,x):
  if self.method=='lowrank_router_full_moe':return self.router_b(self.router_a(x))
  return self.router(x)
 def forward(self,x):
  logits=self.route_logits(x);role=logits.argmax(-1)
  if self.method=='full_moe':y=torch.einsum('bd,bdo->bo',x,self.weight[role])+self.bias[role]
  else:
   h=givens(x,self.angles[role]) if self.method=='mirror' else x;y=h@self.weight+self.bias
   if self.method=='scalar_gate':y=y*self.gate[role,None]
   if self.method=='lowrank_expert':y=y+torch.einsum('bd,brd,bro->bo',x,self.b[role],self.a[role])
  return y,logits,role
 def serialize(self):
  cfg={'method':self.method,'address_seed':self.address_seed,'experts':self.experts,'d':self.d,'out':self.out,'rank':self.rank,'router_rank':self.router_rank,'top_k':1}
  b=io.BytesIO();torch.save({'state_dict':self.state_dict(),'config':cfg},b);return b.getvalue()+json.dumps(cfg,sort_keys=True).encode()
 def serialized_payload_bytes(self):return len(self.serialize())
def compute_proxy(method,examples,experts=4,d=16,out=12):
 expert=examples*d*out*(experts if method=='full_moe' else 1);router=examples*d*experts
 if method=='mirror':expert+=examples*d*experts
 if method=='lowrank_expert':expert+=examples*(d+out)*experts
 return int(3*(expert+router))
