from __future__ import annotations
import io,json,math
import torch
from torch import nn
METHODS=('untied','tied','scalar_gate','lowrank','map','hadamard','hrr','mirror')
def hadamard(d,dtype=torch.float32):
 h=torch.ones((1,1),dtype=dtype)
 while h.shape[0]<d:h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)/math.sqrt(2)
 return h
def address_matrices(seed,method,roles=4,d=16):
 g=torch.Generator().manual_seed(int(seed));mats=[]
 if method=='map':
  for _ in range(roles):
   perm=torch.randperm(d,generator=g);sgn=torch.where(torch.randint(2,(d,),generator=g)>0,1.,-1.);m=torch.zeros(d,d);m[torch.arange(d),perm]=sgn;mats.append(m)
 elif method=='hadamard':
  h=hadamard(d)
  for _ in range(roles):
   s=torch.where(torch.randint(2,(d,),generator=g)>0,1.,-1.);mats.append(h@torch.diag(s)@h.T)
 elif method=='hrr':
  for _ in range(roles):
   phases=torch.rand(d//2+1,generator=g)*2*math.pi;spec=torch.polar(torch.ones_like(phases),phases);spec[0]=torch.complex(torch.where(torch.rand((),generator=g)>0,.999999,-.999999),torch.tensor(0.))
   if d%2==0:spec[-1]=torch.complex(torch.where(torch.rand((),generator=g)>0,.999999,-.999999),torch.tensor(0.))
   kernel=torch.fft.irfft(spec,n=d);idx=(torch.arange(d)[:,None]-torch.arange(d)[None,:])%d;mats.append(kernel[idx])
 return torch.stack(mats)
def givens(x,angles):
 p=x.reshape(*x.shape[:-1],x.shape[-1]//2,2);c,s=torch.cos(angles),torch.sin(angles);a,b=p[...,0],p[...,1]
 return torch.stack((c*a-s*b,s*a+c*b),-1).flatten(-2)
class ExpertViews(nn.Module):
 def __init__(self,method,address_seed,roles=4,d=16,out=12,rank=1):
  super().__init__();self.method=method;self.address_seed=int(address_seed);self.roles=roles;self.d=d;self.out=out;self.rank=rank
  if method=='untied':self.weight=nn.Parameter(torch.randn(roles,d,out)*.08);self.bias=nn.Parameter(torch.zeros(roles,out))
  else:
   self.weight=nn.Parameter(torch.randn(d,out)*.08);self.bias=nn.Parameter(torch.zeros(out))
   if method=='scalar_gate':self.gate=nn.Parameter(torch.ones(roles))
   if method=='lowrank':self.a=nn.Parameter(torch.zeros(roles,out,rank));self.b=nn.Parameter(torch.randn(roles,rank,d)*.02)
   if method=='mirror':self.angles=nn.Parameter(torch.zeros(roles,d//2))
  self.address=address_matrices(address_seed,method,roles,d) if method in ('map','hadamard','hrr') else None
 def forward(self,x,role):
  if self.method=='untied':return torch.einsum('bd,bdo->bo',x,self.weight[role])+self.bias[role]
  if self.method in ('map','hadamard','hrr'):h=torch.bmm(x[:,None,:],self.address[role]).squeeze(1)
  elif self.method=='mirror':h=givens(x,self.angles[role])
  else:h=x
  y=h@self.weight+self.bias
  if self.method=='scalar_gate':y=y*self.gate[role,None]
  if self.method=='lowrank':y=y+torch.einsum('bd,brd,bro->bo',x,self.b[role],self.a[role])
  return y
 def serialize(self):
  cfg={'method':self.method,'address_seed':self.address_seed,'roles':self.roles,'d':self.d,'out':self.out,'rank':self.rank,'address_algorithm':'map-sign-permutation / hadamard-HdiagH / hrr-unit-spectrum-v1'}
  b=io.BytesIO();torch.save({'state_dict':self.state_dict(),'config':cfg},b);return b.getvalue()
 @staticmethod
 def from_serialized(payload):
  state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False);cfg=state['config']
  m=ExpertViews(cfg['method'],cfg['address_seed'],cfg['roles'],cfg['d'],cfg['out'],cfg['rank'])
  m.load_state_dict(state['state_dict']);return m
 def serialized_payload_bytes(self):return len(self.serialize())
def compute_proxy(method,examples,roles=4,d=16,out=12):
 mac=examples*d*out*(roles if method=='untied' else 1)
 if method in ('map','hadamard','hrr','mirror'):mac+=examples*d*roles
 if method=='lowrank':mac+=examples*(d+out)*roles
 return int(mac*3)
