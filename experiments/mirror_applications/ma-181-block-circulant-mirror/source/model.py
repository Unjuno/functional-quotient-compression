import io,math
import torch
from torch import nn

ROLES,D,BLOCK=4,16,4
NBLK=D//BLOCK
METHODS=('untied','block_independent','block_tied','block_shift','block_map','block_hadamard','block_hrr','block_givens','dense_lowrank')

def hadamard(d):
 h=torch.ones((1,1))
 while h.shape[0]<d:h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)/math.sqrt(2)
 return h

def address_matrices(seed,method,roles=ROLES,d=D):
 g=torch.Generator().manual_seed(int(seed));mats=[]
 if method=='block_map':
  for _ in range(roles):
   p=torch.randperm(d,generator=g);s=torch.where(torch.randint(2,(d,),generator=g)>0,1.,-1.);m=torch.zeros(d,d);m[torch.arange(d),p]=s;mats.append(m)
 elif method=='block_hadamard':
  h=hadamard(d)
  for _ in range(roles):
   s=torch.where(torch.randint(2,(d,),generator=g)>0,1.,-1.);mats.append(h@torch.diag(s)@h.T)
 elif method=='block_hrr':
  for _ in range(roles):
   ph=torch.rand(d//2+1,generator=g)*2*math.pi;sp=torch.polar(torch.ones_like(ph),ph);sp[0]=torch.complex(torch.tensor(1.),torch.tensor(0.));sp[-1]=torch.complex(torch.tensor(1.),torch.tensor(0.));k=torch.fft.irfft(sp,n=d);idx=(torch.arange(d)[:,None]-torch.arange(d)[None,:])%d;mats.append(k[idx])
 return torch.stack(mats)

def shift_codes(seed,roles=ROLES):
 return torch.randint(BLOCK,(roles,NBLK,NBLK),generator=torch.Generator().manual_seed(int(seed)),dtype=torch.int8)

def block_matrix(kernels,shifts=None):
 # kernels: [out-block, in-block, block-width]
 w=torch.zeros(D,D)
 for ob in range(NBLK):
  for ib in range(NBLK):
   kernel=kernels[ob,ib]
   if shifts is not None:kernel=torch.roll(kernel,int(shifts[ob,ib]))
   ix=(torch.arange(BLOCK)[:,None]-torch.arange(BLOCK)[None,:])%BLOCK
   w[ob*BLOCK:(ob+1)*BLOCK,ib*BLOCK:(ib+1)*BLOCK]=kernel[ix]
 return w

def givens(x,angles):
 p=x.reshape(*x.shape[:-1],x.shape[-1]//2,2);c,s=torch.cos(angles),torch.sin(angles);a,b=p[...,0],p[...,1]
 return torch.stack((c*a-s*b,s*a+c*b),-1).flatten(-2)

class BlockViews(nn.Module):
 def __init__(self,method,address_seed,roles=ROLES,d=D,block=BLOCK,rank=1):
  super().__init__();self.method=method;self.address_seed=int(address_seed);self.roles=roles;self.d=d;self.block=block;self.nb=d//block;self.rank=rank
  if method=='untied':self.weight=nn.Parameter(torch.randn(roles,d,d)*.08);self.bias=nn.Parameter(torch.zeros(roles,d))
  elif method=='block_independent':self.kernels=nn.Parameter(torch.randn(roles,self.nb,self.nb,block)*.08);self.bias=nn.Parameter(torch.zeros(roles,d))
  else:
   shape=(self.nb,self.nb,block) if method.startswith('block_') else (d,d)
   self.kernels=nn.Parameter(torch.randn(*shape)*.08) if method.startswith('block_') else nn.Parameter(torch.randn(d,d)*.08)
   self.bias=nn.Parameter(torch.zeros(d))
   if method=='block_givens':self.angles=nn.Parameter(torch.zeros(roles,d//2))
   if method=='dense_lowrank':self.l=nn.Parameter(torch.zeros(roles,d,rank));self.r=nn.Parameter(torch.randn(roles,rank,d)*.02)
  if method in ('block_map','block_hadamard','block_hrr'):
   self.register_buffer('address',address_matrices(address_seed,method,roles,d))
  if method=='block_shift':self.register_buffer('shifts',shift_codes(address_seed,roles))
  if method=='block_givens':self.register_buffer('address_seed_tensor',torch.tensor([address_seed],dtype=torch.int64))

 def matrix(self,role):
  if self.method=='untied':return self.weight[role]
  if self.method=='block_independent':return block_matrix(self.kernels[role])
  if self.method=='block_shift':return block_matrix(self.kernels,self.shifts[role])
  if self.method.startswith('block_'):return block_matrix(self.kernels)
  if self.method=='dense_lowrank':return self.kernels+self.l[role]@self.r[role]
  return self.kernels

 def forward(self,x,role):
  if self.method=='untied':return torch.einsum('bi,bij->bj',x,self.weight[role])+self.bias[role]
  if self.method in ('block_independent','block_shift'):
   mats=torch.stack([self.matrix(i) for i in range(self.roles)])
   b=self.bias[role] if self.bias.ndim==2 else self.bias
   return torch.einsum('bi,bij->bj',x,mats[role])+b
  if self.method=='dense_lowrank':
   hidden=torch.bmm(x[:,None,:],self.l[role]).squeeze(1)
   residual=torch.bmm(hidden[:,None,:],self.r[role]).squeeze(1)
   return x@self.kernels+residual+self.bias
  if self.method in ('block_map','block_hadamard','block_hrr'):x=torch.bmm(x[:,None,:],self.address[role]).squeeze(1)
  elif self.method=='block_givens':x=givens(x,self.angles[role])
  w=self.matrix(0 if self.method in ('block_tied','block_map','block_hadamard','block_hrr','block_givens') else role)
  b=self.bias if self.bias.ndim==1 else self.bias[role]
  return x@w+b

 def serialize(self):
  cfg={'method':self.method,'address_seed':self.address_seed,'roles':self.roles,'d':self.d,'block':self.block,'rank':self.rank,'structure':'block-circulant-v1; all registered buffers included'}
  f=io.BytesIO();torch.save({'state_dict':self.state_dict(),'config':cfg},f);return f.getvalue()
 @staticmethod
 def from_serialized(payload):
  x=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False);c=x['config'];m=BlockViews(c['method'],c['address_seed'],c['roles'],c['d'],c['block'],c['rank']);m.load_state_dict(x['state_dict']);return m
 def serialized_payload_bytes(self):return len(self.serialize())

def compute_proxy(method,examples,roles=ROLES,d=D,block=BLOCK):
 if method=='untied':per=roles*d*d
 elif method=='block_independent':per=roles*(d//block)**2*block*int(math.log2(block))*2
 elif method=='dense_lowrank':per=d*d+roles*2*d
 elif method=='block_shift':per=(d//block)**2*block*int(math.log2(block))*2*roles+roles*(d//block)**2
 elif method.startswith('block_'):per=(d//block)**2*block*int(math.log2(block))*2*roles
 else:per=roles*d*d
 return int(examples*per)
