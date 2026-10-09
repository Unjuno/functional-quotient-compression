"""RoPE frequency sharing and Mirror coordinate screen for MA-116."""
import io,math,time
from dataclasses import dataclass
import torch
from torch import nn

DOMAINS,FREQS=4,4
TRAIN_POS=16
TEST_POS=torch.arange(16,128,dtype=torch.float32)
BASE=torch.tensor([0.8,0.4,0.2,0.1])
ADDRESSES=torch.tensor([-0.50,-0.18,0.18,0.50])
DIRECTION=torch.tensor([-0.60,-0.20,0.20,0.60])
METHODS=('fixed_rope','scalar_scaled_rope','mirror_rope','independent_rope')

def world(seed):
 g=torch.Generator().manual_seed(seed)
 jitter=torch.randn(FREQS,generator=g)*.002
 address_jitter=torch.randn(DOMAINS,generator=g)*.001
 omega=BASE[None,:]*torch.exp(jitter[None,:]+(ADDRESSES+address_jitter)[:,None]*DIRECTION[None,:])
 pos=torch.arange(TRAIN_POS,dtype=torch.float32)
 target=torch.stack((torch.cos(omega[:,:,None]*pos),torch.sin(omega[:,:,None]*pos)),dim=-1)
 test=torch.stack((torch.cos(omega[:,:,None]*TEST_POS),torch.sin(omega[:,:,None]*TEST_POS)),dim=-1)
 return target,test

class RopeModel(nn.Module):
 def __init__(self,method):
  super().__init__();self.method=method
  if method=='scalar_scaled_rope':self.logscale=nn.Parameter(torch.zeros(DOMAINS))
  elif method=='mirror_rope':
   self.address=nn.Parameter(torch.zeros(DOMAINS));self.direction=nn.Parameter(torch.ones(FREQS))
  elif method=='independent_rope':self.logfreq=nn.Parameter(torch.log(BASE).repeat(DOMAINS,1))
 def frequencies(self):
  if self.method=='fixed_rope':return BASE[None,:].expand(DOMAINS,-1)
  if self.method=='scalar_scaled_rope':return BASE[None,:]*self.logscale.exp()[:,None]
  if self.method=='mirror_rope':return BASE[None,:]*torch.exp(self.address[:,None]*self.direction[None,:])
  return self.logfreq.exp()
 def forward(self,positions):
  phase=self.frequencies()[:,:,None]*positions[None,None,:]
  return torch.stack((torch.cos(phase),torch.sin(phase)),dim=-1)

def payload(model):
 b=io.BytesIO();torch.save({'method':model.method,'base_frequencies':BASE,'state_dict':model.state_dict()},b);return b.getvalue()

@dataclass
class Result:
 method:str;seed:int;train_mse:float;test_mse:float;payload_bytes:int;updates:int;examples:int;macs:int;wall:float;throughput:float

def train_one(method,seed,updates=800,lr=.02):
 torch.manual_seed(seed+116);train,test=world(seed);pos=torch.arange(TRAIN_POS,dtype=torch.float32);m=RopeModel(method);params=list(m.parameters());opt=torch.optim.Adam(params,lr=lr) if params else None;t=time.perf_counter()
 for _ in range(updates if opt else 0):
  loss=(m(pos)-train).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 wall=time.perf_counter()-t
 with torch.no_grad():
  tr=float((m(pos)-train).square().mean());te=float((m(TEST_POS)-test).square().mean())
 # Four domains x four rotary pairs x positions x sin/cos. MAC proxy counts phase generation.
 actual_updates=updates if opt else 0
 examples=DOMAINS*TRAIN_POS*FREQS
 macs=actual_updates*examples*4
 if method=='mirror_rope':macs+=updates*DOMAINS*FREQS
 elif method=='scalar_scaled_rope':macs+=updates*DOMAINS*FREQS
 return Result(method,seed,tr,te,len(payload(m)),actual_updates,actual_updates*examples,int(macs),wall,actual_updates*examples/wall if wall else 0.)

def run(seed,updates=800,lr=.02):return [train_one(m,seed,updates,lr) for m in METHODS]
