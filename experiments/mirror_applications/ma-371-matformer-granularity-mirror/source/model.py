from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
WIDTHS=(16,32,64)
METHODS=('nested','mirror','film','lora','independent')
def givens(h, angles):
 p=h[:,:8].reshape(h.shape[0],4,2);a,b=p[...,0],p[...,1];c=torch.cos(angles)[None,:];s=torch.sin(angles)[None,:]
 r=torch.stack((c*a-s*b,s*a+c*b),dim=-1).reshape(h.shape[0],8)
 return torch.cat((r,h[:,8:]),dim=1)
def modulate(h,method,code):
 if method=='mirror': return givens(h,code if torch.is_tensor(code) else code['angles'])
 if method=='film':
  scales=code if torch.is_tensor(code) else code['scales']
  return torch.cat((h[:,:8]*scales.repeat_interleave(2),h[:,8:]),dim=1)
 return h
class NestedNet(nn.Module):
 def __init__(self):
  super().__init__();self.w1=nn.Parameter(torch.empty(64,64));self.b1=nn.Parameter(torch.zeros(64));self.w2=nn.Parameter(torch.empty(64,64));self.b2=nn.Parameter(torch.zeros(64));self.head=nn.Parameter(torch.empty(10,64));self.hb=nn.Parameter(torch.zeros(10))
  for x in (self.w1,self.w2,self.head): nn.init.kaiming_uniform_(x,a=5**.5)
 def hidden(self,x,width,layer_codes=None,method='nested'):
  h=F.relu(F.linear(x,self.w1[:width],self.b1[:width]))
  if layer_codes is not None: h=modulate(h,method,layer_codes['l1'])
  h=F.relu(F.linear(h,self.w2[:width,:width],self.b2[:width]))
  if layer_codes is not None: h=modulate(h,method,layer_codes['l2'])
  return h
 def forward(self,x,width,method='nested',codes=None):
  h=self.hidden(x,width,codes,method)
  return F.linear(h,self.head[:,:width],self.hb)
class Independent(nn.Module):
 def __init__(self,width):
  super().__init__();self.l1=nn.Linear(64,width);self.l2=nn.Linear(width,width);self.head=nn.Linear(width,10)
 def forward(self,x): return self.head(F.relu(self.l2(F.relu(self.l1(x)))))
def macs(width): return 64*width+width*width+width*10
def adapt_macs(method,width): return {'mirror':48,'film':16,'lora':width+10}.get(method,0)
