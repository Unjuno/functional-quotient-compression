from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from itertools import product
WIDTHS=(16,32,64)
CONFIGS=tuple(product(WIDTHS,repeat=3))
MIXED=tuple(c for c in CONFIGS if len(set(c))>1)
METHODS=('nested','factor_mirror','factor_film','direct_mirror')
def transform(h,kind,code):
 if kind in ('factor_mirror','direct_mirror'):
  p=h[:,:8].reshape(h.shape[0],4,2);a,b=p[...,0],p[...,1];c=torch.cos(code)[None,:];s=torch.sin(code)[None,:]
  u=torch.stack((c*a-s*b,s*a+c*b),dim=-1).reshape(h.shape[0],8);return torch.cat((u,h[:,8:]),dim=1)
 if kind=='factor_film':return torch.cat((h[:,:8]*code.repeat_interleave(2),h[:,8:]),dim=1)
 return h
class Nested3(nn.Module):
 def __init__(self):
  super().__init__();self.ws=nn.ParameterList([nn.Parameter(torch.empty(64,64)) for _ in range(3)]);self.bs=nn.ParameterList([nn.Parameter(torch.zeros(64)) for _ in range(3)]);self.head=nn.Parameter(torch.empty(10,64));self.hb=nn.Parameter(torch.zeros(10))
  for w in self.ws:nn.init.kaiming_uniform_(w,a=5**.5)
  nn.init.kaiming_uniform_(self.head,a=5**.5)
 def forward(self,x,cfg,kind='nested',codes=None):
  h=x;prev=64
  for i,width in enumerate(cfg):
   h=F.relu(F.linear(h,self.ws[i][:width,:prev],self.bs[i][:width]));prev=width
   if codes is not None:h=transform(h,kind,codes[i])
  return F.linear(h,self.head[:,:cfg[-1]],self.hb)
def macs(cfg):
 d=64;total=0
 for w in cfg:total+=d*w;d=w
 return total+d*10
def adapt_macs(kind):return 3*24 if kind=='factor_mirror' else 3*8 if kind=='factor_film' else 3*24*3
