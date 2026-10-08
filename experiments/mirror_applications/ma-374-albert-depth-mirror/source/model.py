from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
METHODS=('untied','tied','attention_shared','ffn_shared','mirror','film')
def rotate(h,a):
 p=h[:,:8].reshape(h.shape[0],4,2);x,y=p[...,0],p[...,1];c=torch.cos(a)[None,:];s=torch.sin(a)[None,:];z=torch.stack((c*x-s*y,s*x+c*y),dim=-1).reshape(h.shape[0],8);return torch.cat((z,h[:,8:]),1)
def affine(h,s):return torch.cat((h[:,:8]*s.repeat_interleave(2),h[:,8:]),1)
class Block(nn.Module):
 def __init__(self):
  super().__init__();self.a=nn.Linear(64,64);self.f1=nn.Linear(64,128);self.f2=nn.Linear(128,64)
 def forward(self,h):
  h=F.relu(h+self.a(h));return F.relu(h+self.f2(F.relu(self.f1(h))))
class DepthNet(nn.Module):
 def __init__(self,method):
  super().__init__();self.method=method
  na=1 if method in ('tied','ffn_shared','mirror','film') else 3
  nf=1 if method in ('tied','attention_shared','mirror','film') else 3
  self.att=nn.ModuleList([nn.Linear(64,64) for _ in range(na)])
  self.f1=nn.ModuleList([nn.Linear(64,128) for _ in range(nf)]);self.f2=nn.ModuleList([nn.Linear(128,64) for _ in range(nf)])
  self.head=nn.Linear(64,10)
 def forward(self,x,codes=None):
  h=x
  for d in range(3):
   ai=0 if len(self.att)==1 else d;fi=0 if len(self.f1)==1 else d
   h=F.relu(h+self.att[ai](h));h=F.relu(h+self.f2[fi](F.relu(self.f1[fi](h))))
   if codes is not None:
    if self.method=='mirror':h=rotate(h,codes[d])
    else:h=affine(h,codes[d])
  return self.head(h)
