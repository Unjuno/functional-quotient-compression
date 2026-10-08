from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
METHODS=('shared','mirror','film4','film8','rank1','independent')
def transform(h,method,code):
 if method=='mirror':
  p=h[:,:8].reshape(h.shape[0],4,2);a,b=p[...,0],p[...,1];c=torch.cos(code['angles'])[None,:];s=torch.sin(code['angles'])[None,:];r=torch.stack((c*a-s*b,s*a+c*b),-1).reshape(h.shape[0],8);return torch.cat((r,h[:,8:]),1)
 if method in ('film4','film8'):
  scale=code['scale'].repeat_interleave(2);bias=torch.zeros_like(scale) if method=='film4' else code['bias'].repeat_interleave(2)
  return torch.cat((h[:,:8]*scale+bias,h[:,8:]),1)
 return h
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.fc=nn.Linear(64,64);self.head=nn.Linear(64,10)
 def hidden(self,x):return F.relu(self.fc(x))
 def forward(self,x,method='shared',code=None):
  h=self.hidden(x)
  if code is not None:h=transform(h,method,code)
  return self.head(h)
class Bank(nn.Module):
 def __init__(self):super().__init__();self.nets=nn.ModuleList([Net() for _ in range(4)])
 def forward(self,x,ctx):return self.nets[ctx](x)
