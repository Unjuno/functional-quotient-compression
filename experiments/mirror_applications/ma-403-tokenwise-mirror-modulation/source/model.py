from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
METHODS=('shared','mirror_gen','film_gen','static','independent')
def rotate(h,angles):
 p=h[:,:8].reshape(h.shape[0],4,2);a,b=p[...,0],p[...,1];c=torch.cos(angles);s=torch.sin(angles);r=torch.stack((c*a-s*b,s*a+c*b),-1).reshape(h.shape[0],8);return torch.cat((r,h[:,8:]),1)
class ContextNet(nn.Module):
 def __init__(self,method):
  super().__init__();self.method=method;self.fc=nn.Linear(16,32);self.head=nn.Linear(32,4)
  if method=='mirror_gen':self.generator=nn.Linear(36,4)
  elif method=='film_gen':self.generator=nn.Linear(36,8)
  elif method=='static':self.angles=nn.Parameter(torch.zeros(4,4))
 def hidden(self,x):return F.relu(self.fc(x))
 def forward(self,x,ctx):
  h=self.hidden(x)
  if self.method=='mirror_gen':
   one=F.one_hot(ctx,4).to(h.dtype);a=self.generator(torch.cat((h,one),-1));h=rotate(h,a)
  elif self.method=='film_gen':
   one=F.one_hot(ctx,4).to(h.dtype);z=self.generator(torch.cat((h,one),-1));scale=1+z[:,:4];bias=z[:,4:];h=torch.cat((h[:,:8]*scale.repeat_interleave(2,1)+bias.repeat_interleave(2,1),h[:,8:]),1)
  elif self.method=='static':h=rotate(h,self.angles[ctx])
  return self.head(h)
class Independent(nn.Module):
 def __init__(self):super().__init__();self.nets=nn.ModuleList([ContextNet('shared') for _ in range(4)])
 def forward(self,x,ctx):
  out=torch.empty((len(x),4),dtype=x.dtype,device=x.device)
  for c,m in enumerate(self.nets):
   mask=ctx==c
   if mask.any():out[mask]=m(x[mask],ctx[mask])
  return out
