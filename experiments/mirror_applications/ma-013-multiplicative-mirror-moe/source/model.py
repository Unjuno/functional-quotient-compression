import torch
from torch import nn
from torch.nn import functional as F

DIN,HID,DOUT,ROLES=16,32,16,4
METHODS=('independent','tied','film','rank2','multiplicative')

class MultiplicativeExperts(nn.Module):
 def __init__(self,method,seed):
  super().__init__();self.method=method
  if method not in METHODS:raise ValueError(method)
  torch.manual_seed(seed)
  if method=='independent':
   self.wi=nn.Parameter(torch.randn(ROLES,HID,DIN)*.12);self.wo=nn.Parameter(torch.randn(ROLES,DOUT,HID)*.12)
  else:
   self.wi=nn.Parameter(torch.randn(HID,DIN)*.12);self.wo=nn.Parameter(torch.randn(DOUT,HID)*.12)
   if method=='film':self.scale=nn.Parameter(torch.ones(ROLES,HID))
   elif method=='rank2':self.left=nn.Parameter(torch.randn(ROLES,DOUT,2)*.02);self.right=nn.Parameter(torch.randn(ROLES,2,HID)*.02)
   elif method=='multiplicative':
    self.u=nn.Parameter(torch.randn(HID)*.02);self.v=nn.Parameter(torch.randn(HID)*.02)
    self.alpha=nn.Parameter(torch.ones(2));self.beta=nn.Parameter(torch.ones(2))
 def forward(self,x,r):
  if self.method=='independent':wi=self.wi[r];wo=self.wo[r]
  else:wi=self.wi.expand(len(x),-1,-1);wo=self.wo.expand(len(x),-1,-1)
  h=F.gelu(torch.bmm(wi,x.unsqueeze(-1)).squeeze(-1))
  if self.method=='film':h=h*self.scale[r]
  elif self.method=='multiplicative':
   a=self.alpha[r//2,None];b=self.beta[r%2,None]
   h=h*((1+a*self.u)*(1+b*self.v))
  y=torch.bmm(wo,h.unsqueeze(-1)).squeeze(-1)
  if self.method=='rank2':y=y+torch.bmm(self.left[r],torch.bmm(self.right[r],h.unsqueeze(-1))).squeeze(-1)
  return y
