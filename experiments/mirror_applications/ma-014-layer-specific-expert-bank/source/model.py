import torch
from torch import nn
from torch.nn import functional as F

DIN,HID,DOUT,LAYERS,EXPERTS=8,16,8,4,4
METHODS=('independent','tied','gate','rank1','rank2','mirror')

def rotate(angle,dim,transpose=False):
 c,s=torch.cos(angle),torch.sin(angle)
 if transpose:s=-s
 r=torch.eye(dim,dtype=angle.dtype,device=angle.device).expand(len(angle),dim,dim).clone()
 r[:,0,0]=c;r[:,0,1]=-s;r[:,1,0]=s;r[:,1,1]=c
 return r

class LayerExpertBank(nn.Module):
 def __init__(self,method,seed):
  super().__init__();self.method=method;torch.manual_seed(seed)
  if method=='independent':
   self.wi=nn.Parameter(torch.randn(LAYERS,EXPERTS,HID,DIN)*.12);self.wo=nn.Parameter(torch.randn(LAYERS,EXPERTS,DOUT,HID)*.12)
  else:
   self.wi=nn.Parameter(torch.randn(EXPERTS,HID,DIN)*.12);self.wo=nn.Parameter(torch.randn(EXPERTS,DOUT,HID)*.12)
   if method=='gate':self.gate=nn.Parameter(torch.ones(LAYERS,EXPERTS))
   if method in ('rank1','rank2'):
    rank=int(method[-1]);self.left=nn.Parameter(torch.randn(LAYERS,EXPERTS,DOUT,rank)*.02);self.right=nn.Parameter(torch.randn(LAYERS,EXPERTS,rank,HID)*.02)
   if method=='mirror':self.input_angle=nn.Parameter(torch.zeros(LAYERS));self.output_angle=nn.Parameter(torch.zeros(LAYERS))
 def forward(self,x,layer,expert):
  if self.method=='independent':wi=self.wi[layer,expert];wo=self.wo[layer,expert]
  else:wi=self.wi[expert];wo=self.wo[expert]
  q=x
  if self.method=='mirror':q=torch.bmm(rotate(self.input_angle,DIN,True)[layer],x.unsqueeze(-1)).squeeze(-1)
  h=F.gelu(torch.bmm(wi,q.unsqueeze(-1)).squeeze(-1));y=torch.bmm(wo,h.unsqueeze(-1)).squeeze(-1)
  if self.method=='gate':y=y*self.gate[layer,expert,None]
  elif self.method in ('rank1','rank2'):y=y+torch.bmm(self.left[layer,expert],torch.bmm(self.right[layer,expert],h.unsqueeze(-1))).squeeze(-1)
  if self.method=='mirror':y=torch.bmm(rotate(self.output_angle,DOUT)[layer],y.unsqueeze(-1)).squeeze(-1)
  return y
