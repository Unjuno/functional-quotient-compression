from __future__ import annotations
import io,json,math,torch
from torch import nn
H,P,D,HD,L=16,4,32,2,6
METHODS=['full_mha','hard_gqa','rank1_residual','mirror_views']
GROUP=torch.arange(H)%P
def givens(x,angles):
 y=x
 for k in range(D//2):
  i,j=2*k,2*k+1;c,s=torch.cos(angles[...,k]),torch.sin(angles[...,k]);a,b=y[...,i],y[...,j];u,v=c*a-s*b,s*a+c*b;y=torch.cat((y[...,:i],u.unsqueeze(-1),v.unsqueeze(-1),y[...,j+1:]),dim=-1)
 return y
class ExpandedAttention(nn.Module):
 def __init__(self,method,seed=0):
  super().__init__();self.method=method;g=torch.Generator().manual_seed(seed)
  n=H if method=='full_mha' else P;self.w=nn.Parameter(torch.randn(n,3,D,HD,generator=g)*.06)
  if method=='rank1_residual':self.a=nn.Parameter(torch.randn(H,3,D,1,generator=g)*.03);self.b=nn.Parameter(torch.zeros(H,3,1,HD))
  if method=='mirror_views':self.angle=nn.Parameter(torch.zeros(H,D//2))
  self.out=nn.Parameter(torch.randn(D,D,generator=g)*.06)
 def head_contributions(self,x):
  if self.method=='full_mha':qkv=torch.einsum('bld,hkdm->bhklm',x,self.w)
  elif self.method=='mirror_views':
   xv=torch.stack([givens(x,a.expand(x.shape[0],x.shape[1],-1)) for a in self.angle],dim=1);qkv=torch.einsum('bhld,hkdm->bhklm',xv,self.w[GROUP])
  else:
   w=self.w[GROUP]
   if self.method=='rank1_residual':w=w+torch.einsum('hkdr,hkrm->hkdm',self.a,self.b)
   qkv=torch.einsum('bld,hkdm->bhklm',x,w)
  q,k,v=qkv[:,:,0],qkv[:,:,1],qkv[:,:,2];att=torch.softmax(torch.einsum('bhlm,bhtm->bhlt',q,k)/math.sqrt(HD),dim=-1);ctx=torch.einsum('bhlt,bhtm->bhlm',att,v).transpose(1,2)
  return torch.stack([ctx[:,:,h]@self.out[h*HD:(h+1)*HD] for h in range(H)],dim=2)
 def forward(self,x):return self.head_contributions(x).sum(dim=2)
 def serialized_payload_bytes(self):
  b=io.BytesIO();torch.save({'state_dict':self.state_dict(),'config':json.dumps({'method':self.method,'physical_heads':P,'logical_heads':H,'d':D,'head_dim':HD,'seq':L},sort_keys=True)},b);return len(b.getvalue())
def compute_proxy(method,sequences):
 proj=(H if method=='full_mha' else P)*3*D*HD*L
 if method=='mirror_views':proj+=H*D*L
 if method=='rank1_residual':proj+=H*3*(D+HD)*L
 att=2*H*L*L*HD+H*L*L;out=D*D*L
 return sequences*(proj+att+out)
