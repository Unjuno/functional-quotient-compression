from __future__ import annotations
import io,json,math,torch
from torch import nn
H,D,HD,L=4,16,4,8
METHODS=['full_mha','tied_qkv','head_gain','rank1_residual','mirror_view']
def givens(x,angles):
 y=x
 for k in range(D//2):
  i,j=2*k,2*k+1;c,s=torch.cos(angles[...,k]),torch.sin(angles[...,k]);a,b=y[...,i],y[...,j];u,v=c*a-s*b,s*a+c*b;y=torch.cat((y[...,:i],u.unsqueeze(-1),v.unsqueeze(-1),y[...,j+1:]),dim=-1)
 return y
class AttentionHeads(nn.Module):
 def __init__(self,method,seed=0):
  super().__init__();self.method=method;g=torch.Generator().manual_seed(seed)
  if method=='full_mha':self.w=nn.Parameter(torch.randn(H,3,D,HD,generator=g)*.08)
  else:
   self.w=nn.Parameter(torch.randn(3,D,HD,generator=g)*.08)
   if method=='head_gain':self.gain=nn.Parameter(torch.ones(H,3,HD))
   elif method=='rank1_residual':self.a=nn.Parameter(torch.randn(H,3,D,1,generator=g)*.04);self.b=nn.Parameter(torch.zeros(H,3,1,HD))
   elif method=='mirror_view':self.angle=nn.Parameter(torch.zeros(H,D//2))
  self.out=nn.Parameter(torch.randn(D,D,generator=g)*.08)
 def head_contributions(self,x):
  if self.method=='full_mha':qkv=torch.einsum('bld,hkdm->bhklm',x,self.w)
  elif self.method=='mirror_view':
   xv=torch.stack([givens(x,a.expand(x.shape[0],x.shape[1],-1)) for a in self.angle],dim=1);qkv=torch.einsum('bhld,kdm->bhklm',xv,self.w)
  else:
   w=self.w[None].expand(H,-1,-1,-1)
   if self.method=='head_gain':w=w*self.gain.unsqueeze(-2)
   elif self.method=='rank1_residual':w=w+torch.einsum('hkdr,hkrm->hkdm',self.a,self.b)
   qkv=torch.einsum('bld,hkdm->bhklm',x,w)
  q,k,v=qkv[:,:,0],qkv[:,:,1],qkv[:,:,2]
  att=torch.softmax(torch.einsum('bhlm,bhtm->bhlt',q,k)/math.sqrt(HD),dim=-1)
  ctx=torch.einsum('bhlt,bhtm->bhlm',att,v).transpose(1,2)
  return torch.stack([ctx[:,:,i]@self.out[i*HD:(i+1)*HD] for i in range(H)],dim=2)
 def forward(self,x):return self.head_contributions(x).sum(dim=2)
 def serialized_payload_bytes(self):
  b=io.BytesIO();torch.save({'state_dict':self.state_dict(),'config':json.dumps({'method':self.method,'heads':H,'d':D,'head_dim':HD,'seq':L},sort_keys=True)},b);return len(b.getvalue())
def compute_proxy(method,examples):
 projection=(H*3*D*HD if method=='full_mha' else 3*D*HD+H*D*2 if method=='mirror_view' else 3*D*HD+(H*3*D if method=='rank1_residual' else H*3*HD if method=='head_gain' else 0))
 attention=2*H*L*L*HD+H*L*L
 output=D*D
 return examples*(projection+attention+output)
