from __future__ import annotations
import io,json,math,torch
from torch import nn
H,D,HD,L=4,16,4,8
METHODS=['mha','gqa2','mqa','rank1_kv','mirror_kv']
def rot(z,a):
 y=z
 for i in range(HD//2):
  j=i*2;k=j+1;c,s=torch.cos(a[...,i]),torch.sin(a[...,i]);x,y0=y[...,j],y[...,k];u,v=c*x-s*y0,s*x+c*y0;y=torch.cat((y[...,:j],u.unsqueeze(-1),v.unsqueeze(-1),y[...,k+1:]),dim=-1)
 return y
class KVAttention(nn.Module):
 def __init__(self,method,seed=0):
  super().__init__();self.method=method;g=torch.Generator().manual_seed(seed);n=H if method=='mha' else 2 if method=='gqa2' else 1
  self.q=nn.Parameter(torch.randn(H,D,HD,generator=g)*.08);self.k=nn.Parameter(torch.randn(n,D,HD,generator=g)*.08);self.v=nn.Parameter(torch.randn(n,D,HD,generator=g)*.08);self.out=nn.Parameter(torch.randn(D,D,generator=g)*.08)
  if method=='rank1_kv':self.a=nn.Parameter(torch.randn(H,2,D,1,generator=g)*.04);self.b=nn.Parameter(torch.zeros(H,2,1,HD))
  if method=='mirror_kv':self.ka=nn.Parameter(torch.zeros(H,HD//2));self.va=nn.Parameter(torch.zeros(H,HD//2))
 def head_contrib(self,x):
  q=torch.einsum('bld,hdm->bhlm',x,self.q)
  ids=torch.arange(H,device=x.device) if self.method=='mha' else torch.arange(H,device=x.device)%2 if self.method=='gqa2' else torch.zeros(H,dtype=torch.long,device=x.device)
  wk,wv=self.k[ids],self.v[ids]
  if self.method=='rank1_kv':
   wk=wk+torch.einsum('hdr,hrm->hdm',self.a[:,0],self.b[:,0]);wv=wv+torch.einsum('hdr,hrm->hdm',self.a[:,1],self.b[:,1])
  k=torch.einsum('bld,hdm->bhlm',x,wk);v=torch.einsum('bld,hdm->bhlm',x,wv)
  if self.method=='mirror_kv':k=torch.stack([rot(k[:,h],self.ka[h].expand(x.shape[0],x.shape[1],-1)) for h in range(H)],dim=1);v=torch.stack([rot(v[:,h],self.va[h].expand(x.shape[0],x.shape[1],-1)) for h in range(H)],dim=1)
  a=torch.softmax(torch.einsum('bhlm,bhtm->bhlt',q,k)/math.sqrt(HD),dim=-1);c=torch.einsum('bhlt,bhtm->bhlm',a,v).transpose(1,2)
  return torch.stack([c[:,:,h]@self.out[h*HD:(h+1)*HD] for h in range(H)],dim=2)
 def forward(self,x):return self.head_contrib(x).sum(dim=2)
 def serialized_payload_bytes(self):
  b=io.BytesIO();torch.save({'state_dict':self.state_dict(),'config':json.dumps({'method':self.method,'d':D,'heads':H,'head_dim':HD,'seq':L},sort_keys=True)},b);return len(b.getvalue())
def physical_kv_heads(method):return H if method=='mha' else 2 if method=='gqa2' else 1
def cache_bytes(method,batch=1,tokens=L):return 2*physical_kv_heads(method)*HD*tokens*4*batch
def compute_proxy(method,sequences):
 n=physical_kv_heads(method);proj=H*D*HD+2*n*D*HD
 if method=='rank1_kv':proj+=H*2*(D+HD)
 att=2*H*L*L*HD+H*L*L;out=D*D*L
 return sequences*(proj*L+att+out)
