from __future__ import annotations
import io,json,math,torch
from torch import nn
H,D,HD,L=4,16,4,16
METHODS=['mha','gqa2','mqa','rank1_kv','mirror_kv']
def rotate_heads(z,angle):
 # z [B,H,L,HD], angle [H,HD/2]; pairwise Givens rotation with broadcasting.
 pairs=z.reshape(*z.shape[:-1],HD//2,2);a,b=pairs[...,0],pairs[...,1];c=torch.cos(angle)[None,:,None,:];s=torch.sin(angle)[None,:,None,:]
 return torch.stack((c*a-s*b,s*a+c*b),dim=-1).flatten(-2)
def causal_weights(q,k):
 score=torch.einsum('bhlm,bhtm->bhlt',q,k)/math.sqrt(HD);mask=torch.ones(L,L,dtype=torch.bool,device=q.device).tril();return torch.softmax(score.masked_fill(~mask,-1e9),dim=-1)
class CausalKV(nn.Module):
 def __init__(self,method,seed=0):
  super().__init__();self.method=method;g=torch.Generator().manual_seed(seed);n=H if method=='mha' else 2 if method=='gqa2' else 1
  self.q=nn.Parameter(torch.randn(H,D,HD,generator=g)*.08);self.k=nn.Parameter(torch.randn(n,D,HD,generator=g)*.08);self.v=nn.Parameter(torch.randn(n,D,HD,generator=g)*.08);self.out=nn.Parameter(torch.randn(D,D,generator=g)*.08)
  if method=='rank1_kv':self.a=nn.Parameter(torch.randn(H,2,D,1,generator=g)*.03);self.b=nn.Parameter(torch.zeros(H,2,1,HD))
  if method=='mirror_kv':self.ka=nn.Parameter(torch.zeros(H,HD//2));self.va=nn.Parameter(torch.zeros(H,HD//2))
 def head_contributions(self,x):
  q=torch.einsum('bld,hdm->bhlm',x,self.q);ids=torch.arange(H,device=x.device) if self.method=='mha' else torch.arange(H,device=x.device)%2 if self.method=='gqa2' else torch.zeros(H,dtype=torch.long,device=x.device);wk,wv=self.k[ids],self.v[ids]
  if self.method=='rank1_kv':wk=wk+torch.einsum('hdr,hrm->hdm',self.a[:,0],self.b[:,0]);wv=wv+torch.einsum('hdr,hrm->hdm',self.a[:,1],self.b[:,1])
  k=torch.einsum('bld,hdm->bhlm',x,wk);v=torch.einsum('bld,hdm->bhlm',x,wv)
  if self.method=='mirror_kv':k=rotate_heads(k,self.ka);v=rotate_heads(v,self.va)
  att=causal_weights(q,k);ctx=torch.einsum('bhlt,bhtm->bhlm',att,v).transpose(1,2)
  return torch.stack([ctx[:,:,h]@self.out[h*HD:(h+1)*HD] for h in range(H)],dim=2)
 def forward(self,x):return self.head_contributions(x).sum(dim=2)
 def serialized_payload_bytes(self):
  b=io.BytesIO();torch.save({'state_dict':self.state_dict(),'config':json.dumps({'method':self.method,'d':D,'heads':H,'head_dim':HD,'seq':L,'causal':True},sort_keys=True)},b);return len(b.getvalue())
def physical_kv_heads(method):return H if method=='mha' else 2 if method=='gqa2' else 1
def cache_bytes(method,batch=1,tokens=L):return 2*physical_kv_heads(method)*HD*tokens*4*batch
def compute_proxy(method,sequences):
 proj=H*D*HD+2*physical_kv_heads(method)*D*HD
 if method=='rank1_kv':proj+=H*2*(D+HD)
 att=2*H*L*(L+1)/2*HD+H*L*(L+1)/2;out=D*D*L
 return int(sequences*(proj*L+att+out))
