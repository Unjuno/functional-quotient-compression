from __future__ import annotations
import hashlib,io,time
import numpy as np
import torch
from torch import nn
D,H,M=16,32,4;PAIRS=[(2*i,2*i+1) for i in range(H//2)]
torch.set_num_threads(1)
def serialize(state):
 b=io.BytesIO();torch.save({'state':state,'config':{'D':D,'H':H,'M':M}},b);v=b.getvalue();return len(v),hashlib.sha256(v).hexdigest()
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.w1=nn.Parameter(torch.empty(D,H));self.b1=nn.Parameter(torch.zeros(H));self.w2=nn.Parameter(torch.empty(H,1));self.b2=nn.Parameter(torch.zeros(1));nn.init.xavier_uniform_(self.w1);nn.init.xavier_uniform_(self.w2)
 def forward(self,x):return torch.tanh(x@self.w1+self.b1)@self.w2+self.b2
class Packed(nn.Module):
 def __init__(self,w1,b1,w2,b2):
  super().__init__();self.w1=nn.Parameter(w1.repeat(M,1,1));self.b1=nn.Parameter(b1.repeat(M,1));self.w2=nn.Parameter(w2.repeat(M,1,1));self.b2=nn.Parameter(b2.repeat(M,1))
 def forward(self,x):
  h=torch.tanh(torch.einsum('mnd,mdh->mnh',x,self.w1)+self.b1[:,None,:]);return torch.einsum('mnh,mho->mno',h,self.w2)+self.b2[:,None,:]
class Shared(nn.Module):
 def __init__(self,kind,w1,b1,w2,b2):
  super().__init__();self.w1=nn.Parameter(w1.clone());self.b1=nn.Parameter(b1.clone());self.w2=nn.Parameter(w2.clone());self.b2=nn.Parameter(b2.clone());self.kind=kind
  if kind=='film':self.code=nn.Parameter(torch.ones(M,H))
  else:self.code=nn.Parameter(torch.zeros(M,H//2))
 def forward_all(self,x):
  h=torch.tanh(x@self.w1+self.b1)
  if self.kind=='film':h=h*self.code[:,None,:]
  else:
   pair=h.reshape(M,h.shape[1],H//2,2);u=pair[...,0];v=pair[...,1]
   c=torch.cos(self.code)[:,None,:];s=torch.sin(self.code)[:,None,:]
   h=torch.stack((c*u-s*v,s*u+c*v),dim=-1).reshape(M,h.shape[1],H)
  return (h@self.w2+self.b2).squeeze(-1)
 def forward(self,x,member):return self.forward_all(x[None])[member]
def run(seed=72501):
 torch.manual_seed(seed);rng=np.random.default_rng(seed)
 W1=torch.randn(D,H)*.6;b1=torch.randn(H)*.2;W2=torch.randn(H,1)*.4;b2=torch.tensor([.05])
 angles=torch.tensor(rng.normal(0,.55,size=(M,H//2)),dtype=torch.float32)
 def teacher(x,m):
  h=torch.tanh(x@W1+b1);z=h.clone()
  for j,(a,b) in enumerate(PAIRS):
   c=torch.cos(angles[m,j]);s=torch.sin(angles[m,j]);u=h[:,a];v=h[:,b];z[:,a]=c*u-s*v;z[:,b]=s*u+c*v
  return z@W2+b2
 Xtr=torch.randn(M,512,D);Xte=torch.randn(M,512,D);Ytr=torch.stack([teacher(Xtr[m],m).squeeze(-1) for m in range(M)]);Yte=torch.stack([teacher(Xte[m],m).squeeze(-1) for m in range(M)])
 # Packed vectorized training stores independent member parameters but batches the member axis.
 packed=Packed(W1,b1,W2,b2);opt=torch.optim.Adam(packed.parameters(),lr=.01)
 for step in range(400):
  ix=torch.randint(0,512,(M,128));xb=torch.stack([Xtr[m,ix[m]] for m in range(M)]);yb=torch.stack([Ytr[m,ix[m]] for m in range(M)])
  pred=packed(xb).squeeze(-1);loss=((pred-yb)**2).mean();opt.zero_grad();loss.backward();opt.step()
 packed_preds=packed(Xte).squeeze(-1).detach();packed_state={k:v.detach().cpu() for k,v in packed.state_dict().items()}
 nets=[]
 for m in range(M):
  n=Net();st={k:v[m].detach().clone() for k,v in packed.state_dict().items()};n.load_state_dict(st);nets.append(n)
 independent_state={f'm{m}_{k}':v.detach().cpu() for m,n in enumerate(nets) for k,v in n.state_dict().items()}
 independent_preds=torch.stack([nets[m](Xte[m]).squeeze(-1).detach() for m in range(M)])
 # shared-control fits, initialized from the common teacher physical weights.
 shared_rows={}
 for kind in ('film','givens'):
  model=Shared(kind,W1,b1,W2,b2);opt=torch.optim.Adam(model.parameters(),lr=.01)
  for step in range(400):
   ix=torch.randint(0,512,(M,128));xb=torch.stack([Xtr[m,ix[m]] for m in range(M)]);yb=torch.stack([Ytr[m,ix[m]] for m in range(M)])
   pred=model.forward_all(xb);loss=((pred-yb)**2).mean();opt.zero_grad();loss.backward();opt.step()
  pred=model.forward_all(Xte).detach()
  shared_rows[kind]=(model,pred,{k:v.detach().cpu() for k,v in model.state_dict().items()})
 methods=[('independent',independent_preds,independent_state),('packed_independent',packed_preds,packed_state)]
 methods += [(kind,pred,state) for kind,(model,pred,state) in shared_rows.items()]
 rows=[]
 for name,pred,state in methods:
  size,digest=serialize(state);per=((pred-Yte)**2).mean(dim=1).numpy();ensemble=((pred.mean(0)-Yte.mean(0))**2).mean().item()
  corrs=[]
  for i in range(M):
   for j in range(i):corrs.append(float(np.corrcoef(pred[i].numpy(),pred[j].numpy())[0,1]))
  mac=int(M*(2*D*H+2*H)+ (M*H if name=='film' else (M*H*3//2 if name=='givens' else 0)))
  # average CPU inference time for all four member functions
  t0=time.perf_counter()
  with torch.no_grad():
   for _ in range(15):
    if name=='independent':
     _=[nets[m](Xte[m]) for m in range(M)]
    elif name=='packed_independent':
     _=packed(Xte)
    else:
     model=shared_rows[name][0];_=model.forward_all(Xte)
  elapsed=(time.perf_counter()-t0)*1000/15
  for m in range(M):rows.append({'seed':seed,'method':name,'member':m,'serialized_bytes':size,'payload_sha256':digest,'member_mse':float(per[m]),'ensemble_mse':float(ensemble),'mean_pair_correlation':float(np.mean(corrs)),'active_macs':mac,'inference_wall_ms':elapsed,'train_updates':400,'status_note':'teacher-aligned synthetic regression; packed uses copied independent fit'})
 return rows
def save(rows,path):
 import csv
 with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=72501);p.add_argument('--out',default='RESULTS_CORE.csv');a=p.parse_args();rows=run(a.seed);save(rows,a.out)
