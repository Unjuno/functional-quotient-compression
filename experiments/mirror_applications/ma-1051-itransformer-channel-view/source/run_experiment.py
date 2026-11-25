from __future__ import annotations
import json, hashlib, time, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
SEED=105101

def seed_all(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.set_num_threads(1)

class Block(nn.Module):
 def __init__(self):
  super().__init__(); self.n1=nn.LayerNorm(32); self.att=nn.MultiheadAttention(32,4,dropout=0,batch_first=True); self.n2=nn.LayerNorm(32); self.ff=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,32))
 def forward(self,x):
  q=self.n1(x); a=self.att(q,q,q,need_weights=False)[0]; x=x+a; return x+self.ff(self.n2(x))

class ITransformer(nn.Module):
 def __init__(self, channels=16, view='shared'):
  super().__init__(); self.channels=channels; self.view=view
  self.temporal=nn.Linear(96,32)
  self.attn=nn.ModuleList([Block(),Block()])
  self.forecast=nn.Linear(32,24)
  if view in ('mirror','film'):
   # Two learned scalars per variate, initialized to identity/no-op.
   self.code=nn.Parameter(torch.zeros(channels,2))
 def forward(self,x,mask=None):
  # x B x context x variates
  z=self.temporal(x.transpose(1,2))
  if self.view=='mirror':
   a=self.code[:,0].tanh(); g=self.code[:,1].tanh()
   u=z[...,0]; v=z[...,1]
   c=torch.cos(a); s=torch.sin(a)
   z=torch.cat([((c*u-s*v)*(1+g)).unsqueeze(-1),((s*u+c*v)*(1+g)).unsqueeze(-1),z[...,2:]],dim=-1)
  elif self.view=='film':
   z=torch.cat([(z[...,0]*(1+self.code[:,0])+self.code[:,1]).unsqueeze(-1),z[...,1:]],dim=-1)
  if mask is not None: z=z*mask[:,:,None]
  for block in self.attn: z=block(z)
  return self.forecast(z).transpose(1,2)

def load():
 f=ROOT/'runs/raw/PEMS07.npz'
 if not f.exists():
  import urllib.request
  u='https://huggingface.co/datasets/jimmygao3218/PEMS07/resolve/e071fab0481c3db4eeb4193e79d821220b7b5bff/PEMS07.npz'
  f.write_bytes(urllib.request.urlopen(u,timeout=120).read())
 a=np.load(f,allow_pickle=True)['data'][:,:,:1].astype(np.float32)[:,:,0]
 y=a[:,:16]; n=len(y); tr=int(.6*n); dv=int(.8*n)
 mu=y[:tr].mean(0); sd=y[:tr].std(0)+1e-6; y=(y-mu)/sd
 return y,tr,dv

def origins(n,tr,dv,split,count):
 if split=='train': lo,hi=96,tr-24
 elif split=='dev': lo,hi=tr,dv-24
 else: lo,hi=dv,n-24
 return np.linspace(lo,hi-1,count,dtype=int)

def batch_xy(y,os):
 xs=np.stack([y[o-96:o] for o in os]); ys=np.stack([y[o:o+24] for o in os])
 return torch.from_numpy(xs),torch.from_numpy(ys)

def score(model,y,os,mask=None):
 model.eval(); vals=[]
 with torch.no_grad():
  for o in os:
   x,t=batch_xy(y,[o]); p=model(x,mask)[0].numpy(); target=t[0].numpy(); vals.append(np.mean((p-target)**2,axis=0))
 return float(np.mean(np.stack(vals)))

def fit(model,y,os,epochs=12):
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01); order=np.arange(len(os)); updates=0; t0=time.perf_counter()
 for _ in range(epochs):
  np.random.shuffle(order)
  for j in range(0,len(order),16):
   sel=[os[i] for i in order[j:j+16]]; x,t=batch_xy(y,sel); opt.zero_grad(); loss=((model(x)-t)**2).mean(); loss.backward(); opt.step(); updates+=1
 return updates,time.perf_counter()-t0

def payload(model,name):
 d=ROOT/'runs'/name; d.mkdir(exist_ok=True)
 weights=d/'weights.pt'; meta=d/'inference.json'
 torch.save(model.state_dict(),weights); meta.write_text(json.dumps({'model':'iTransformer-small','channels':model.channels,'view':model.view,'context':96,'horizon':24,'seed':SEED},sort_keys=True)+'\n')
 return weights.stat().st_size+meta.stat().st_size,weights.stat().st_size,meta.stat().st_size

def main():
 seed_all(SEED); y,tr,dv=load(); otrain=origins(len(y),tr,dv,'train',48); odev=origins(len(y),tr,dv,'dev',12); oaudit=origins(len(y),tr,dv,'audit',12)
 # Ensure every variant starts from the same shared trunk initialization.
 seed_all(SEED); init=ITransformer(16,'shared').state_dict(); out={}
 variants=['shared','mirror','film']
 for typ in variants:
  seed_all(SEED); model=ITransformer(16,typ); model.load_state_dict(init,strict=False)
  upd,sec=fit(model,y,otrain)
  dev=score(model,y,odev); audit=score(model,y,oaudit)
  # Input ablation zeros out a fixed quarter of channels while retaining cross-variate token slots.
  m=torch.ones(1,16);m[:,[0,4,8,12]]=0
  abl=score(model,y,oaudit,m)
  nbytes,wbytes,metabytes=payload(model,typ)
  out[typ]={'updates':upd,'train_seconds':sec,'dev_mse':dev,'audit_mse':audit,'ablation_audit_mse':abl,'payload_bytes':nbytes,'weights_bytes':wbytes,'metadata_bytes':metabytes}
 # Independent upper reference: 16 separate univariate iTransformers, trained on same origins.
 indiv=[]; total_updates=0; total_sec=0.; preds=[]
 for c in range(16):
  seed_all(SEED+c); m=ITransformer(1,'shared'); yc=y[:,c:c+1]
  # collect same temporal origins, but channel dimension is one.
  u,s=fit(m,yc,otrain); total_updates+=u; total_sec+=s; indiv.append(m)
  preds.append(m)
 def ind_score(os):
  vals=[]
  with torch.no_grad():
   for o in os:
    per=[]
    for c,m in enumerate(indiv):
     x,t=batch_xy(y[:,c:c+1],[o]); p=m(x)[0,:,0].numpy(); tar=t[0,:,0].numpy();per.append(np.mean((p-tar)**2))
    vals.append(np.mean(per))
  return float(np.mean(vals))
 droot=ROOT/'runs'/'independent';droot.mkdir(exist_ok=True)
 torch.save([m.state_dict() for m in indiv],droot/'weights.pt')
 (droot/'inference.json').write_text(json.dumps({'model':'16 independent univariate iTransformer-small','channels':16,'context':96,'horizon':24,'seed':SEED},sort_keys=True)+'\n')
 out['independent']={'updates':total_updates,'train_seconds':total_sec,'dev_mse':ind_score(odev),'audit_mse':ind_score(oaudit),'ablation_audit_mse':None,'payload_bytes':(droot/'weights.pt').stat().st_size+(droot/'inference.json').stat().st_size,'weights_bytes':(droot/'weights.pt').stat().st_size,'metadata_bytes':(droot/'inference.json').stat().st_size}
 (ROOT/'runs/results.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
