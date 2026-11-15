"""Small paired screen for multiscale seasonal mixer views."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]

def seed_all(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)

class Mixer(nn.Module):
 def __init__(self, seq=96,horizon=24,window=2,levels=3,width=48,kind='shared',domains=2):
  super().__init__(); self.seq=seq; self.h=horizon; self.window=window; self.levels=levels; self.kind=kind; self.domains=domains
  lens=[seq//window**i for i in range(levels)]
  self.season=nn.ModuleList([nn.Sequential(nn.Linear(lens[i],lens[i+1]),nn.GELU(),nn.Linear(lens[i+1],lens[i+1])) for i in range(levels-1)])
  self.trend=nn.ModuleList([nn.Sequential(nn.Linear(lens[i+1],lens[i]),nn.GELU(),nn.Linear(lens[i],lens[i])) for i in reversed(range(levels-1))])
  self.norm=nn.ModuleList([nn.LayerNorm(1) for _ in lens])
  self.embed=nn.Linear(1,width); self.pdm=nn.ModuleList([nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width)) for _ in range(2)])
  self.pred=nn.ModuleList([nn.Linear(width,horizon) for _ in lens])
  self.view=nn.Parameter(torch.zeros(domains,levels,2)) if kind in ('mirror','additive') else None
  self.register_buffer('scales',torch.tensor([window**i for i in range(levels)],dtype=torch.float32))
 def forward(self,x,dom):
  # x: B,96,1 standardized per window
  B=x.size(0); orig=x.squeeze(-1); xs=[orig[:,::self.window**i] for i in range(self.levels)]
  # Trend/seasonal split via per-window moving average; decomposition is frozen.
  seasons=[];trends=[]
  for z in xs:
   t=torch.nn.functional.avg_pool1d(torch.nn.functional.pad(z[:,None,:],(3,3),mode='replicate'),7,stride=1).squeeze(1)
   trends.append(t); seasons.append(z-t)
  s=seasons[0]
  for i in range(self.levels-1):
   low=seasons[i+1]; s=low+self.season[i](s)
  t=trends[-1]
  for j,i in enumerate(reversed(range(self.levels-1))): t=trends[i]+self.trend[j](t)
  # Feature mixer operates on normalized temporal sequence at each scale.
  base=orig
  if self.kind in ('mirror','additive'):
   coord=self.view[dom] # B,L,2
   if self.kind=='mirror':
    # bounded rotation angle and log-gain (2 scalars per domain/scale)
    theta=coord[...,0].tanh()[:,0,None]
    gain=coord[...,1].tanh()[:,0,None]
    base=base*(1+gain)+theta*torch.roll(base,1,dims=1)
   else:
    base=base*(1+coord[:,0,0,None])+coord[:,0,1,None]
  e=self.embed(base[...,None])
  for layer in self.pdm: e=e+layer(e)
  preds=[]
  for i in range(self.levels):
   # pool encoded representation to matching scale
   ei=e[:,::self.window**i,:]
   p=self.pred[i](ei.mean(dim=1))
   if self.kind in ('mirror','additive'):
    c=self.view[dom,i]
    if self.kind=='mirror': p=p*(1+c[:,1,None].tanh())+c[:,0,None].tanh()
    else: p=p*(1+c[:,0,None])+c[:,1,None]
   preds.append(p)
  return torch.stack(preds).mean(0)

def load_data():
 out=[]
 for name in ['ETTh1','ETTm1']:
  df=pd.read_csv(ROOT/'runs/raw'/f'{name}.csv')
  arr=df.iloc[:,1:].to_numpy(np.float32)
  # Target is OT; normalized separately using train-only stats.
  y=arr[:,-1]; n=len(y); tr=int(n*.6); dv=int(n*.8)
  mu=y[:tr].mean(); sd=y[:tr].std()+1e-6; y=(y-mu)/sd
  out.append((name,y,tr,dv))
 return out

def windows(data,split,limit):
 rows=[]
 for dom,(name,y,tr,dv) in enumerate(data):
  lo,hi={'train':(96,tr),'dev':(tr,dv),'audit':(dv,len(y))}[split]
  origins=np.arange(lo,hi-24+1)
  if len(origins)>limit: origins=origins[np.linspace(0,len(origins)-1,limit,dtype=int)]
  for o in origins: rows.append((dom,y[o-96:o],y[o:o+24]))
 return rows

def fit(kind,seed,train,dev,epochs=10,initial_state=None):
 seed_all(seed); model=Mixer(kind=kind)
 if initial_state is not None: model.load_state_dict(initial_state,strict=False)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 ids=np.arange(len(train)); t0=time.perf_counter(); updates=0
 for ep in range(epochs):
  np.random.shuffle(ids)
  for st in range(0,len(ids),64):
   batch=[train[i] for i in ids[st:st+64]]
   d=torch.tensor([r[0] for r in batch],dtype=torch.long)
   x=torch.tensor(np.stack([r[1] for r in batch]),dtype=torch.float32)[...,None]
   y=torch.tensor(np.stack([r[2] for r in batch]),dtype=torch.float32)
   opt.zero_grad();loss=((model(x,d)-y)**2).mean();loss.backward();opt.step();updates+=1
  # fixed epoch budget: no audit; dev logged only for fixed description
 return model,updates,time.perf_counter()-t0

def score(model,rows):
 model.eval(); by={0:[],1:[]}
 with torch.no_grad():
  for dom,x,y in rows:
   p=model(torch.tensor(x,dtype=torch.float32)[None,:,None],torch.tensor([dom]))[0].numpy()
   by[dom].append(float(np.mean((p-y)**2)))
 return {str(k):float(np.mean(v)) for k,v in by.items()},float(np.mean([np.mean(v) for v in by.values()]))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=10);a=ap.parse_args();seed=105001
 data=load_data(); train=windows(data,'train',48); dev=windows(data,'dev',12); audit=windows(data,'audit',12)
 results={}; output=ROOT/'runs'; output.mkdir(exist_ok=True)
 seed_all(seed); init=Mixer(kind='shared').state_dict()
 for kind in ['shared','mirror','additive']:
  model,updates,seconds=fit(kind,seed,train,dev,a.epochs,init)
  # metrics are saved separately; audit opening only after every configuration is frozen by common fixed budget.
  dev_score=score(model,dev); audit_score=score(model,audit)
  payload=output/f'{kind}.pt'; torch.save(model.state_dict(),payload)
  meta=output/f'{kind}.json'; meta.write_text(json.dumps({'kind':kind,'config':{'seq':96,'horizon':24,'window':2,'levels':3,'width':48},'seed':seed},sort_keys=True))
  bytes_total=payload.stat().st_size+meta.stat().st_size
  results[kind]={'updates':updates,'train_seconds':seconds,'dev_mse_by_domain':dev_score[0],'dev_macro_mse':dev_score[1],'audit_mse_by_domain':audit_score[0],'audit_macro_mse':audit_score[1],'payload_bytes':bytes_total,'state_bytes':payload.stat().st_size,'metadata_bytes':meta.stat().st_size}
  print(kind,results[kind])
 (output/'screen_results.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')

if __name__=='__main__': main()
