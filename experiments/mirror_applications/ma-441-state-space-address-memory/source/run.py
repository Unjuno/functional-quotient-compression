#!/usr/bin/env python3
import argparse,hashlib,json,random,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2];OUT=ROOT/'artifacts';PAY=OUT/'payloads'
DEV=[44100,44101];FRESH=[44110,44111,44112];SEEDS=[0,1,2];LRS=[.003,.01];METHODS=['external','mirror','register','fastweight'];DELAYS=[8,32,128,512];D=4;UPDATES=500;BATCH=64

def fix(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def world(w):
 g=torch.Generator().manual_seed(w);q,_=torch.linalg.qr(torch.randn(D,D,generator=g));return q@torch.diag(torch.tensor([.5,.62,.72,.82]))@q.T
def batch(w,n,L,seed):
 g=torch.Generator().manual_seed(seed);role=torch.randint(D,(n,),generator=g);noise=torch.randn(n,L,D,generator=g);return role,noise
def dimension(method):return {'external':1,'mirror':2,'register':4,'fastweight':16}[method]
class Memory(nn.Module):
 def __init__(self,method):
  super().__init__();self.method=method
  if method=='external':
   self.register_buffer('table',torch.eye(D));return
  d=dimension(method);self.code=nn.Parameter(torch.randn(D,d)*.15);self.noise=nn.Parameter(torch.zeros(d,D));self.decay=nn.Parameter(torch.tensor(-4.0));self.read=nn.Linear(d,D)
 def state(self,role,noise,L):
  a=torch.sigmoid(self.decay);d=self.code.shape[1];b=self.code[role]
  # Closed-form recurrent register: h_t=a*h_{t-1}+W*x_t, with L distractor tokens.
  p=a**torch.arange(L-1,-1,-1,dtype=noise.dtype)
  if self.method=='fastweight':
   one=torch.nn.functional.one_hot(role,D).float();b=torch.einsum('bi,bj->bij',one,one).reshape(role.shape[0],D*D)
  return (a**L)*b+torch.einsum('t,btd,ad->ba',p,noise,self.noise)
 def forward(self,role,noise,L):
  if self.method=='external':return self.table[role]*20
  return self.read(self.state(role,noise,L))
def data(w,n,L,seed):return batch(w,n,L,seed)
def fit(w,s,method,lr,phase):
 fix(w*100+s*7+sum(map(ord,method)));m=Memory(method);opt=torch.optim.AdamW(m.parameters(),lr=lr) if list(m.parameters()) else None;st=time.perf_counter()
 if opt is not None:
  for _ in range(UPDATES):
   L=random.choice(DELAYS);r,x=data(w,BATCH,L,w*10000+_);y=torch.nn.functional.one_hot(r,D).float();logits=m(r,x,L);loss=nn.functional.cross_entropy(logits,y.argmax(1));opt.zero_grad();loss.backward();opt.step()
 wall=time.perf_counter()-st;p=PAY/f'{phase}_{w}_{s}_{method}.pt';torch.save({'method':method,'world':w,'state':m.state_dict()},p);b=p.read_bytes();return m,{'bytes':len(b),'hash':hashlib.sha256(b).hexdigest(),'path':str(p.relative_to(REPO)),'train_wall':wall}
def evaluate(m,w,delay,seed):
 r,x=data(w,512,delay,seed);y=torch.nn.functional.one_hot(r,D).float();st=time.perf_counter()
 with torch.no_grad():log=m(r,x,delay);pred=log.argmax(1);prob=log.softmax(-1)
 wall=time.perf_counter()-st;acc=(pred==r).float().mean().item();nll=nn.functional.cross_entropy(log,y.argmax(1)).item();drift=(1-acc);return nll,acc,wall,drift

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);PAY.mkdir(parents=True,exist_ok=True);worlds=DEV if a.phase=='development' else FRESH;rows=[];sel=json.loads((OUT/'development_selection.json').read_text())['selected_learning_rate_by_method'] if a.phase=='fresh' else {}
 for w in worlds:
  for method in METHODS:
   if a.phase=='development':
    scores={lr:[] for lr in LRS}
    for lr in LRS:
     for s in SEEDS:
      m,meta=fit(w,s,method,lr,a.phase)
      for delay in DELAYS:
       nll,acc,wall,drift=evaluate(m,w,delay,w*100+s+delay);scores[lr].append(nll)
       rows.append({'condition':'development','world_or_seed':f'{w}-{s}-{delay}','method':method,'serialized_bytes':meta['bytes'],'train_tokens_or_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':f'{dimension(method)*D} MAC/token','wall_time_s':round(wall,6),'primary_metric':'recall_NLL','primary_value':nll,'secondary_metric':'accuracy','secondary_value':acc,'status_note':f'lr={lr}; drift={drift}; state_bytes={dimension(method)*4}; hash={meta["hash"]}; payload={meta["path"]}'})
    sel[method]=min(LRS,key=lambda lr:sum(scores[lr])/len(scores[lr]))
   else:
    for s in SEEDS:
     m,meta=fit(w,s,method,sel[method],a.phase)
     for delay in DELAYS:
      nll,acc,wall,drift=evaluate(m,w,delay,w*100+s+delay);rows.append({'condition':'fresh','world_or_seed':f'{w}-{s}-{delay}','method':method,'serialized_bytes':meta['bytes'],'train_tokens_or_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':f'{dimension(method)*D} MAC/token','wall_time_s':round(wall,6),'primary_metric':'recall_NLL','primary_value':nll,'secondary_metric':'accuracy','secondary_value':acc,'status_note':f'lr={sel[method]}; drift={drift}; state_bytes={dimension(method)*4}; train_wall={meta["train_wall"]:.6f}; hash={meta["hash"]}; payload={meta["path"]}'})
 if a.phase=='development':(OUT/'development_selection.json').write_text(json.dumps({'selected_learning_rate_by_method':sel,'rule':'lowest mean recall NLL across delays, worlds, seeds','fresh_worlds_not_accessed':True},indent=2)+'\n')
 (OUT/f'{a.phase}_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'phase':a.phase,'selection':sel,'rows':len(rows)},indent=2))
if __name__=='__main__':main()
