#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';SEEDS=(36101,36102,36111,36112,36113);T=16;D=16;C=4;ROLES=3;CAP=.5;UPDATES=1000;BATCH=64

class Block(nn.Module):
 def __init__(self):
  super().__init__();self.a=nn.Linear(D,D);self.b=nn.Linear(D,D)
 def forward(self,x):return x+0.5*self.b(torch.relu(self.a(x)))
class Net(nn.Module):
 def __init__(self,kind,seed):
  super().__init__();torch.manual_seed(seed);self.kind=kind;self.embed=nn.Linear(D,D);self.head=nn.Linear(D,C)
  if kind=='native_moD':self.blocks=nn.ModuleList([Block() for _ in range(ROLES)])
  else:self.shared=Block()
  self.role=nn.Parameter(torch.zeros(ROLES)) if kind in ('direct_gate','mirror_view') else None
 def forward(self,x):
  h=torch.relu(self.embed(x));score=x.square().sum(-1);k=max(1,int(T*CAP));idx=score.topk(k,dim=1).indices
  mask=torch.zeros_like(score,dtype=torch.bool).scatter_(1,idx,True)
  for r in range(ROLES):
   sel=h[mask];
   if sel.numel()==0:continue
   if self.kind=='native_moD':out=self.blocks[r](sel)
   else:
    out=self.shared(sel)
    if self.role is not None:out=sel+torch.exp(self.role[r])*(out-sel)
   h=h.clone();h[mask]=out
  return self.head(h),mask

def data(seed):
 g=torch.Generator().manual_seed(seed);x=torch.randn(1536,T,D,generator=g);w=torch.randn(D,C,generator=g);log=x@w;log=log+0.3*torch.sin(x[:,:,:,None].mean(2));y=log.argmax(-1);return x,y
def archive(m):
 b=io.BytesIO();np.savez_compressed(b,**{k:v.detach().cpu().numpy() for k,v in m.state_dict().items()});return b.getvalue()
def run(seed,kind):
 x,y=data(seed);m=Net(kind,seed);opt=torch.optim.AdamW(m.parameters(),lr=1e-3);start=time.perf_counter()
 for u in range(UPDATES):
  ix=torch.randint(len(x),(BATCH,));opt.zero_grad();log,_=m(x[ix]);loss=nn.functional.cross_entropy(log.reshape(-1,C),y[ix].reshape(-1));loss.backward();opt.step()
 elapsed=time.perf_counter()-start
 with torch.no_grad():log,mask=m(x[1024:]);loss=float(nn.functional.cross_entropy(log.reshape(-1,C),y[1024:].reshape(-1)));acc=float((log.argmax(-1)==y[1024:]).float().mean())
 p=archive(m);mac=int(UPDATES*BATCH*T*CAP*ROLES*D*D*2)
 return {'seed':seed,'method':kind,'payload_bytes':len(p),'sha256':hashlib.sha256(p).hexdigest(),'test_NLL':loss,'token_accuracy':acc,'routed_tokens_per_sequence':int(mask[0].sum()),'router_overflow':0,'examples':UPDATES*BATCH*T,'optimizer_updates':UPDATES,'active_MAC_proxy':mac,'wall_seconds':elapsed}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');a=ap.parse_args();rows=[]
 for seed in SEEDS[:2] if a.dev_only else SEEDS:
  for kind in ('native_moD','tied_moD','direct_gate','mirror_view'):
   r=run(seed,kind);rows.append(r);print(json.dumps(r,sort_keys=True))
 OUT.mkdir(exist_ok=True);p=OUT/('development.csv' if a.dev_only else 'results.csv')
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
