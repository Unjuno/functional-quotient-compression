#!/usr/bin/env python3
import argparse,hashlib,io,json,math,random,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';METHODS=['shared','mirror_role','delta_gate','independent'];DEV=[43400,43401];FRESH=[43410,43411,43412];SEEDS=[0,1,2];LRS=[.003,.01];R,T,D=4,16,2;UPDATES,BATCH=300,32
def fixseed(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def rot(a):
 c=a.cos();s=a.sin();return torch.stack([torch.stack([c,-s]),torch.stack([s,c])])
def data(w,n=256):
 g=torch.Generator().manual_seed(w);decay=torch.tensor([-.18,-.58]);angles=torch.rand(R,generator=g)*.8-.4;dw=torch.randn(D,generator=g)*.18;B=torch.randn(D,D,generator=g)*.4;C=torch.randn(D,D,generator=g)*.6;x=torch.randn(R,n,T,D,generator=g);ys=[]
 for r in range(R):
  q=rot(angles[r]);h=torch.zeros(n,D);out=[]
  for t in range(T):
   dt=F.softplus(x[r,:,t]@dw+.15);alpha=torch.exp(dt[:,None]*decay);ad=q[None]@torch.diag_embed(alpha)@q.T[None];h=torch.bmm(ad,h[:,:,None]).squeeze(-1)+dt[:,None]*(x[r,:,t]@B.T);out.append(h@C.T)
  ys.append(torch.stack(out,1))
 return x,torch.stack(ys),decay,dw,B,C,angles
def sample_eval(w,n,decay,dw,B,C,angles):
 g=torch.Generator().manual_seed(w+9000);x=torch.randn(R,n,T,D,generator=g);ys=[]
 for r in range(R):
  q=rot(angles[r]);h=torch.zeros(n,D);out=[]
  for t in range(T):
   dt=F.softplus(x[r,:,t]@dw+.15);alpha=torch.exp(dt[:,None]*decay);ad=q[None]@torch.diag_embed(alpha)@q.T[None];h=torch.bmm(ad,h[:,:,None]).squeeze(-1)+dt[:,None]*(x[r,:,t]@B.T);out.append(h@C.T)
  ys.append(torch.stack(out,1))
 return x,torch.stack(ys)
class Model(nn.Module):
 def __init__(self,m):
  super().__init__();self.m=m;self.raw_decay=nn.Parameter(torch.tensor([-.9,-.1]));self.dw=nn.Parameter(torch.zeros(D));self.db=nn.Parameter(torch.tensor(.15));self.B=nn.Parameter(torch.randn(D,D)*.1);self.C=nn.Parameter(torch.randn(D,D)*.1)
  if m=='mirror_role':self.angles=nn.Parameter(torch.zeros(R))
  if m=='delta_gate':self.gate=nn.Parameter(torch.ones(R))
  if m=='independent':self.raw_decay=nn.Parameter(torch.randn(R,D)*-.1);self.B=nn.Parameter(torch.randn(R,D,D)*.1);self.C=nn.Parameter(torch.randn(R,D,D)*.1)
 def forward(self,x,role):
  h=torch.zeros(x.size(0),D);outs=[];decay=-F.softplus(self.raw_decay)
  for t in range(T):
   dt=F.softplus(x[:,t]@self.dw+self.db)
   if self.m=='delta_gate':dt=dt*self.gate[role]
   if self.m=='independent':
    de=decay[role];bb=self.B[role];cc=self.C[role];ad=torch.diag_embed(torch.exp(dt[:,None]*de))
   else:
    de=decay
    if self.m=='mirror_role':
     q=torch.stack([rot(self.angles[int(r)]) for r in role]);ad=q@torch.diag_embed(torch.exp(dt[:,None]*de[None]))@q.transpose(1,2)
    else:ad=torch.diag_embed(torch.exp(dt[:,None]*de))
    bb=self.B;cc=self.C
   h=torch.bmm(ad,h[:,:,None]).squeeze(-1)+dt[:,None]*(torch.bmm(bb,x[:,t,:,None]).squeeze(-1) if bb.ndim==3 else x[:,t]@bb.T)
   outs.append(torch.bmm(cc,h[:,:,None]).squeeze(-1) if cc.ndim==3 else h@cc.T)
  return torch.stack(outs,1)
def serialize(m):
 b=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},b,_use_new_zipfile_serialization=True);return b.getvalue()
def run(m,w,s,lr,split):
 fixseed(w*100+s);x,y,decay,dw,B,C,angles=data(w);model=Model(m);opt=torch.optim.AdamW(model.parameters(),lr=lr);st=time.perf_counter();xx=x.reshape(R*256,T,D);yy=y.reshape(R*256,T,D);roles=torch.arange(R).repeat_interleave(256)
 for _ in range(UPDATES):
  ix=torch.randint(len(xx),(BATCH,));loss=((model(xx[ix],roles[ix])-yy[ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
 wall=time.perf_counter()-st;vx,vy=sample_eval(w,128,decay,dw,B,C,angles);metric=[]
 with torch.no_grad():
  for r in range(R):
   pred=model(vx[r],torch.full((128,),r));metric.append(((pred-vy[r]).square().mean().sqrt()/(vy[r].square().mean().sqrt()+1e-9)).item())
 raw=serialize(model);name=f'{split}_{w}_{s}_{m}_lr{lr:g}.pt';(OUT/'payloads'/name).write_bytes(raw);cl=Model(m);cl.load_state_dict(torch.load(io.BytesIO(raw),weights_only=True));rb=serialize(cl);assert raw==rb
 return {'split':split,'world':w,'seed':s,'method':m,'lr':lr,'payload_file':name,'serialized_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'roundtrip_sha256':hashlib.sha256(rb).hexdigest(),'examples':UPDATES*BATCH,'updates':UPDATES,'active_mac_proxy_per_token':20,'sequence_steps':T,'wall_time_s':round(wall,6),'mean_sequence_nrmse':sum(metric)/R,'per_role_nrmse':metric}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);(OUT/'payloads').mkdir(exist_ok=True);rows=[];sel={}
 if a.phase=='dev':
  for m in METHODS:
   sc={lr:[] for lr in LRS}
   for lr in LRS:
    for w in DEV:
     for s in SEEDS:r=run(m,w,s,lr,'development');rows.append(r);sc[lr].append(r['mean_sequence_nrmse'])
   sel[m]=min(LRS,key=lambda lr:sum(sc[lr])/len(sc[lr]))
  (OUT/'development_selection.json').write_text(json.dumps(sel,indent=2)+'\n');mode='w'
 else:
  sel=json.loads((OUT/'development_selection.json').read_text())
  for m in METHODS:
   for w in FRESH:
    for s in SEEDS:rows.append(run(m,w,s,sel[m],'fresh'))
  mode='a'
 with (OUT/'runs.jsonl').open(mode) as f:
  for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
 print(json.dumps({'phase':a.phase,'rows':len(rows),'selection':sel},indent=2))
if __name__=='__main__':main()

