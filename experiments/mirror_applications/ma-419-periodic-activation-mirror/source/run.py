#!/usr/bin/env python3
import argparse,hashlib,io,json,math,random,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';METHODS=['shared','mirror3','latent3','modnet','independent'];DEV=[41900,41901];FRESH=[41910,41911,41912];SEEDS=[0,1,2];LRS=[.003,.01];NF,H=16,32;UPDATES,BATCH=500,256
def fixseed(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def data(w):
 g=torch.Generator().manual_seed(w);amp=.5+torch.rand(NF,generator=g);freq=.5+2.5*torch.rand(NF,generator=g);phase=(torch.rand(NF,generator=g)*2-1)*math.pi;tx=torch.rand(NF,64,1,generator=g)*2*math.pi-math.pi;vx=torch.rand(NF,128,1,generator=g)*2*math.pi-math.pi
 return tx,amp[:,None]*torch.sin(freq[:,None]*tx[...,0]+phase[:,None]),vx,amp[:,None]*torch.sin(freq[:,None]*vx[...,0]+phase[:,None])
class Model(nn.Module):
 def __init__(self,m):
  super().__init__();self.m=m;din=4 if m=='latent3' else 1
  if m=='independent':self.w1=nn.Parameter(torch.randn(NF,1,H)*.4);self.b1=nn.Parameter(torch.zeros(NF,H));self.w2=nn.Parameter(torch.randn(NF,H,1)*.1);self.b2=nn.Parameter(torch.zeros(NF,1))
  else:self.w1=nn.Parameter(torch.randn(din,H)*.4);self.b1=nn.Parameter(torch.zeros(H));self.w2=nn.Parameter(torch.randn(H,1)*.1);self.b2=nn.Parameter(torch.zeros(1))
  if m=='mirror3':self.amp=nn.Parameter(torch.ones(NF));self.phase=nn.Parameter(torch.zeros(NF));self.freq=nn.Parameter(torch.ones(NF))
  if m=='latent3':self.code=nn.Parameter(torch.randn(NF,3)*.03)
  if m=='modnet':
   self.code=nn.Parameter(torch.randn(NF,2)*.03);self.gen=nn.Linear(2,3*H)
   with torch.no_grad():self.gen.weight.zero_();self.gen.bias[:H].fill_(1.);self.gen.bias[H:2*H].fill_(0.);self.gen.bias[2*H:].fill_(.5413)
 def forward(self,x,f):
  if self.m=='independent':h=torch.sin(torch.bmm(x[:,None,:],self.w1[f]).squeeze(1)+self.b1[f]);return (torch.bmm(h[:,None,:],self.w2[f]).squeeze(1)+self.b2[f]).squeeze(-1)
  if self.m=='latent3':x=torch.cat([x,self.code[f]],-1)
  z=x@self.w1+self.b1
  if self.m=='mirror3':h=self.amp[f,None]*torch.sin(self.freq[f,None]*z+self.phase[f,None])
  elif self.m=='modnet':
   q=self.gen(self.code[f]);a,p,r=q[:,:H],q[:,H:2*H],q[:,2*H:];h=torch.sigmoid(a)*2*torch.sin(F.softplus(r)*z+p)
  else:h=torch.sin(z)
  return (h@self.w2+self.b2).squeeze(-1)
def serialize(m):
 b=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},b,_use_new_zipfile_serialization=True);return b.getvalue()
def run(m,w,s,lr,split):
 fixseed(w*100+s);tx,ty,vx,vy=data(w);model=Model(m);opt=torch.optim.AdamW(model.parameters(),lr=lr);st=time.perf_counter();xx=tx.reshape(-1,1);yy=ty.reshape(-1);fid=torch.arange(NF).repeat_interleave(64)
 for _ in range(UPDATES):
  ix=torch.randint(len(xx),(BATCH,));loss=F.mse_loss(model(xx[ix],fid[ix]),yy[ix]);opt.zero_grad();loss.backward();opt.step()
 wall=time.perf_counter()-st;per=[]
 with torch.no_grad():
  for f in range(NF):
   pred=model(vx[f],torch.full((128,),f));per.append(((pred-vy[f]).square().mean().sqrt()/(vy[f].square().mean().sqrt()+1e-9)).item())
 raw=serialize(model);name=f'{split}_{w}_{s}_{m}_lr{lr:g}.pt';(OUT/'payloads'/name).write_bytes(raw);cl=Model(m);cl.load_state_dict(torch.load(io.BytesIO(raw),weights_only=True));rb=serialize(cl);assert raw==rb
 mac=(1+(3 if m=='latent3' else 0))*H+H+(3*H+2 if m=='modnet' else 4 if m=='mirror3' else 0)
 return {'split':split,'world':w,'seed':s,'method':m,'lr':lr,'payload_file':name,'serialized_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'roundtrip_sha256':hashlib.sha256(rb).hexdigest(),'examples':UPDATES*BATCH,'updates':UPDATES,'active_mac_proxy_per_query':mac,'wall_time_s':round(wall,6),'mean_normalized_rmse':sum(per)/NF,'per_signal_normalized_rmse':per}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);(OUT/'payloads').mkdir(exist_ok=True);rows=[];sel={}
 if a.phase=='dev':
  for m in METHODS:
   sc={lr:[] for lr in LRS}
   for lr in LRS:
    for w in DEV:
     for s in SEEDS:r=run(m,w,s,lr,'development');rows.append(r);sc[lr].append(r['mean_normalized_rmse'])
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

