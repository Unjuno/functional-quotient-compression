#!/usr/bin/env python3
import argparse,hashlib,io,json,math,random,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';METHODS=['shared','mirror2','latent1','latent2','latent8','independent'];DEV=[41600,41601];FRESH=[41610,41611,41612];SEEDS=[0,1,2];LRS=[.003,.01];NF,D,H=16,1,32;UPDATES,BATCH=500,256
def fixseed(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def data(world):
 g=torch.Generator().manual_seed(world);coeff=torch.randn(NF,6,generator=g)*.45;trainx=torch.rand(NF,64,1,generator=g)*2*math.pi-math.pi;testx=torch.rand(NF,128,1,generator=g)*2*math.pi-math.pi
 def targets(x):
  y=torch.zeros(x.shape[:-1])
  for k in range(3):y+=coeff[:,None,2*k]*torch.sin((k+1)*x[...,0])+coeff[:,None,2*k+1]*torch.cos((k+1)*x[...,0])
  return y
 return trainx,targets(trainx),testx,targets(testx)
class Model(nn.Module):
 def __init__(self,m):
  super().__init__();self.m=m;self.d={'latent1':1,'latent2':2,'latent8':8}.get(m,0)
  if m=='independent':self.w1=nn.Parameter(torch.randn(NF,1,H)*.2);self.b1=nn.Parameter(torch.zeros(NF,H));self.w2=nn.Parameter(torch.randn(NF,H,1)*.2);self.b2=nn.Parameter(torch.zeros(NF,1))
  else:
   self.w1=nn.Parameter(torch.randn(1+self.d,H)*.2);self.b1=nn.Parameter(torch.zeros(H));self.w2=nn.Parameter(torch.randn(H,1)*.2);self.b2=nn.Parameter(torch.zeros(1))
  if m=='mirror2':self.angles=nn.Parameter(torch.zeros(NF,2))
  if self.d:self.codes=nn.Parameter(torch.randn(NF,self.d)*.03)
 def forward(self,x,f):
  if self.m=='independent':
   h=F.relu(torch.bmm(x[:,None,:],self.w1[f]).squeeze(1)+self.b1[f]);return (torch.bmm(h[:,None,:],self.w2[f]).squeeze(1)+self.b2[f]).squeeze(-1)
  if self.d:x=torch.cat([x,self.codes[f]],-1)
  h=F.relu(x@self.w1+self.b1)
  if self.m=='mirror2':
   z=h.clone();ang=self.angles[f]
   for k in range(2):
    i,j=2*k,2*k+1;a,b=z[:,i].clone(),z[:,j].clone();co=ang[:,k].cos();si=ang[:,k].sin();z[:,i]=co*a-si*b;z[:,j]=si*a+co*b
   h=z
  return (h@self.w2+self.b2).squeeze(-1)
def serialize(m):
 b=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},b,_use_new_zipfile_serialization=True);return b.getvalue()
def evaluate(m,x,y):
 out={}
 with torch.no_grad():
  for f in range(NF):
   ff=torch.full((x.size(1),),f);p=m(x[f],ff);rmse=(p-y[f]).square().mean().sqrt();scale=y[f].square().mean().sqrt()+1e-9;out[str(f)]=(rmse/scale).item()
 return out
def run(method,w,s,lr,split):
 fixseed(w*100+s);tx,ty,vx,vy=data(w);model=Model(method);opt=torch.optim.AdamW(model.parameters(),lr=lr);st=time.perf_counter();flatx=tx.reshape(-1,1);flaty=ty.reshape(-1);fid=torch.arange(NF).repeat_interleave(tx.size(1))
 for _ in range(UPDATES):
  ix=torch.randint(len(flatx),(BATCH,));pred=model(flatx[ix],fid[ix]);loss=F.mse_loss(pred,flaty[ix]);opt.zero_grad();loss.backward();opt.step()
 wall=time.perf_counter()-st;metric=evaluate(model,vx,vy);raw=serialize(model);name=f'{split}_{w}_{s}_{method}_lr{lr:g}.pt';(OUT/'payloads'/name).write_bytes(raw);cl=Model(method);cl.load_state_dict(torch.load(io.BytesIO(raw),weights_only=True));rb=serialize(cl);assert raw==rb
 mac=(1+model.d)*H+H+(8 if method=='mirror2' else 0)
 return {'split':split,'world':w,'seed':s,'method':method,'lr':lr,'payload_file':name,'serialized_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'roundtrip_sha256':hashlib.sha256(rb).hexdigest(),'examples':UPDATES*BATCH,'updates':UPDATES,'active_mac_proxy_per_query':mac,'wall_time_s':round(wall,6),'mean_normalized_rmse':sum(metric.values())/NF,'per_function_normalized_rmse':metric}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);(OUT/'payloads').mkdir(exist_ok=True);rows=[];sel={}
 if a.phase=='dev':
  for m in METHODS:
   scores={lr:[] for lr in LRS}
   for lr in LRS:
    for w in DEV:
     for s in SEEDS:r=run(m,w,s,lr,'development');rows.append(r);scores[lr].append(r['mean_normalized_rmse'])
   sel[m]=min(LRS,key=lambda lr:sum(scores[lr])/len(scores[lr]))
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

