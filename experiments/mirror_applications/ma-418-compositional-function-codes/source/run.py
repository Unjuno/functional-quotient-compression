#!/usr/bin/env python3
import argparse,hashlib,io,json,math,random,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';METHODS=['shared','mirror_factor','latent_factor','film_factor','pair_table','oracle_independent'];DEV=[41800,41801];FRESH=[41810,41811,41812];SEEDS=[0,1,2];LRS=[.003,.01];NO,NS,H=4,4,32;UPDATES,BATCH=600,256
HELD=[i*4+i for i in range(4)]
def fixseed(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def rotate_hidden(h,a,b):
 z=h.clone()
 for k,ang in enumerate((a,b)):
  i,j=2*k,2*k+1;u,v=z[:,i].clone(),z[:,j].clone();co=ang.cos();si=ang.sin();z[:,i]=co*u-si*v;z[:,j]=si*u+co*v
 return z
def data(w):
 g=torch.Generator().manual_seed(w);x=torch.rand(16,256,1,generator=g)*2*math.pi-math.pi;w1=torch.randn(1,H,generator=g)*.5;b1=torch.randn(H,generator=g)*.1;w2=torch.randn(H,1,generator=g)*.5;b2=torch.randn(1,generator=g)*.1;oa=torch.randn(NO,generator=g)*.5;sa=torch.randn(NS,generator=g)*.5;y=[]
 for p in range(16):
  o,s=p//4,p%4;h=F.relu(x[p]@w1+b1);h=rotate_hidden(h,torch.full((len(h),),oa[o]),torch.full((len(h),),sa[s]));y.append((h@w2+b2).squeeze(-1))
 return x,torch.stack(y),HELD
class Model(nn.Module):
 def __init__(self,m):
  super().__init__();self.m=m
  if m=='oracle_independent':self.w1=nn.Parameter(torch.randn(16,1,H)*.2);self.b1=nn.Parameter(torch.zeros(16,H));self.w2=nn.Parameter(torch.randn(16,H,1)*.2);self.b2=nn.Parameter(torch.zeros(16,1))
  else:
   self.w1=nn.Parameter(torch.randn(1,H)*.2);self.b1=nn.Parameter(torch.zeros(H));self.w2=nn.Parameter(torch.randn(H,1)*.2);self.b2=nn.Parameter(torch.zeros(1))
  if m=='mirror_factor':self.oa=nn.Parameter(torch.zeros(NO));self.sa=nn.Parameter(torch.zeros(NS))
  if m=='latent_factor':self.oc=nn.Parameter(torch.randn(NO,2)*.03);self.sc=nn.Parameter(torch.randn(NS,2)*.03);self.w1=nn.Parameter(torch.randn(5,H)*.2)
  if m=='film_factor':self.os=nn.Parameter(torch.zeros(NO,H));self.osh=nn.Parameter(torch.zeros(NO,H));self.ss=nn.Parameter(torch.zeros(NS,H));self.ssh=nn.Parameter(torch.zeros(NS,H))
  if m=='pair_table':self.codes=nn.Parameter(torch.zeros(16,2))
 def forward(self,x,pair):
  o=pair//4;s=pair%4
  if self.m=='oracle_independent':
   h=F.relu(torch.bmm(x[:,None,:],self.w1[pair]).squeeze(1)+self.b1[pair]);return (torch.bmm(h[:,None,:],self.w2[pair]).squeeze(1)+self.b2[pair]).squeeze(-1)
  h=F.relu(torch.cat([x,self.oc[o],self.sc[s]],-1)@self.w1+self.b1) if self.m=='latent_factor' else F.relu(x@self.w1+self.b1)
  if self.m=='mirror_factor':h=rotate_hidden(h,self.oa[o],self.sa[s])
  elif self.m=='pair_table':h=rotate_hidden(h,self.codes[pair,0],self.codes[pair,1])
  elif self.m=='film_factor':h=h*(1+self.os[o]+self.ss[s])+self.osh[o]+self.ssh[s]
  return (h@self.w2+self.b2).squeeze(-1)
def serialize(m):
 b=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},b,_use_new_zipfile_serialization=True);return b.getvalue()
def metrics(model,x,y):
 seen=[];held=[]
 with torch.no_grad():
  for p in range(16):
   xx=x[p];pp=model(xx,torch.full((len(xx),),p));r=((pp-y[p]).square().mean().sqrt()/(y[p].square().mean().sqrt()+1e-9)).item();(held if p in HELD else seen).append(r)
 return sum(seen)/len(seen),sum(held)/len(held)
def run(m,w,s,lr,split):
 fixseed(w*100+s);x,y,held=data(w);model=Model(m);opt=torch.optim.AdamW(model.parameters(),lr=lr);st=time.perf_counter();pairs=[p for p in range(16) if m=='oracle_independent' or p not in HELD];ids=torch.tensor(pairs).repeat_interleave(128);xs=torch.cat([x[p,:128] for p in pairs]);ys=torch.cat([y[p,:128] for p in pairs])
 for _ in range(UPDATES):
  ix=torch.randint(len(xs),(BATCH,));loss=F.mse_loss(model(xs[ix],ids[ix]),ys[ix]);opt.zero_grad();loss.backward();opt.step()
 wall=time.perf_counter()-st;seen,held_rmse=metrics(model,x[:,128:],y[:,128:]);raw=serialize(model);name=f'{split}_{w}_{s}_{m}_lr{lr:g}.pt';(OUT/'payloads'/name).write_bytes(raw);cl=Model(m);cl.load_state_dict(torch.load(io.BytesIO(raw),weights_only=True));rb=serialize(cl);assert raw==rb
 mac=H+H+(8 if m in ('mirror_factor','pair_table') else 4*H if m=='latent_factor' else 2*H if m=='film_factor' else 0)
 return {'split':split,'world':w,'seed':s,'method':m,'lr':lr,'payload_file':name,'serialized_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'roundtrip_sha256':hashlib.sha256(rb).hexdigest(),'examples':UPDATES*BATCH,'updates':UPDATES,'active_mac_proxy_per_query':mac,'wall_time_s':round(wall,6),'seen_pair_normalized_rmse':seen,'heldout_pair_normalized_rmse':held_rmse}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);(OUT/'payloads').mkdir(exist_ok=True);rows=[];sel={}
 if a.phase=='dev':
  for m in METHODS:
   sc={lr:[] for lr in LRS}
   for lr in LRS:
    for w in DEV:
     for s in SEEDS:r=run(m,w,s,lr,'development');rows.append(r);sc[lr].append(r['seen_pair_normalized_rmse'])
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

