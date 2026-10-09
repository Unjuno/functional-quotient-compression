#!/usr/bin/env python3
import argparse,hashlib,io,json,math,random,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';METHODS=['shared','mirror_time','linear_time','discrete'];DEV=[42900,42901];FRESH=[42910,42911,42912];SEEDS=[0,1,2];LRS=[.003,.01];D=2;DEPTH=8;TRAIN_DEPTH=4;UPDATES,BATCH=400,128
def fixseed(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def rot(a):
 c=a.cos();s=a.sin();return torch.stack([torch.stack([c,-s]),torch.stack([s,c])])
def world(w,n=1024):
 g=torch.Generator().manual_seed(w);q=rot(torch.rand((),generator=g)*math.pi);base=q@torch.diag(torch.tensor([.70,.88]))@q.T;omega=.12;xs=torch.randn(n,D,generator=g);ys=[];z=xs
 for d in range(1,DEPTH+1):a=rot(torch.tensor(omega*d));z=z@(a@base@a.T).T;ys.append(z)
 return xs,torch.stack(ys,1),base,omega
def teacher_trajectory(x,base,omega):
 z=x;out=[]
 for d in range(1,DEPTH+1):q=rot(torch.tensor(omega*d));z=z@(q@base@q.T).T;out.append(z)
 return torch.stack(out,1)
class Model(nn.Module):
 def __init__(self,m):
  super().__init__();self.m=m
  if m=='discrete':self.a=nn.Parameter(torch.randn(4,D,D)*.1)
  else:self.a=nn.Parameter(torch.randn(D,D)*.1)
  if m=='mirror_time':self.omega=nn.Parameter(torch.tensor(.04))
  if m=='linear_time':self.b=nn.Parameter(torch.zeros(D,D))
 def raw_matrix(self,d):
  if self.m=='mirror_time':q=rot(self.omega*d);return q@self.a@q.T
  if self.m=='linear_time':return self.a+d*self.b
  if self.m=='discrete':return self.a[min(d-1,3)]
  return self.a
 def matrix(self,d):
  a=self.raw_matrix(d);norm=torch.linalg.matrix_norm(a,ord=2);return a*torch.clamp(.95/(norm+1e-9),max=1.)
 def constrain(self):
  with torch.no_grad():
   if self.m=='discrete':
    for i in range(4):
     n=torch.linalg.matrix_norm(self.a[i],ord=2)
     if n>.95:self.a[i].mul_(.95/n)
   elif self.m=='linear_time':
    for d in range(1,DEPTH+1):
     n=torch.linalg.matrix_norm(self.a+d*self.b,ord=2)
     if n>.95:self.b.mul_(.95/n)
   else:
    n=torch.linalg.matrix_norm(self.a,ord=2)
    if n>.95:self.a.mul_(.95/n)
 def trajectory(self,x,depth):
  z=x;out=[]
  for d in range(1,depth+1):z=z@self.matrix(d).T;out.append(z)
  return torch.stack(out,1)
def serialize(m):
 b=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},b,_use_new_zipfile_serialization=True);return b.getvalue()
def run(m,w,s,lr,split):
 fixseed(w*100+s);x,y,base,omega=world(w);model=Model(m);opt=torch.optim.AdamW(model.parameters(),lr=lr);st=time.perf_counter()
 for _ in range(UPDATES):
  ix=torch.randint(len(x),(BATCH,));pred=model.trajectory(x[ix],TRAIN_DEPTH);loss=((pred-y[ix,:TRAIN_DEPTH])**2).mean();opt.zero_grad();loss.backward();opt.step();model.constrain()
 wall=time.perf_counter()-st;vx=torch.randn(256,D,generator=torch.Generator().manual_seed(w+1000));vy=teacher_trajectory(vx,base,omega);pred=model.trajectory(vx,DEPTH)
 per=[]
 for d in range(DEPTH):per.append(((pred[:,d]-vy[:,d]).square().mean().sqrt()/(vy[:,d].square().mean().sqrt()+1e-9)).item())
 raw=serialize(model);name=f'{split}_{w}_{s}_{m}_lr{lr:g}.pt';(OUT/'payloads'/name).write_bytes(raw);cl=Model(m);cl.load_state_dict(torch.load(io.BytesIO(raw),weights_only=True));rb=serialize(cl);assert raw==rb;norms=[torch.linalg.matrix_norm(model.matrix(d),ord=2).item() for d in range(1,DEPTH+1)]
 return {'split':split,'world':w,'seed':s,'method':m,'lr':lr,'payload_file':name,'serialized_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'roundtrip_sha256':hashlib.sha256(rb).hexdigest(),'examples':UPDATES*BATCH,'updates':UPDATES,'active_mac_proxy_per_step':4,'recurrence_steps':DEPTH,'wall_time_s':round(wall,6),'mean_error_depth1_4':sum(per[:4])/4,'mean_error_depth5_8':sum(per[4:])/4,'error_by_depth':per,'spectral_norms_by_depth':norms}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);(OUT/'payloads').mkdir(exist_ok=True);rows=[];sel={}
 if a.phase=='dev':
  for m in METHODS:
   sc={lr:[] for lr in LRS}
   for lr in LRS:
    for w in DEV:
     for s in SEEDS:r=run(m,w,s,lr,'development');rows.append(r);sc[lr].append(r['mean_error_depth1_4'])
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
