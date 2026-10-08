#!/usr/bin/env python3
import argparse,hashlib,io,json,math,random,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';METHODS=['shared','mirror','rank1','independent'];DEV=[43600,43601];FRESH=[43610,43611,43612];SEEDS=[0,1,2];LRS=[.003,.01];R,D,T=4,8,32;UPDATES,BATCH=300,32
def fixseed(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def make_rot(ang):
 q=torch.eye(D)
 for k in range(4):
  i,j=2*k,2*k+1;c=ang[k].cos();s=ang[k].sin();rot=torch.eye(D);rot[i,i]=c;rot[j,j]=c;rot[i,j]=-s;rot[j,i]=s;q=rot@q
 return q
def teacher(w):
 g=torch.Generator().manual_seed(w);a=torch.randn(D,D,generator=g);a=a*(.62/torch.linalg.matrix_norm(a,ord=2));b=torch.randn(D,D,generator=g)*.15;angles=torch.rand(R,4,generator=g)*.8-.4;As=torch.stack([make_rot(angles[r])@a@make_rot(angles[r]).T for r in range(R)]);return As,b
def sample(w,n,seqseed):
 As,b=teacher(w);g=torch.Generator().manual_seed(seqseed);x=torch.randn(n,T,D,generator=g);roles=torch.randint(R,(n,T),generator=g);ys=[]
 for i in range(n):
  h=torch.zeros(D);out=[]
  for t in range(T):h=As[roles[i,t]]@h+b@x[i,t];out.append(h)
  ys.append(torch.stack(out))
 return x,roles,torch.stack(ys)
class Model(nn.Module):
 def __init__(self,m):
  super().__init__();self.m=m;self.b=nn.Parameter(torch.randn(D,D)*.1)
  if m=='independent':self.a=nn.Parameter(torch.randn(R,D,D)*.1)
  else:self.a=nn.Parameter(torch.randn(D,D)*.1)
  if m=='mirror':self.angles=nn.Parameter(torch.zeros(R,4))
  if m=='rank1':self.u=nn.Parameter(torch.randn(R,D,1)*.01);self.v=nn.Parameter(torch.randn(R,1,D)*.01)
 def matrices(self):
  if self.m=='independent':return self.a
  if self.m=='mirror':return torch.stack([make_rot(self.angles[r])@self.a@make_rot(self.angles[r]).T for r in range(R)])
  if self.m=='rank1':return self.a[None]+self.u@self.v
  return self.a[None].expand(R,-1,-1)
 def forward(self,x,roles):
  aa=self.matrices();h=torch.zeros(x.size(0),D);out=[]
  for t in range(T):h=torch.bmm(aa[roles[:,t]],h[:,:,None]).squeeze(-1)+x[:,t]@self.b.T;out.append(h)
  return torch.stack(out,1)
def serialize(m):
 buf=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},buf,_use_new_zipfile_serialization=True);return buf.getvalue()
def run(m,w,s,lr,split):
 fixseed(w*100+s);x,roles,y=sample(w,256,w+100);model=Model(m);opt=torch.optim.AdamW(model.parameters(),lr=lr);st=time.perf_counter()
 for _ in range(UPDATES):
  ix=torch.randint(256,(BATCH,));pred=model(x[ix],roles[ix]);loss=((pred-y[ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
 train_wall=time.perf_counter()-st;vx,vr,vy=sample(w,128,w+2000);model.eval()
 with torch.no_grad():
  pred=model(vx,vr);err=((pred-vy).square().mean().sqrt()/(vy.square().mean().sqrt()+1e-9)).item();st=time.perf_counter()
  for _ in range(8):model(vx,vr)
  infer_wall=(time.perf_counter()-st)/8
 raw=serialize(model);name=f'{split}_{w}_{s}_{m}_lr{lr:g}.pt';(OUT/'payloads'/name).write_bytes(raw);cl=Model(m);cl.load_state_dict(torch.load(io.BytesIO(raw),weights_only=True));rb=serialize(cl);assert raw==rb
 mac=(2*D*D)+(D*D if m=='mirror' else D*4 if m=='rank1' else 0)
 return {'split':split,'world':w,'seed':s,'method':m,'lr':lr,'payload_file':name,'serialized_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'roundtrip_sha256':hashlib.sha256(rb).hexdigest(),'examples':UPDATES*BATCH,'updates':UPDATES,'active_mac_proxy_per_token':mac,'sequence_length':T,'train_wall_time_s':round(train_wall,6),'inference_wall_time_per_batch_s':round(infer_wall,6),'sequence_nrmse':err}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);(OUT/'payloads').mkdir(exist_ok=True);rows=[];sel={}
 if a.phase=='dev':
  for m in METHODS:
   sc={lr:[] for lr in LRS}
   for lr in LRS:
    for w in DEV:
     for s in SEEDS:r=run(m,w,s,lr,'development');rows.append(r);sc[lr].append(r['sequence_nrmse'])
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

