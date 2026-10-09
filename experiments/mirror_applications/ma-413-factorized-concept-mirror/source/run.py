#!/usr/bin/env python3
import argparse,hashlib,io,json,math,random,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';METHODS=['shared','factor_mirror','factor_film','factor_rank1','direct_table','oracle_independent'];DEV=[41300,41301];FRESH=[41310,41311,41312];SEEDS=[0,1,2];LRS=[.003,.01];D,H,C=32,48,10;UPDATES,BATCH=300,128;SEEN=[0,1,2]
def fixseed(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def rotate(x,angles,attrs):
 z=x.clone()
 for k,bit in enumerate([attrs%2,attrs//2]):
  i,j=2*k,2*k+1;a,b=z[:,i].clone(),z[:,j].clone();q=angles[k]*bit;co=q.cos();si=q.sin();z[:,i]=co*a-si*b;z[:,j]=si*a+co*b
 return z
def data(world):
 g=torch.Generator().manual_seed(world);x=torch.randn(4,2048,D,generator=g);base1=torch.randn(D,H,generator=g)/math.sqrt(D);base2=torch.randn(H,C,generator=g)/math.sqrt(H);angles=torch.tensor([.62,-.48]);ys=[]
 for a in range(4):ys.append((F.relu(rotate(x[a],angles,a)@base1)@base2).argmax(-1))
 train_x=torch.cat([x[a,:1024] for a in SEEN]);train_y=torch.cat([ys[a][:1024] for a in SEEN]);train_c=torch.cat([torch.full((1024,),a) for a in SEEN]);val={a:(x[a,1024:1280],ys[a][1024:1280]) for a in SEEN};test={a:(x[a,1280:1792],ys[a][1280:1792]) for a in range(4)}
 oracle_x=x[:,:1024].reshape(-1,D);oracle_y=torch.stack([ys[a][:1024] for a in range(4)]).reshape(-1);oracle_c=torch.arange(4).repeat_interleave(1024)
 return train_x,train_c,train_y,val,test,(oracle_x,oracle_c,oracle_y)
class Model(nn.Module):
 def __init__(self,m):
  super().__init__();self.m=m;self.w1=nn.Parameter(torch.randn(D,H)*.12);self.w2=nn.Parameter(torch.randn(H,C)*.12);self.b1=nn.Parameter(torch.zeros(H));self.b2=nn.Parameter(torch.zeros(C))
  if m=='factor_mirror':self.angles=nn.Parameter(torch.zeros(2))
  if m=='factor_film':self.scale=nn.Parameter(torch.ones(2,H));self.shift=nn.Parameter(torch.zeros(2,H))
  if m=='factor_rank1':self.a=nn.Parameter(torch.randn(2,D,1)*.01);self.b=nn.Parameter(torch.randn(2,1,H)*.01)
  if m=='direct_table':self.codes=nn.Parameter(torch.zeros(4,2))
  if m=='oracle_independent':self.w1=nn.Parameter(torch.randn(4,D,H)*.12);self.w2=nn.Parameter(torch.randn(4,H,C)*.12);self.b1=nn.Parameter(torch.zeros(4,H));self.b2=nn.Parameter(torch.zeros(4,C))
 def forward(self,x,c):
  if self.m=='oracle_independent':h=F.relu(torch.bmm(x[:,None,:],self.w1[c]).squeeze(1)+self.b1[c]);return torch.bmm(h[:,None,:],self.w2[c]).squeeze(1)+self.b2[c]
  if self.m=='factor_mirror':z=rotate(x,self.angles,c)
  elif self.m=='direct_table':
   z=x.clone()
   for k in range(2):
    i,j=2*k,2*k+1;q=self.codes[c,k];a,b=z[:,i].clone(),z[:,j].clone();z[:,i]=q.cos()*a-q.sin()*b;z[:,j]=q.sin()*a+q.cos()*b
  else:z=x
  w=self.w1
  if self.m=='factor_rank1':w=w+torch.einsum('ba,adh->bdh',torch.stack([c%2,c//2],1).float(),self.a@self.b)
  h=torch.bmm(z[:,None,:],w).squeeze(1)+self.b1 if w.ndim==3 else z@w+self.b1;h=F.relu(h)
  if self.m=='factor_film':bits=torch.stack([c%2,c//2],1).float();h=h*(1+bits@self.scale)+bits@self.shift
  return h@self.w2+self.b2
def serialize(m):
 b=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},b,_use_new_zipfile_serialization=True);return b.getvalue()
def eval_accuracy(model,data_by_context):
 out={}
 with torch.no_grad():
  for c,(x,y) in data_by_context.items():out[str(c)]=(model(x,torch.full((len(x),),c)).argmax(-1)==y).float().mean().item()
 return out
def run(m,w,s,lr,split):
 fixseed(w*100+s);tx,tc,ty,val,test,oracle=data(w);model=Model(m);opt=torch.optim.AdamW(model.parameters(),lr=lr);st=time.perf_counter();train_x,train_c,train_y=oracle if m=='oracle_independent' else (tx,tc,ty)
 for _ in range(UPDATES):
  ix=torch.randint(len(train_x),(BATCH,));loss=F.cross_entropy(model(train_x[ix],train_c[ix]),train_y[ix]);opt.zero_grad();loss.backward();opt.step()
 wall=time.perf_counter()-st;v=eval_accuracy(model,val);t=eval_accuracy(model,test);seen=[t[str(c)] for c in SEEN];raw=serialize(model);name=f'{split}_{w}_{s}_{m}_lr{lr:g}.pt';(OUT/'payloads'/name).write_bytes(raw);cl=Model(m);cl.load_state_dict(torch.load(io.BytesIO(raw),weights_only=True));rb=serialize(cl);assert raw==rb
 params=sum(p.numel() for p in model.parameters());mac=2*D*H+2*H*C+(8 if m in ('factor_mirror','direct_table') else 2*H if m=='factor_film' else 2*D*H if m=='factor_rank1' else 0)
 return {'split':split,'world':w,'seed':s,'method':m,'lr':lr,'payload_file':name,'serialized_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'roundtrip_sha256':hashlib.sha256(rb).hexdigest(),'examples':UPDATES*BATCH,'updates':UPDATES,'active_mac_proxy_per_example':mac,'parameter_count_diagnostic':params,'wall_time_s':round(wall,6),'validation_seen_accuracy':v,'fresh_seen_accuracy':sum(seen)/len(seen),'fresh_heldout_11_accuracy':t['3'],'fresh_per_combination_accuracy':t}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);(OUT/'payloads').mkdir(exist_ok=True);rows=[];sel={}
 if a.phase=='dev':
  for m in METHODS:
   scores={lr:[] for lr in LRS}
   for lr in LRS:
    for w in DEV:
     for s in SEEDS:r=run(m,w,s,lr,'development');rows.append(r);scores[lr].append(sum(r['validation_seen_accuracy'].values())/len(r['validation_seen_accuracy']))
   sel[m]=max(LRS,key=lambda lr:sum(scores[lr])/len(scores[lr]))
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
