#!/usr/bin/env python3
import argparse,hashlib,io,json,math,random,time,re
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';METHODS=['continuous2','mirror2','mirror_cb4','mirror_cb8','mirror_cb16','latent_cb4','latent_cb8','latent_cb16','independent'];DEV=[41700,41701];FRESH=[41710,41711,41712];SEEDS=[0,1,2];LRS=[.003,.01];NF,H=16,32;UPDATES,BATCH=500,256
def fixseed(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def data(w):
 g=torch.Generator().manual_seed(w);coef=torch.randn(NF,6,generator=g)*.45;tx=torch.rand(NF,64,1,generator=g)*2*math.pi-math.pi;vx=torch.rand(NF,128,1,generator=g)*2*math.pi-math.pi
 def target(x):
  y=torch.zeros(x.shape[:-1])
  for k in range(3):y+=coef[:,None,2*k]*torch.sin((k+1)*x[...,0])+coef[:,None,2*k+1]*torch.cos((k+1)*x[...,0])
  return y
 return tx,target(tx),vx,target(vx)
class Model(nn.Module):
 def __init__(self,m):
  super().__init__();self.m=m;self.kind='mirror' if m.startswith('mirror') else 'latent';self.k=int(m.split('cb')[-1]) if '_cb' in m else 0;self.d=2 if m=='continuous2' or m.startswith('latent_cb') else 0
  self.w1=nn.Parameter(torch.randn(1+self.d,H)*.2);self.b1=nn.Parameter(torch.zeros(H));self.w2=nn.Parameter(torch.randn(H,1)*.2);self.b2=nn.Parameter(torch.zeros(1))
  if m=='continuous2':self.codes=nn.Parameter(torch.randn(NF,2)*.03)
  elif m=='mirror2':self.codes=nn.Parameter(torch.zeros(NF,2))
  elif self.k:self.codebook=nn.Parameter(torch.randn(self.k,2)*.03);self.assign_logits=nn.Parameter(torch.randn(NF,self.k)*.01)
  elif m=='independent':self.w1=nn.Parameter(torch.randn(NF,1,H)*.2);self.b1=nn.Parameter(torch.zeros(NF,H));self.w2=nn.Parameter(torch.randn(NF,H,1)*.2);self.b2=nn.Parameter(torch.zeros(NF,1))
 def indices(self):return self.assign_logits.argmax(-1).to(torch.uint8)
 def select_code(self,f):
  if self.k:
   probs=torch.softmax(self.assign_logits[f],-1);hard=F.one_hot(probs.argmax(-1),self.k).to(probs.dtype);st=hard-probs.detach()+probs if self.training else hard
   return st@self.codebook
  return self.codes[f]
 def forward(self,x,f):
  if self.m=='independent':
   h=F.relu(torch.bmm(x[:,None,:],self.w1[f]).squeeze(1)+self.b1[f]);return (torch.bmm(h[:,None,:],self.w2[f]).squeeze(1)+self.b2[f]).squeeze(-1)
  if self.m=='continuous2' or (self.k and self.kind=='latent'):
   code=self.select_code(f);x=torch.cat([x,code],-1)
  h=F.relu(x@self.w1+self.b1)
  if self.kind=='mirror' and self.m!='continuous2':
   code=self.select_code(f);z=h.clone()
   for k in range(2):
    i,j=2*k,2*k+1;a,b=z[:,i].clone(),z[:,j].clone();co=code[:,k].cos();si=code[:,k].sin();z[:,i]=co*a-si*b;z[:,j]=si*a+co*b
   h=z
  return (h@self.w2+self.b2).squeeze(-1)
 def inference_state(self):
  d={k:v.detach().cpu().contiguous() for k,v in self.state_dict().items()}
  if self.k:
   d.pop('assign_logits');d['selected_indices']=self.indices().cpu().contiguous()
  return d
def serialize_state(d):
 b=io.BytesIO();torch.save(d,b,_use_new_zipfile_serialization=True);return b.getvalue()
def run(m,w,s,lr,split):
 fixseed(w*100+s);tx,ty,vx,vy=data(w);model=Model(m);opt=torch.optim.AdamW(model.parameters(),lr=lr);st=time.perf_counter();flat=tx.reshape(-1,1);targets=ty.reshape(-1);fid=torch.arange(NF).repeat_interleave(tx.size(1))
 for _ in range(UPDATES):
  ix=torch.randint(len(flat),(BATCH,));loss=F.mse_loss(model(flat[ix],fid[ix]),targets[ix]);opt.zero_grad();loss.backward();opt.step()
 wall=time.perf_counter()-st;model.eval();per=[]
 with torch.no_grad():
  for f in range(NF):
   p=model(vx[f],torch.full((vx.size(1),),f));per.append(((p-vy[f]).square().mean().sqrt()/(vy[f].square().mean().sqrt()+1e-9)).item())
 raw=serialize_state(model.inference_state());name=f'{split}_{w}_{s}_{m}_lr{lr:g}.pt';(OUT/'payloads'/name).write_bytes(raw);loaded=torch.load(io.BytesIO(raw),weights_only=True);rb=serialize_state(loaded);assert raw==rb
 mac=(1+(2 if m=='continuous2' or (m.startswith('latent_cb')) else 0))*H+H+(8 if m.startswith('mirror') else 0)
 unique=len(set(int(x) for x in model.indices().tolist())) if model.k else NF
 return {'split':split,'world':w,'seed':s,'method':m,'lr':lr,'payload_file':name,'serialized_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'roundtrip_sha256':hashlib.sha256(rb).hexdigest(),'examples':UPDATES*BATCH,'updates':UPDATES,'active_mac_proxy_per_query':mac,'wall_time_s':round(wall,6),'mean_normalized_rmse':sum(per)/NF,'per_function_normalized_rmse':per,'codebook_entries':model.k if model.k else NF,'unique_codes_used':unique}
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

