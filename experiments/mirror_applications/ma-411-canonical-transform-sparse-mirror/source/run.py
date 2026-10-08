#!/usr/bin/env python3
import argparse,hashlib,io,json,random,time,math
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';METHODS=['canonical','sparse_raw','sparse_cond','rank2','dense_residual','independent'];DEV=[41100,41101];FRESH=[41110,41111,41112];SEEDS=[0,1,2];LRS=[.003,.01];D=32;K=8;UPDATES=200;BATCH=128
def fixseed(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def hadamard(n=32):
 h=torch.ones((1,1))
 while h.size(0)<n:h=torch.cat([torch.cat([h,h],1),torch.cat([h,-h],1)],0)
 return h/math.sqrt(n)
def support():return torch.tensor([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]],dtype=torch.long)
def world_data(w):
 g=torch.Generator().manual_seed(w);x=torch.randn(4096,D,generator=g);h=hadamard();idx=support();ys=[]
 for c in range(4):
  vals=torch.randn(K,generator=g)*.035;m=h.clone();m[idx[:,0],idx[:,1]]+=vals;ys.append(m)
 cid=torch.arange(4096)%4;y=torch.stack([x[i]@ys[int(cid[i])] for i in range(4096)])
 return x,cid,y,ys
class Model(nn.Module):
 def __init__(self,m):
  super().__init__();self.m=m;self.register_buffer('indices',support());self.register_buffer('base',hadamard())
  if m.startswith('sparse'):self.values=nn.Parameter(torch.zeros(4,K))
  if m=='rank2':self.a=nn.Parameter(torch.randn(4,D,2)*.01);self.b=nn.Parameter(torch.randn(4,2,D)*.01)
  if m=='dense_residual':self.resid=nn.Parameter(torch.zeros(4,D,D))
  if m=='independent':self.mats=nn.Parameter(self.base[None].expand(4,-1,-1).clone()+torch.randn(4,D,D)*.01)
 def mat(self,c):
  if self.m=='independent':return self.mats[c]
  base=self.base
  if self.m.startswith('sparse'):
   out=base.clone();v=self.values[c]
   out[self.indices[:,0],self.indices[:,1]]=out[self.indices[:,0],self.indices[:,1]]+v
   return out
  if self.m=='rank2':return base+self.a[c]@self.b[c]
  if self.m=='dense_residual':return base+self.resid[c]
  return base
 def constrain(self):
  if self.m=='sparse_cond':
   # Since ||H^-1||2=1 and ||Delta||2<=||Delta||F=||values||2,
   # the 1/3 Frobenius bound guarantees cond(H+Delta)<=(1+1/3)/(1-1/3)=2.
   with torch.no_grad():self.values.mul_(min(1.,(1/3)/(self.values.norm().item()+1e-12)))
 def forward(self,x,c):
  out=x.new_zeros((x.size(0),D))
  for k in range(4):
   ix=(c==k).nonzero().squeeze(-1)
   if ix.numel():out=out.index_copy(0,ix,x[ix]@self.mat(k))
  return out
def serialize(model):
 b=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},b,_use_new_zipfile_serialization=True);return b.getvalue()
def run(m,w,s,lr,split):
 fixseed(w*100+s);x,c,y,_=world_data(w);model=Model(m);opt=torch.optim.AdamW([p for p in model.parameters()],lr=lr) if list(model.parameters()) else None;st=time.perf_counter()
 for _ in range(UPDATES):
  ix=torch.randint(0,3072,(BATCH,));pred=model(x[ix],c[ix]);loss=((pred-y[ix])**2).mean()
  if opt:opt.zero_grad();loss.backward();opt.step();model.constrain()
 wall=time.perf_counter()-st
 with torch.no_grad():
  pred=model(x[3072:],c[3072:]);norm=((pred-y[3072:])**2).mean().sqrt()/(y[3072:].square().mean().sqrt()+1e-12);conds=[torch.linalg.cond(model.mat(k)).item() for k in range(4)];per=[(((pred[c[3072:]==k]-y[3072:][c[3072:]==k])**2).mean().sqrt()/(y[3072:][c[3072:]==k].square().mean().sqrt()+1e-12)).item() for k in range(4)]
 raw=serialize(model);name=f'{split}_{w}_{s}_{m}_lr{lr:g}.pt';(OUT/'payloads'/name).write_bytes(raw);cl=Model(m);cl.load_state_dict(torch.load(io.BytesIO(raw),weights_only=True));rb=serialize(cl);assert raw==rb
 mac={'canonical':D*D,'sparse_raw':D*D+K,'sparse_cond':D*D+K,'rank2':D*D+2*D*2,'dense_residual':2*D*D,'independent':D*D}[m]
 return {'split':split,'world':w,'seed':s,'method':m,'lr':lr,'payload_file':name,'serialized_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'roundtrip_sha256':hashlib.sha256(rb).hexdigest(),'examples':UPDATES*BATCH,'updates':UPDATES,'active_mac_proxy_per_example':mac,'wall_time_s':round(wall,6),'normalized_rmse':norm.item(),'condition_numbers':conds,'per_context_normalized_rmse':per}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);(OUT/'payloads').mkdir(exist_ok=True);rows=[];sel={}
 if a.phase=='dev':
  for m in METHODS:
   scores={lr:[] for lr in LRS}
   for lr in LRS:
    for w in DEV:
     for s in SEEDS:r=run(m,w,s,lr,'development');rows.append(r);scores[lr].append(r['normalized_rmse'])
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
