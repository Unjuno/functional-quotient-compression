#!/usr/bin/env python3
import argparse,hashlib,json,random,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2];OUT=ROOT/'artifacts';PAY=OUT/'payloads'
DEV=[43900,43901];FRESH=[43910,43911,43912];SEEDS=[0,1,2];LRS=[.003,.01];METHODS=['shared','mirror','latent','independent'];D=4;K=4;P=4;UPDATES=400;BATCH=96

def fix(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def rot(a):
 c=a.cos();s=a.sin();return torch.stack([torch.stack([c,-s]),torch.stack([s,c])])
def qrot(a):
 q=torch.eye(D);q[:2,:2]=rot(a);return q
def world(w):
 g=torch.Generator().manual_seed(w);raw=torch.randn(D,D,generator=g);q,_=torch.linalg.qr(raw);eig=torch.tensor([.54,.63,.72,.81]);A=q@torch.diag(eig)@q.T;angles=torch.tensor([-.75,-.25,.32,.78])+(torch.rand(K,generator=g)-.5)*.1;return A,angles
def data(w,n):
 A,angles=world(w);g=torch.Generator().manual_seed(w+8181+n);x=torch.randn(n,D,generator=g)*.5;role=torch.randint(K,(n,),generator=g);ys=[]
 for i in range(n):
  Q=qrot(angles[role[i]]);M=Q@A@Q.T;z=x[i];ys.append(torch.stack([(z:=M@z) for _ in range(P)]))
 return x,role,torch.stack(ys)
class Model(nn.Module):
 def __init__(self,method,w):
  super().__init__();self.method=method;A,_=world(w);self.A=nn.Parameter(A.clone())
  if method=='mirror':self.angle=nn.Parameter(torch.zeros(K))
  if method=='latent':self.code=nn.Parameter(torch.zeros(K,2));self.basis=nn.Parameter(torch.randn(2,D,D)*.03)
  if method=='independent':self.ind=nn.Parameter(A.repeat(K,1,1).clone())
 def mat(self,r):
  if self.method=='independent':return self.ind[r]
  if self.method=='mirror':q=qrot(self.angle[r]);return q@self.A@q.T
  if self.method=='latent':return self.A+torch.einsum('j,jab->ab',self.code[r],self.basis)
  return self.A
 def forward(self,x,r):
  mats=torch.stack([self.mat(i) for i in range(K)])[r];z=x;ys=[]
  for _ in range(P):z=torch.bmm(mats,z.unsqueeze(-1)).squeeze(-1);ys.append(z)
  return torch.stack(ys,1)
def save(m,meth,w,s,phase):
 p=PAY/f'{phase}_{w}_{s}_{meth}.pt';torch.save({'method':meth,'world':w,'state':m.state_dict()},p);b=p.read_bytes();return len(b),hashlib.sha256(b).hexdigest(),str(p.relative_to(REPO))
def fit(w,s,meth,lr,phase):
 fix(w*100+s*17+sum(map(ord,meth)));x,r,y=data(w,2048);m=Model(meth,w);opt=torch.optim.AdamW(m.parameters(),lr=lr);st=time.perf_counter()
 for _ in range(UPDATES):
  ix=torch.randint(len(x),(BATCH,));loss=(m(x[ix],r[ix])-y[ix]).square().mean();opt.zero_grad();loss.backward();opt.step()
 tw=time.perf_counter()-st;nb,h,p=save(m,meth,w,s,phase);return m,{'bytes':nb,'hash':h,'path':p,'train_wall':tw}
def evaluate(m,w):
 x,r,y=data(w,512);st=time.perf_counter()
 with torch.no_grad():pred=m(x,r)
 wall=time.perf_counter()-st;err=((pred-y).square().mean().sqrt()/(y.square().mean().sqrt()+1e-12)).item();rho=max(torch.linalg.eigvals(m.mat(i)).abs().max().item() for i in range(K));return err,wall,rho
def mac(m):return {'shared':D*D,'mirror':D*D+8,'latent':D*D+2*D*D,'independent':D*D}[m]
def replay(path):
 d=torch.load(path,map_location='cpu',weights_only=False);m=Model(d['method'],d['world']);m.load_state_dict(d['state']);return m,d
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);PAY.mkdir(parents=True,exist_ok=True);worlds=DEV if a.phase=='development' else FRESH;rows=[];chosen=json.loads((OUT/'development_selection.json').read_text())['selected_learning_rate_by_method'] if a.phase=='fresh' else {}
 for w in worlds:
  for meth in METHODS:
   if a.phase=='development':
    scores={lr:[] for lr in LRS}
    for lr in LRS:
     for s in SEEDS:
      m,meta=fit(w,s,meth,lr,a.phase);err,wall,rho=evaluate(m,w);scores[lr].append(err);rows.append({'condition':'development','world_or_seed':f'{w}-{s}','method':meth,'serialized_bytes':meta['bytes'],'train_tokens_or_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':f'{mac(meth)} MAC/step','wall_time_s':round(meta['train_wall'],6),'primary_metric':'packet_NRMSE','primary_value':err,'secondary_metric':'spectral_radius','secondary_value':rho,'status_note':f'lr={lr}; hash={meta["hash"]}; payload={meta["path"]}'})
    chosen[meth]=min(LRS,key=lambda z:sum(scores[z])/len(scores[z]))
   else:
    for s in SEEDS:
     m,meta=fit(w,s,meth,chosen[meth],a.phase);err,wall,rho=evaluate(m,w);rows.append({'condition':'fresh','world_or_seed':f'{w}-{s}','method':meth,'serialized_bytes':meta['bytes'],'train_tokens_or_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':f'{mac(meth)} MAC/step','wall_time_s':round(wall,6),'primary_metric':'packet_NRMSE','primary_value':err,'secondary_metric':'spectral_radius','secondary_value':rho,'status_note':f'lr={chosen[meth]}; train_wall={meta["train_wall"]:.6f}; hash={meta["hash"]}; payload={meta["path"]}'})
 if a.phase=='development':(OUT/'development_selection.json').write_text(json.dumps({'selected_learning_rate_by_method':chosen,'rule':'lowest mean validation packet NRMSE over both development worlds and seeds 0-2','fresh_worlds_not_accessed':True},indent=2)+'\n')
 (OUT/f'{a.phase}_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'phase':a.phase,'selection':chosen,'rows':len(rows)},indent=2))
if __name__=='__main__':main()
