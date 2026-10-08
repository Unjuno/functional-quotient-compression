#!/usr/bin/env python3
import argparse,hashlib,json,random,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2];OUT=ROOT/'artifacts';PAY=OUT/'payloads'
DEV=[44000,44001];FRESH=[44010,44011,44012];SEEDS=[0,1,2];LRS=[.003,.01];METHODS=['shared','mirror','rank1','independent'];D=4;SKILLS=4;T=8;UPDATES=250;BATCH=64

def fix(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def rot(a):
 c=a.cos();s=a.sin();return torch.stack([torch.stack([c,-s]),torch.stack([s,c])])
def qrot(a):q=torch.eye(D);q[:2,:2]=rot(a);return q
def world(w):
 g=torch.Generator().manual_seed(w);q,_=torch.linalg.qr(torch.randn(D,D,generator=g));base=q@torch.diag(torch.tensor([.54,.62,.70,.78]))@q.T;angles=torch.tensor([-.6,-.2,.25,.62])+(torch.rand(SKILLS,generator=g)-.5)*.12;return base,angles
def mats(w):
 a,ang=world(w);return torch.stack([qrot(x)@a@qrot(x).T for x in ang])
def data(w,k,n):
 A=mats(w)[k];g=torch.Generator().manual_seed(w*100+k*7+n);x=torch.randn(n,D,generator=g)*.5;z=x;ys=[]
 for _ in range(T):z=z@A.T;ys.append(z)
 return x,torch.stack(ys,1)
class Model(nn.Module):
 def __init__(self,method,w):
  super().__init__();self.method=method;base,_=world(w);self.A=nn.Parameter(base.clone())
  if method=='mirror':self.angle=nn.Parameter(torch.zeros(SKILLS))
  if method=='rank1':self.u=nn.Parameter(torch.zeros(SKILLS,D));self.v=nn.Parameter(torch.zeros(SKILLS,D))
  if method=='independent':self.ind=nn.Parameter(base.repeat(SKILLS,1,1).clone())
 def matrix(self,k):
  if self.method=='independent':return self.ind[k]
  if self.method=='mirror':q=qrot(self.angle[k]);return q@self.A@q.T
  if self.method=='rank1':return self.A+torch.outer(self.u[k],self.v[k])
  return self.A
 def forward(self,x,k):
  a=self.matrix(k);z=x;ys=[]
  for _ in range(T):z=z@a.T;ys.append(z)
  return torch.stack(ys,1)
def score(m,w,k):
 x,y=data(w,k,256);st=time.perf_counter()
 with torch.no_grad():p=m(x,k)
 wall=time.perf_counter()-st;err=((p-y).square().mean().sqrt()/(y.square().mean().sqrt()+1e-12)).item();rho=torch.linalg.eigvals(m.matrix(k)).abs().max().item();return err,wall,rho
def save(m,w,s,phase):
 p=PAY/f'{phase}_{w}_{s}_{m.method}.pt';torch.save({'method':m.method,'world':w,'state':m.state_dict()},p);b=p.read_bytes();return len(b),hashlib.sha256(b).hexdigest(),str(p.relative_to(REPO))
def fit(w,s,meth,lr,phase):
 fix(w*100+s*19+sum(map(ord,meth)));m=Model(meth,w);opt=torch.optim.AdamW(m.parameters(),lr=lr);trainwall=0.;stage=[]
 for k in range(SKILLS):
  x,y=data(w,k,1024);st=time.perf_counter()
  for _ in range(UPDATES):
   ix=torch.randint(len(x),(BATCH,));loss=(m(x[ix],k)-y[ix]).square().mean();opt.zero_grad();loss.backward();opt.step()
  trainwall+=time.perf_counter()-st
  vals=[score(m,w,j)[0] for j in range(k+1)];stage.append(vals)
 nb,h,p=save(m,w,s,phase);return m,stage,{'bytes':nb,'hash':h,'path':p,'trainwall':trainwall}
def mac(m):return {'shared':D*D,'mirror':D*D+2,'rank1':D*D+2*D,'independent':D*D}[m]
def replay(path):
 d=torch.load(path,map_location='cpu',weights_only=False);m=Model(d['method'],d['world']);m.load_state_dict(d['state']);return m,d
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);PAY.mkdir(parents=True,exist_ok=True);worlds=DEV if a.phase=='development' else FRESH;rows=[];sel=json.loads((OUT/'development_selection.json').read_text())['selected_learning_rate_by_method'] if a.phase=='fresh' else {}
 for w in worlds:
  for meth in METHODS:
   if a.phase=='development':
    scores={lr:[] for lr in LRS}
    for lr in LRS:
     for s in SEEDS:
      m,st,meta=fit(w,s,meth,lr,a.phase);mean=sum(st[-1])/SKILLS;forget=max((st[-1][j]-st[j][j] for j in range(SKILLS-1)),default=0);scores[lr].append(mean+max(0,forget));rows.append({'condition':'development','world_or_seed':f'{w}-{s}','method':meth,'serialized_bytes':meta['bytes'],'train_tokens_or_examples':SKILLS*UPDATES*BATCH,'optimizer_updates':SKILLS*UPDATES,'active_compute_proxy':f'{mac(meth)} MAC/step','wall_time_s':round(meta['trainwall'],6),'primary_metric':'final_mean_skill_NRMSE','primary_value':mean,'secondary_metric':'max_forgetting','secondary_value':forget,'status_note':f'lr={lr}; stage_metrics={st}; hash={meta["hash"]}; payload={meta["path"]}'})
    sel[meth]=min(LRS,key=lambda lr:sum(scores[lr])/len(scores[lr]))
   else:
    for s in SEEDS:
     m,st,meta=fit(w,s,meth,sel[meth],a.phase)
     for k in range(SKILLS):
      err,wall,rho=score(m,w,k);forget=max((st[-1][j]-st[j][j] for j in range(k)),default=0);rows.append({'condition':'fresh','world_or_seed':f'{w}-{s}-skill{k}','method':meth,'serialized_bytes':meta['bytes'],'train_tokens_or_examples':SKILLS*UPDATES*BATCH,'optimizer_updates':SKILLS*UPDATES,'active_compute_proxy':f'{mac(meth)} MAC/step','wall_time_s':round(wall,6),'primary_metric':'skill_NRMSE','primary_value':err,'secondary_metric':'spectral_radius','secondary_value':rho,'status_note':f'forgetting={forget}; train_wall={meta["trainwall"]:.6f}; hash={meta["hash"]}; payload={meta["path"]}'})
 if a.phase=='development':(OUT/'development_selection.json').write_text(json.dumps({'selected_learning_rate_by_method':sel,'rule':'min mean final NRMSE plus positive max forgetting over dev worlds/seeds','fresh_worlds_not_accessed':True},indent=2)+'\n')
 (OUT/f'{a.phase}_runs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));print(json.dumps({'phase':a.phase,'selection':sel,'rows':len(rows)},indent=2))
if __name__=='__main__':main()
