#!/usr/bin/env python3
import argparse,hashlib,json,random,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2];OUT=ROOT/'artifacts';PAY=OUT/'payloads'
DEV=[43800,43801];FRESH=[43810,43811,43812];SEEDS=[0,1,2];METHODS=['shared','gate','mirror','rank2','independent'];LRS=[.003,.01]
N=8;ROLES=4;TRAIN_LEN=64;EVAL=[64,128];UPDATES=500;BATCH=64

def fix(s): random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def spectral_basis():
 i=torch.linspace(-1,1,N); slow=torch.exp(-((i+.55)/.48)**2); fast=torch.exp(-((i-.55)/.48)**2)
 return torch.stack([slow/slow.norm(),fast/fast.norm()])
BASIS=spectral_basis()
def teacher_logits(w):
 g=torch.Generator().manual_seed(w); base=torch.linspace(-1.25,-.12,N); coeff=torch.tensor([[-.25,-.1],[-.1,.25],[.25,-.05],[.08,.28]])+(torch.rand(ROLES,2,generator=g)-.5)*.06
 return base,coeff

def poles(logits):return .55+.43*torch.sigmoid(logits)
def gen(w,n,L):
 base,coef=teacher_logits(w); allp=poles(base+coef@BASIS);g=torch.Generator().manual_seed(w+77777+n+L)
 x=torch.randn(n,L,generator=g); roles=torch.randint(0,ROLES,(n,),generator=g); y=[]
 for b in range(n):
  z=torch.zeros(N); seq=[]
  for t in range(L): z=allp[roles[b]]*z+x[b,t];seq.append(z.sum()/N**.5)
  y.append(torch.stack(seq))
 return x,roles,torch.stack(y)
class Kernel(nn.Module):
 def __init__(self,method,w):
  super().__init__();self.method=method;base,_=teacher_logits(w); self.logits=nn.Parameter(base.clone())
  if method=='gate':self.gate=nn.Parameter(torch.ones(ROLES))
  if method=='mirror':self.code=nn.Parameter(torch.zeros(ROLES,2))
  if method=='rank2':self.resid=nn.Parameter(torch.zeros(ROLES,2))
  if method=='independent':self.ind=nn.Parameter(base.repeat(ROLES,1).clone())
 def lp(self,r):
  if self.method=='independent':return self.ind[r]
  if self.method=='mirror':return self.logits+self.code[r]@BASIS
  if self.method=='rank2':
   v=torch.zeros(N);v[0]=self.resid[r,0];v[-1]=self.resid[r,1];return self.logits+v
  return self.logits
 def forward(self,x,roles,L=None):
  L=L or x.shape[1]; p=torch.stack([poles(self.lp(r)) for r in range(ROLES)])[roles];z=torch.zeros(x.shape[0],N);out=[]
  for t in range(L):
   z=p*z+x[:,t,None]; y=z.sum(-1)/N**.5
   if self.method=='gate': y=y*self.gate[roles]
   out.append(y)
  return torch.stack(out,1)
def save(m,method,w,s,phase):
 p=PAY/f'{phase}_{w}_{s}_{method}.pt';torch.save({'method':method,'world':w,'state':m.state_dict()},p);b=p.read_bytes();return len(b),hashlib.sha256(b).hexdigest(),str(p.relative_to(REPO))
def fit(w,s,meth,lr,phase):
 fix(w*100+s*13+sum(map(ord,meth)));x,r,y=gen(w,1024,TRAIN_LEN);m=Kernel(meth,w);opt=torch.optim.AdamW(m.parameters(),lr=lr);st=time.perf_counter()
 for _ in range(UPDATES):
  idx=torch.randint(len(x),(BATCH,));loss=(m(x[idx],r[idx])-y[idx]).square().mean();opt.zero_grad();loss.backward();opt.step()
 wall=time.perf_counter()-st;nb,h,path=save(m,meth,w,s,phase);return m,{'serialized_bytes':nb,'hash':h,'path':path,'train_wall':wall}
def metric(m,w,L):
 x,r,y=gen(w,256,L);st=time.perf_counter()
 with torch.no_grad():pred=m(x,r,L)
 wall=time.perf_counter()-st
 return ((pred-y).square().mean().sqrt()/(y.square().mean().sqrt()+1e-12)).item(),wall,max(poles(m.lp(i)).max().item() for i in range(ROLES))
def mac(m):return {'shared':N,'gate':N+1,'mirror':N+2,'rank2':N+2,'independent':N}[m]
def replay(path):
 d=torch.load(path,map_location='cpu',weights_only=False);m=Kernel(d['method'],d['world']);m.load_state_dict(d['state']);return m,d

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);PAY.mkdir(parents=True,exist_ok=True);rows=[];worlds=DEV if a.phase=='development' else FRESH
 chosen=json.loads((OUT/'development_selection.json').read_text())['selected_learning_rate_by_method'] if a.phase=='fresh' else {}
 for w in worlds:
  for meth in METHODS:
   if a.phase=='development':
    score={lr:[] for lr in LRS}
    for lr in LRS:
     for s in SEEDS:
      m,meta=fit(w,s,meth,lr,a.phase);v,wall,rho=metric(m,w,TRAIN_LEN);score[lr].append(v)
      rows.append({'condition':'development','world_or_seed':f'{w}-{s}','method':meth,'serialized_bytes':meta['serialized_bytes'],'train_tokens_or_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':f'{mac(meth)} MAC/step','wall_time_s':round(meta['train_wall'],6),'primary_metric':'sequence_NRMSE','primary_value':v,'secondary_metric':'max_pole','secondary_value':rho,'status_note':f'lr={lr}; hash={meta["hash"]}; payload={meta["path"]}'})
    chosen[meth]=min(LRS,key=lambda lr:sum(score[lr])/len(score[lr]))
   else:
    for s in SEEDS:
     m,meta=fit(w,s,meth,chosen[meth],a.phase)
     for L in EVAL:
      v,wall,rho=metric(m,w,L);rows.append({'condition':'fresh','world_or_seed':f'{w}-{s}-{L}','method':meth,'serialized_bytes':meta['serialized_bytes'],'train_tokens_or_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':f'{mac(meth)} MAC/step','wall_time_s':round(wall,6),'primary_metric':'sequence_NRMSE','primary_value':v,'secondary_metric':'max_pole','secondary_value':rho,'status_note':f'lr={chosen[meth]}; train_wall={meta["train_wall"]:.6f}; hash={meta["hash"]}; payload={meta["path"]}'})
 if a.phase=='development':(OUT/'development_selection.json').write_text(json.dumps({'selected_learning_rate_by_method':chosen,'fresh_worlds_not_accessed':True,'rule':'lowest mean length-64 validation NRMSE across development worlds and seeds'},indent=2)+'\n')
 (OUT/f'{a.phase}_runs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));print(json.dumps({'phase':a.phase,'selection':chosen,'rows':len(rows)},indent=2))
if __name__=='__main__':main()
