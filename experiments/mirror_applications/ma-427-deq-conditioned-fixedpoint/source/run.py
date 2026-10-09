#!/usr/bin/env python3
import argparse,hashlib,io,json,math,random,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';METHODS=['shared','mirror','scalar','rank1','independent'];DEV=[42700,42701];FRESH=[42710,42711,42712];SEEDS=[0,1,2];LRS=[.003,.01];M,D=4,2;TRAIN_N,TEST_N=512,256;UPDATES,BATCH=300,64;TRAIN_ITERS=16;TOL=1e-5;MAX_ITERS=128
def fixseed(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def rot(a):
 c=a.cos();s=a.sin();return torch.stack([torch.stack([c,-s]),torch.stack([s,c])])
def make_world(w):
 g=torch.Generator().manual_seed(w);q=rot(torch.rand((),generator=g)*math.pi);base=q@torch.diag(torch.tensor([.25,.55]))@q.T;angles=(torch.rand(M,generator=g)*1.2-.6);As=torch.stack([rot(a)@base@rot(a).T for a in angles]);B=torch.randn(D,D,generator=g)*.18
 def sample(n,offset):
  x=torch.randn(M,n,D,generator=torch.Generator().manual_seed(w+offset));out=[]
  for c in range(M):
   z=torch.zeros(n,D)
   for _ in range(100):z=torch.tanh(z@As[c].T+x[c]@B.T)
   out.append(z)
  return x,torch.stack(out)
 tx,ty=sample(TRAIN_N,10);vx,vy=sample(TEST_N,20)
 return tx,ty,vx,vy,As,B
class Model(nn.Module):
 def __init__(self,m):
  super().__init__();self.m=m;self.b=nn.Parameter(torch.randn(D,D)*.1)
  if m=='independent':self.a=nn.Parameter(torch.randn(M,D,D)*.1)
  else:self.a=nn.Parameter(torch.randn(D,D)*.1)
  if m=='mirror':self.angles=nn.Parameter(torch.zeros(M))
  if m=='scalar':self.gate=nn.Parameter(torch.ones(M))
  if m=='rank1':self.u=nn.Parameter(torch.randn(M,D,1)*.01);self.v=nn.Parameter(torch.randn(M,1,D)*.01)
 def matrices(self):
  if self.m=='independent':return self.a
  if self.m=='mirror':return torch.stack([rot(self.angles[i])@self.a@rot(self.angles[i]).T for i in range(M)])
  if self.m=='scalar':return self.gate[:,None,None]*self.a
  if self.m=='rank1':return self.a[None]+self.u@self.v
  return self.a[None].expand(M,-1,-1)
 def constrain(self):
  with torch.no_grad():
   mats=self.matrices()
   for mat in mats:
    n=torch.linalg.matrix_norm(mat,ord=2)
    if n>.8:
     scale=.8/n
     if self.m=='independent':self.a.data[torch.where((mats==mat).all((1,2)))[0][0]].mul_(scale)
     elif self.m=='scalar':self.gate.data[(mats==mat).all((1,2)).nonzero()[0][0]].mul_(scale)
     elif self.m=='rank1':
      idx=(mats==mat).all((1,2)).nonzero()[0][0];self.u.data[idx].mul_(scale)
     else:self.a.data.mul_(scale)
 def forward(self,x,mode,iters=TRAIN_ITERS):
  aa=self.matrices()[mode];bb=self.b.expand(len(x),-1,-1);bx=torch.bmm(bb,x.unsqueeze(-1)).squeeze(-1);z=torch.zeros_like(x)
  for _ in range(iters):z=torch.tanh(torch.bmm(aa,z.unsqueeze(-1)).squeeze(-1)+bx)
  return z
 def solve(self,x,mode):
  aa=self.matrices()[mode];bx=torch.bmm(self.b.expand(len(x),-1,-1),x.unsqueeze(-1)).squeeze(-1);z=torch.zeros_like(x)
  for i in range(1,MAX_ITERS+1):
   zn=torch.tanh(torch.bmm(aa,z.unsqueeze(-1)).squeeze(-1)+bx);err=(zn-z).abs().amax(-1);z=zn
   if (err<TOL).all():return z,i,True
  return z,MAX_ITERS,False
def serialize(model):
 b=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},b,_use_new_zipfile_serialization=True);return b.getvalue()
def run(m,w,s,lr,split):
 fixseed(w*100+s);tx,ty,vx,vy,As,B=make_world(w);model=Model(m);opt=torch.optim.AdamW(model.parameters(),lr=lr);st=time.perf_counter();x=tx.reshape(-1,D);y=ty.reshape(-1,D);mode=torch.arange(M).repeat_interleave(TRAIN_N)
 for _ in range(UPDATES):
  ix=torch.randint(len(x),(BATCH,));pred=model(x[ix],mode[ix]);loss=((pred-y[ix])**2).mean();opt.zero_grad();loss.backward();opt.step();model.constrain()
 wall=time.perf_counter()-st;errs=[];iters=[];fails=[];norms=[]
 with torch.no_grad():
  for c in range(M):
   z,n,ok=model.solve(vx[c],torch.full((TEST_N,),c));errs.append(((z-vy[c]).square().mean().sqrt()/(vy[c].square().mean().sqrt()+1e-9)).item());iters.append(n);fails.append(not ok);norms.append(torch.linalg.matrix_norm(model.matrices()[c],ord=2).item())
 raw=serialize(model);name=f'{split}_{w}_{s}_{m}_lr{lr:g}.pt';(OUT/'payloads'/name).write_bytes(raw);cl=Model(m);cl.load_state_dict(torch.load(io.BytesIO(raw),weights_only=True));rb=serialize(cl);assert raw==rb
 return {'split':split,'world':w,'seed':s,'method':m,'lr':lr,'payload_file':name,'serialized_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'roundtrip_sha256':hashlib.sha256(rb).hexdigest(),'examples':UPDATES*BATCH,'updates':UPDATES,'active_mac_proxy_per_state_per_iter':4,'wall_time_s':round(wall,6),'mean_equilibrium_nrmse':sum(errs)/M,'per_code_nrmse':errs,'solve_iterations_per_code':iters,'nonconvergence_per_code':fails,'max_operator_spectral_norm':max(norms)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);(OUT/'payloads').mkdir(exist_ok=True);rows=[];sel={}
 if a.phase=='dev':
  for m in METHODS:
   sc={lr:[] for lr in LRS}
   for lr in LRS:
    for w in DEV:
     for s in SEEDS:r=run(m,w,s,lr,'development');rows.append(r);sc[lr].append(r['mean_equilibrium_nrmse'])
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
