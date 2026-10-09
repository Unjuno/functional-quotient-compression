#!/usr/bin/env python3
import csv,hashlib,io,json,math,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; PAY=ART/'payloads'
DEV=[45500,45501]; FRESH=[45520,45521,45522]; SEEDS=[0,1,2]; STEPS=500

def rotation(t):
 c,s=torch.cos(t),torch.sin(t); z=torch.zeros_like(c); o=torch.ones_like(c)
 return torch.stack([c,-s,s,c],-1).reshape(*t.shape,2,2)
def teachers(world,seed,start,count):
 g=torch.Generator().manual_seed(world*1000003+seed*997+start*53)
 ph1=(torch.rand(count,generator=g)-.5)*2.4; ph2=(torch.rand(count,generator=g)-.5)*2.4
 k1=(torch.rand(count,generator=g)-.5)*1.8; k2=(torch.rand(count,generator=g)-.5)*1.8
 def shear(k):
  A=torch.eye(2).repeat(count,1,1); A[:,0,1]=k; return A
 A1=rotation(ph1)@shear(k1); A2=rotation(ph2)@shear(k2)
 return A1,A2

def data(world,seed,start,count):
 A1,A2=teachers(world,seed,start,count); T=A2@A1
 comm=torch.linalg.matrix_norm(A2@A1-A1@A2,dim=(-2,-1))
 g=torch.Generator().manual_seed(world*99131+seed*71+start*17)
 x=torch.randn(count,64,2,generator=g); q=torch.randn(count,512,2,generator=torch.Generator().manual_seed(world*1217+seed*79+start*29))
 y=x@T.transpose(-1,-2); qy=q@T.transpose(-1,-2)
 return A1,A2,T,comm,x,y,q,qy

def fit(x,y,kind):
 n=x.shape[0]; device=x.device
 torch.manual_seed(int(x[0,0,0].item()*0+4550000+n))
 if kind=='independent':
  p1=torch.nn.Parameter(torch.eye(2).repeat(n,1,1)); p2=torch.nn.Parameter(torch.eye(2).repeat(n,1,1)); params=[p1,p2]
 elif kind=='tied':
  W=torch.nn.Parameter(torch.eye(2).repeat(n,1,1)); params=[W]
 elif kind=='mirror':
  W=torch.nn.Parameter(torch.eye(2).repeat(n,1,1)); a=torch.nn.Parameter(torch.zeros(n)); b=torch.nn.Parameter(torch.zeros(n)); params=[W,a,b]
 else:
  W=torch.nn.Parameter(torch.eye(2).repeat(n,1,1)); u1=torch.nn.Parameter(torch.randn(n,2,1)*.02);v1=torch.nn.Parameter(torch.randn(n,1,2)*.02);u2=torch.nn.Parameter(torch.randn(n,2,1)*.02);v2=torch.nn.Parameter(torch.randn(n,1,2)*.02);params=[W,u1,v1,u2,v2]
 opt=torch.optim.Adam(params,lr=.03); begin=time.perf_counter()
 for _ in range(STEPS):
  opt.zero_grad()
  if kind=='independent': M=p2@p1
  elif kind=='tied': M=W@W
  elif kind=='mirror': M=(rotation(b)@W)@(rotation(a)@W)
  else: M=(W+u2@v2)@(W+u1@v1)
  pred=x@M.transpose(-1,-2); loss=(pred-y).square().mean(); loss.backward();opt.step()
 wall=time.perf_counter()-begin
 with torch.no_grad():
  if kind=='independent': state={'A1':p1.detach(),'A2':p2.detach()}; M=p2@p1
  elif kind=='tied':state={'W':W.detach()};M=W@W
  elif kind=='mirror':state={'W':W.detach(),'theta1':a.detach(),'theta2':b.detach()};M=(rotation(b)@W)@(rotation(a)@W)
  else:state={'W':W.detach(),'u1':u1.detach(),'v1':v1.detach(),'u2':u2.detach(),'v2':v2.detach()};M=(W+u2@v2)@(W+u1@v1)
  return state,M,wall

def payload(kind,state,N):
 obj={'format':'ma455-v1','method':kind,'N':N}
 obj.update({k:v[:N].cpu() for k,v in state.items()}); b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def run_phase(phase):
 worlds=DEV if phase=='development' else FRESH; rows=[]; selection=[]
 for w in worlds:
  for s in SEEDS:
   A1,A2,T,comm,x,y,q,qy=data(w,s,100 if phase=='development' else 4000,64)
   for kind in ['tied','mirror','lowrank','independent']:
    st,M,wall=fit(x,y,kind)
    qbegin=time.perf_counter(); pred=q@M.transpose(-1,-2); query_wall=(time.perf_counter()-qbegin)/64
    denom=qy.square().mean(dim=(1,2)).sqrt().clamp_min(1e-9); err=((pred-qy).square().mean(dim=(1,2)).sqrt()/denom)
    # Sequential order audit: reverse the two teacher blocks, using the same query points.
    swapped=(A1@A2); swap_pred=q@swapped.transpose(-1,-2); swap_err=((swap_pred-qy).square().mean(dim=(1,2)).sqrt()/denom)
    for N in [1,20,64]:
     b=payload(kind,st,N); path=PAY/f'{w}_{s}_{kind}_N{N}.pt';path.write_bytes(b)
     rows.append({'world':w,'seed':s,'method':kind,'n':N,'tasks':N,'nrmse_mean':float(err[:N].mean()),'payload_bytes':len(b),'bytes_per_task':len(b)/N,'compute_MAC_per_example':({'tied':8,'mirror':16,'lowrank':24,'independent':8}[kind]),'optimizer_updates':STEPS,'support_fit_wall_seconds':wall,'query_wall_seconds_mean':query_wall,'mean_commutator_fro':float(comm[:N].mean()),'order_swapped_nrmse':float(swap_err[:N].mean()),'hash':hashlib.sha256(b).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
    if phase=='development': selection.append({'world':w,'seed':s,'method':kind,'nrmse':float(err.mean()),'wall':wall})
 with open(ART/('development_runs.json' if phase=='development' else 'fresh_runs.jsonl'),'w') as f:
  if phase=='development':json.dump(selection,f,indent=2)
  else:f.write(''.join(json.dumps(r)+'\n' for r in rows))
 if phase=='fresh':
  with open(ROOT/'RESULTS_CORE.csv','w',newline='') as f: wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows),'settings':{'steps':STEPS,'lr':.03}}))
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','fresh'],required=True);run_phase(ap.parse_args().phase)
