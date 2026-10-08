#!/usr/bin/env python3
import argparse,json,random,time,hashlib,io
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts';P=A/'payloads';STEPS=[1,2,4,8];DEV=[44700,44701];FRESH=[44710,44711,44712];SEEDS=[0,1,2]
def task(w,tid,seed,dom):
 g=torch.Generator().manual_seed(w*1000003+tid*977+seed*251+dom*31);z=torch.randn(2,generator=g)*.5
 sx=torch.randn(4,2,generator=g)
 if dom==1:sx=sx*torch.tensor([2.,.35])
 sy=sx@z;qx=torch.randn(128,2,generator=g)
 if dom==1:qx=qx*torch.tensor([2.,.35])
 return sx,sy,qx,qx@z
def adapt(eta,sx,sy,n):
 m=torch.zeros(2,requires_grad=True)
 for k in range(n):
  loss=(sx@m-sy).square().mean();g=torch.autograd.grad(loss,m)[0];m=(m-eta[k]*g).detach().requires_grad_(True)
 return m.detach()
def meta_adapt(eta,sx,sy,n=8):
 m=torch.zeros(2,requires_grad=True)
 for k in range(n):
  loss=(sx@m-sy).square().mean();g=torch.autograd.grad(loss,m,create_graph=True)[0];m=m-eta[k]*g
 return m
def learn(kind):
 torch.manual_seed(44777+len(kind));base=nn.Parameter(torch.full((8,2),-1.6));mod=nn.Parameter(torch.zeros(8,2)) if kind=='conditioned' else None
 opt=torch.optim.Adam([base]+([mod] if mod is not None else []),lr=.025);start=time.perf_counter()
 for it in range(200):
  loss=[]
  for j in range(16):
   dom=j%2 if kind in ('conditioned','separate') else j%2
   sx,sy,qx,qy=task(44750,300+j,it%5,dom)
   e=torch.nn.functional.softplus(base+(mod*(1 if dom else -1) if mod is not None else 0))+.0001
   if kind=='separate': # separate schedules use two independent halves via different base rows keyed by domain
    pass
   m=meta_adapt(e,sx,sy,8);loss.append((qx@m-qy).square().mean())
  l=torch.stack(loss).mean();opt.zero_grad();l.backward();opt.step()
 eta0=(torch.nn.functional.softplus(base-(mod if mod is not None else 0))+.0001).detach()
 eta1=(torch.nn.functional.softplus(base+(mod if mod is not None else 0))+.0001).detach()
 return eta0,eta1,time.perf_counter()-start
# More direct independent schedule baseline trained with domain-masked batches.
def learn_separate():
 etas=[];wall=0
 for d in [0,1]:
  torch.manual_seed(44777+d);raw=nn.Parameter(torch.full((8,2),-1.6));opt=torch.optim.Adam([raw],lr=.025);t=time.perf_counter()
  for it in range(200):
   ls=[]
   for j in range(16):
    sx,sy,qx,qy=task(44750,300+j,it%5,d);m=meta_adapt(torch.nn.functional.softplus(raw)+.0001,sx,sy,8);ls.append((qx@m-qy).square().mean())
   l=torch.stack(ls).mean();opt.zero_grad();l.backward();opt.step()
  etas.append((torch.nn.functional.softplus(raw).detach()+.0001));wall+=time.perf_counter()-t
 return etas[0],etas[1],wall
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','fresh'],required=True);a=ap.parse_args();A.mkdir(exist_ok=True);P.mkdir(exist_ok=True)
 if a.phase=='development':
  u0,u1,wu=learn('unconditioned');c0,c1,wc=learn('conditioned');s0,s1,ws=learn_separate()
  # Tune Adam by aggregate dev query error.
  scores={}
  for lr in [.01,.03,.1,.3]:
   vals=[]
   for w in DEV:
    for seed in SEEDS:
     for d in [0,1]:
      for tid in range(100,120):
       sx,sy,qx,qy=task(w,tid,seed,d);m=torch.zeros(2,requires_grad=True);o=torch.optim.Adam([m],lr=lr)
       for _ in range(8):o.zero_grad();((sx@m-sy).square().mean()).backward();o.step()
       vals.append((((qx@m.detach()-qy).square().mean().sqrt())/(qy.square().mean().sqrt()+1e-12)).item())
   scores[str(lr)]=sum(vals)/len(vals)
  torch.save({'unconditioned':[u0,u1],'conditioned':[c0,c1],'separate':[s0,s1]},P/'schedules.pt')
  (A/'development_selection.json').write_text(json.dumps({'adam_lr':min(scores,key=scores.get),'adam_scores':scores,'meta_wall_s':{'unconditioned':wu,'conditioned':wc,'separate':ws},'fresh_untouched':True},indent=2)+'\n');print(json.dumps({'adam':min(scores,key=scores.get),'meta_wall':{'u':wu,'c':wc,'s':ws}},indent=2));return
 sel=json.loads((A/'development_selection.json').read_text());ss=torch.load(P/'schedules.pt',weights_only=True);rows=[]
 for w in FRESH:
  for seed in SEEDS:
   for d in [0,1]:
    for method in ['adam','unconditioned','conditioned','separate']:
     codes={n:[] for n in STEPS};errs={n:[] for n in STEPS};walls={n:[] for n in STEPS}
     for tid in range(3000,3010):
      sx,sy,qx,qy=task(w,tid,seed,d)
      for n in STEPS:
       t=time.perf_counter()
       if method=='adam':
        m=torch.zeros(2,requires_grad=True);op=torch.optim.Adam([m],lr=float(sel['adam_lr']))
        for _ in range(n):op.zero_grad();((sx@m-sy).square().mean()).backward();op.step()
        m=m.detach()
       else:
        idx=0 if d==0 else 1;key={'unconditioned':'unconditioned','conditioned':'conditioned','separate':'separate'}[method];m=adapt(ss[key][idx],sx,sy,n)
       walls[n].append(time.perf_counter()-t);errs[n].append((((qx@m-qy).square().mean().sqrt())/(qy.square().mean().sqrt()+1e-12)).item());codes[n].append(m)
     for n in STEPS:
      config={'step':n,'domain_code':d if method=='conditioned' else None,'adam_lr':sel['adam_lr'] if method=='adam' else None}
      obj={'format':'ma447-v1','method':method,'domain':d,'task_codes':torch.stack(codes[n]),'policy':(ss[method][d] if method in ('conditioned','separate') else ss['unconditioned'][0] if method=='unconditioned' else torch.tensor([float(sel['adam_lr'])])),'config':config}
      b=io.BytesIO();torch.save(obj,b);data=b.getvalue();path=P/f'{w}_{seed}_{d}_{method}_{n}.pt';path.write_bytes(data)
      rows.append({'world':w,'seed':seed,'domain':d,'method':method,'step':n,'nrmse_mean':sum(errs[n])/10,'payload_bytes':len(data),'bytes_per_task':len(data)/10,'query_wall_seconds_mean':sum(walls[n])/10,'hash':hashlib.sha256(data).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 (A/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'phase':'fresh','rows':len(rows)}))
if __name__=='__main__':main()
