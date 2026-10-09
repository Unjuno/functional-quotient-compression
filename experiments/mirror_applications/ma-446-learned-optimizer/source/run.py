#!/usr/bin/env python3
import argparse,hashlib,io,json,random,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads'
D=2;STEPS=[1,2,4,8];DEV=[44600,44601];FRESH=[44610,44611,44612];SEEDS=[0,1,2]

def fix(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def task(w,tid,seed):
 g=torch.Generator().manual_seed(w*1000003+tid*997+seed*271)
 z=torch.randn(D,generator=g)*.5
 sx=torch.randn(4,D,generator=g); sy=sx@z
 qx=torch.randn(128,D,generator=g); qy=qx@z
 return z,sx,sy,qx,qy
def normerr(m,qx,qy):return (((qx@m-qy).square().mean().sqrt())/(qy.square().mean().sqrt()+1e-12)).item()
def learn_schedule():
 fix(44677); raw=nn.Parameter(torch.full((8,D),-1.5));opt=torch.optim.Adam([raw],lr=.03);start=time.perf_counter()
 for it in range(200):
  losses=[]
  for j in range(16):
   _,sx,sy,qx,qy=task(44700,100+j,it%7); m=torch.zeros(D,requires_grad=True)
   eta=torch.nn.functional.softplus(raw)+1e-4
   for k in range(8):
    loss=(sx@m-sy).square().mean();g=torch.autograd.grad(loss,m,create_graph=True)[0];m=m-eta[k]*g
   losses.append((qx@m-qy).square().mean())
  loss=torch.stack(losses).mean();opt.zero_grad();loss.backward();opt.step()
 return (torch.nn.functional.softplus(raw).detach()+1e-4),time.perf_counter()-start
def standard(method,lr,sx,sy,steps):
 m=torch.zeros(D,requires_grad=True);opt=torch.optim.SGD([m],lr=lr) if method=='sgd' else torch.optim.Adam([m],lr=lr)
 for _ in range(steps):
  opt.zero_grad();((sx@m-sy).square().mean()).backward();opt.step()
 return m.detach()
def learned(eta,sx,sy,steps):
 m=torch.zeros(D,requires_grad=True)
 for k in range(steps):
  l=(sx@m-sy).square().mean();g=torch.autograd.grad(l,m)[0];m=(m-eta[k]*g).detach().requires_grad_(True)
 return m
def payload(path,method,codes,config):
 obj={'format':'ma446-v1','method':method,'task_codes':torch.stack(codes),'config':config}
 b=io.BytesIO();torch.save(obj,b);data=b.getvalue();path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
 return len(data),hashlib.sha256(data).hexdigest(),str(path.relative_to(ROOT.parents[2]))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','fresh'],required=True);a=ap.parse_args();ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if a.phase=='development':
  eta,learnwall=learn_schedule();
  # Tune conventional methods on dev IDs 100..199.
  choices={}; scores={}
  for method,lrs in [('sgd',[.03,.1,.3,1.]),('adam',[.01,.03,.1,.3])]:
   for lr in lrs:
    es=[]
    for w in DEV:
     for seed in SEEDS:
      for tid in range(100,200):
       _,sx,sy,qx,qy=task(w,tid,seed);m=standard(method,lr,sx,sy,4);es.append(normerr(m,qx,qy))
    scores[(method,lr)]=sum(es)/len(es)
   choices[method]=min(lrs,key=lambda lr:scores[(method,lr)])
  torch.save(eta,PAY/'learned_schedule.pt')
  (ART/'development_selection.json').write_text(json.dumps({'lrs':choices,'learned_schedule':eta.tolist(),'learned_schedule_training_wall_s':learnwall,'rule':'minimum dev step-4 NRMSE; fresh untouched'},indent=2)+'\n')
  print(json.dumps({'phase':'development','choices':choices,'learned_eta':eta.tolist(),'learnwall':learnwall},indent=2));return
 sel=json.loads((ART/'development_selection.json').read_text());eta=torch.tensor(sel['learned_schedule']);rows=[]
 for w in FRESH:
  for seed in SEEDS:
   for method in ['zero','sgd','adam','learned']:
    lr=sel['lrs'].get(method,0.);by_step={s:[] for s in STEPS};codes={s:[] for s in STEPS};walls={s:[] for s in STEPS}
    for tid in range(2000,2020):
     _,sx,sy,qx,qy=task(w,tid,seed)
     for s in STEPS:
      st=time.perf_counter()
      if method=='zero':m=torch.zeros(D)
      elif method=='learned':m=learned(eta,sx,sy,s)
      else:m=standard(method,lr,sx,sy,s)
      walls[s].append(time.perf_counter()-st);by_step[s].append(normerr(m,qx,qy));codes[s].append(m)
    for s in STEPS:
     config={'steps':s,'lr':lr,'schedule':eta.tolist() if method=='learned' else None,'N':20}
     b,h,path=payload(PAY/f'{w}_{seed}_{method}_{s}.pt',method,codes[s],config)
     rows.append({'world':w,'seed':seed,'method':method,'step':s,'nrmse_mean':sum(by_step[s])/20,'payload_bytes':b,'bytes_per_task':b/20,'update_mac_per_task':s*D*D*2,'query_wall_seconds_mean':sum(walls[s])/20,'hash':h,'path':path})
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'phase':'fresh','rows':len(rows)},indent=2))
if __name__=='__main__':main()
