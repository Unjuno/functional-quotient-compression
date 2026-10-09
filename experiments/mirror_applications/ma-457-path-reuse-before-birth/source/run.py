#!/usr/bin/env python3
import csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads'
DEV=[45700,45701];FRESH=[45710,45711,45712];SEEDS=[0,1,2];THRESHOLDS=[.01,.03,.1];ANGLES=torch.linspace(-math.pi,math.pi,361)
def rot(t):
 c,s=torch.cos(t),torch.sin(t);return torch.stack([c,-s,s,c]).reshape(2,2)
def seq(world,seed,start=300,count=32):
 g=torch.Generator().manual_seed(world*1000003+seed*997+start*53)
 base=torch.randn(2,2,generator=g);base=base+torch.eye(2)*1.5
 mats=[]
 for i in range(count):
  if i==0: M=base
  elif i<24:
   theta=(torch.rand((),generator=g)-.5)*2*math.pi;M=rot(theta)@base
  else:
   theta=(torch.rand((),generator=g)-.5)*2*math.pi;delta=torch.randn(2,2,generator=g)*.35;M=rot(theta)@(base+delta)
  mats.append(M)
 x=[];y=[];v=[];vy=[];q=[];qy=[]
 for i,M in enumerate(mats):
  gg=torch.Generator().manual_seed(world*99131+seed*71+start*17+i*23)
  xx=torch.randn(64,2,generator=gg); yy=xx@M.T
  qq=torch.randn(256,2,generator=torch.Generator().manual_seed(world*1217+seed*79+start*29+i*31)); qy.append(qq@M.T)
  x.append(xx[:32]);y.append(yy[:32]);v.append(xx[32:]);vy.append(yy[32:]);q.append(qq)
 return mats,x,y,v,vy,q,qy
def fitmat(x,y):
 return torch.linalg.solve(x.T@x+torch.eye(2)*1e-6,x.T@y).T
def nrmse(pred,y):return float((pred-y).square().mean().sqrt()/(y.square().mean().sqrt()+1e-9))
def evalmat(M,x,y):return nrmse(x@M.T,y)
def choose(mats,x,y,v,vy,mirror,threshold):
 best=(float('inf'),None,None)
 for j,M in enumerate(mats):
  if mirror:
   errs=[]
   for ang in ANGLES:
    eff=rot(ang)@M;errs.append(nrmse(v@eff.T,vy))
   ix=int(torch.tensor(errs).argmin());err=errs[ix];ang=float(ANGLES[ix]);state=(j,ang)
  else:
   err=nrmse(v@M.T,vy);state=(j,0.)
  if err<best[0]:best=(err,state,None)
 return best if best[0]<=threshold else (best[0],None,None)
def run_seq(w,s,threshold,write=False):
 mats,x,y,v,vy,q,qy=seq(w,s);out={};
 for method,mirror in [('path',False),('mirror',True)]:
  bank=[];routes=[];errs=[];births=0;birth_counts=[];route_mac=0;route_mac_prefix=[];query_times=[];start=time.perf_counter()
  for i in range(32):
   if bank:
    err,state,_=choose(bank,x[i],y[i],v[i],vy[i],mirror,threshold);route_mac += len(bank)*(361 if mirror else 1)*32*8
   else:state=None
   if state is None:
    # birth a new frozen private module from support only
    M=fitmat(x[i],y[i]);bank.append(M);births+=1;state=(len(bank)-1,0.)
   j,ang=state;eff=(rot(torch.tensor(ang))@bank[j]) if mirror else bank[j]
   birth_counts.append(births);route_mac_prefix.append(route_mac)
   qb=time.perf_counter();pred=q[i]@eff.T;query_times.append(time.perf_counter()-qb)
   errs.append(nrmse(pred,qy[i]));routes.append((j,ang))
  wall=time.perf_counter()-start
  # Prior-task retention is the final-query score for tasks already passed; modules are immutable.
  retention=statistics.mean(errs[:24])
  out[method]={'bank':bank,'routes':routes,'errors':errs,'birth_counts':birth_counts,'births':births,'route_mac':route_mac,'route_mac_prefix':route_mac_prefix,'query_times':query_times,'wall':wall,'retention':retention}
 pbegin=time.perf_counter();private_errors=[];private_times=[]
 for i in range(32):
  pm=fitmat(x[i],y[i]);qb=time.perf_counter();pred=q[i]@pm.T;private_times.append(time.perf_counter()-qb);private_errors.append(nrmse(pred,qy[i]))
 private_wall=time.perf_counter()-pbegin
 out['private']={'errors':private_errors,'births':32,'wall':private_wall,'route_mac':0,'route_mac_prefix':[0]*32,'query_times':private_times,'retention':None}
 return out

def pack(method,result,N,threshold):
 obj={'format':'ma457-v1','method':method,'N':N,'threshold':threshold}
 r=result[method]
 if method=='private':obj['modules']=[fitmat(*seq_dummy) for seq_dummy in []]
 if method=='private':
  # serialize independently fitted task modules for the prefix
  pass
 else:
  obj['module_bank']=torch.stack(r['bank']);obj['routes']=torch.tensor(r['routes'][:N],dtype=torch.float32)
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def private_package(w,s,N,threshold):
 mats,x,y,v,vy,q,qy=seq(w,s);obj={'format':'ma457-v1','method':'private','N':N,'threshold':threshold,'modules':torch.stack([fitmat(x[i],y[i]) for i in range(N)])};b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def main(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True);worlds=DEV if phase=='development' else FRESH
 if phase=='development':
  scores=[]
  for t in THRESHOLDS:
   birth=[];err=[];size=[]
   for w in worlds:
    for s in SEEDS:
     r=run_seq(w,s,t)
     for m in ('path','mirror'):
      birth.append(r[m]['births']);err.extend(r[m]['errors'])
   scores.append({'threshold':t,'mean_births':statistics.mean(birth),'mean_query_nrmse':statistics.mean(err)})
  eligible=[z for z in scores if z['mean_query_nrmse']<=.03]
  chosen=min(eligible,key=lambda z:z['mean_births'])['threshold'] if eligible else min(scores,key=lambda z:z['mean_query_nrmse'])['threshold']
  (ART/'development_selection.json').write_text(json.dumps({'threshold':chosen,'scores':scores,'fresh_untouched':True},indent=2)+'\n');print(json.dumps({'selected_threshold':chosen,'scores':scores}));return
 selection=json.loads((ART/'development_selection.json').read_text());threshold=selection['threshold'];rows=[]
 for w in worlds:
  for s in SEEDS:
   r=run_seq(w,s,threshold)
   for method in ('path','mirror','private'):
    for N in (1,8,16,32):
     data=private_package(w,s,N,threshold) if method=='private' else pack(method,r,N,threshold)
     path=PAY/f'{w}_{s}_{method}_N{N}.pt';path.write_bytes(data)
     if method=='private': err=statistics.mean(r['private']['errors'][:N]);birth=N;route=0;wall=r['private']['wall'];ret=statistics.mean(r['private']['errors'][:min(N,24)])
     else:err=statistics.mean(r[method]['errors'][:N]);birth=r[method]['birth_counts'][N-1];route=r[method]['route_mac_prefix'][N-1];wall=r[method]['wall'];ret=statistics.mean(r[method]['errors'][:min(N,24)])
     rows.append({'world':w,'seed':s,'method':method,'n':N,'tasks':N,'nrmse_mean':err,'payload_bytes':len(data),'bytes_per_task':len(data)/N,'module_births_cumulative':birth,'route_search_MAC_sequence':route,'query_MAC_per_task':(8 if method!='mirror' else 12),'support_route_wall_seconds':wall,'query_wall_seconds_mean':statistics.mean((r['private']['query_times'] if method=='private' else r[method]['query_times'])[:N]),'prior_task_retention_nrmse':ret,'hash':hashlib.sha256(data).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with open(ART/'fresh_runs.jsonl','w') as f:f.write(''.join(json.dumps(r)+'\n' for r in rows))
 with open(ROOT/'RESULTS_CORE.csv','w',newline='') as f:wri=csv.DictWriter(f,fieldnames=list(rows[0]));wri.writeheader();wri.writerows(rows)
 print(json.dumps({'phase':'fresh','rows':len(rows),'threshold':threshold}))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);main(p.parse_args().phase)
