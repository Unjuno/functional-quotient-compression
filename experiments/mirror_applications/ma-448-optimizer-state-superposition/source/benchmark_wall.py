#!/usr/bin/env python3
import csv,time,statistics
from pathlib import Path
import torch
from run import task,continue_task
ROOT=Path(__file__).resolve().parents[1];PAY=ROOT/'artifacts'/'payloads';out=[]
for w in [44830,44831,44832]:
 bundles={m:torch.load(PAY/f'{m}_{w}_N32.pt',weights_only=False) for m in ['exact','shared','pca','mirror']}
 for m,d in bundles.items():
  walls=[];errs=[]
  states=[]
  for key in ['theta','m','v','mean','basis','codes']:
   if key in d:states.append(d[key])
  for i in range(96):
   seed=i//32;tid=2000+i%32;target,x,y,qx,qy=task(w,tid,seed);theta=d['theta'][i]
   if m=='exact':mom,second=d['m'][i],d['v'][i]
   elif m=='shared':mom,second=torch.zeros(16),torch.zeros(16)
   else:
    z=d['mean']+d['codes'][i]@d['basis'].T;mom,second=z[:16],z[16:].clamp_min(0)
   # benchmark continuation compute; quality is checked against saved run aggregates separately
   t=time.perf_counter();continue_task({'theta':theta,'x':x,'y':y,'qx':qx,'qy':qy},mom,second);walls.append(time.perf_counter()-t)
  out.append({'world':w,'method':m,'tasks':96,'continuation_wall_mean_s':statistics.mean(walls),'continuation_wall_median_s':statistics.median(walls),'state_reconstruction_MAC_per_task':(0 if m in ('exact','shared') else 2*32*4)})
with open(ROOT/'artifacts'/'compute_wall.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
for r in out:print(r)
