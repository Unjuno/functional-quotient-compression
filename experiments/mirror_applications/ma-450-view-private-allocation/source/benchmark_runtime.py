#!/usr/bin/env python3
import csv,json,statistics,time
from pathlib import Path
import torch
from run import task,fit_view
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts';sel=json.loads((A/'development_selection.json').read_text());th=sel['threshold'];wg=sel['controller_weight'][0];bi=sel['controller_bias'][0];out=[]
D=16
for w in [45010,45011,45012]:
 for seed in [0,1,2]:
  for method in ['always_mirror','always_private','simple_threshold','learned_controller','oracle']:
   ts=[];mac=[]
   for tid in range(3000,3040):
    kind,z,r,target,x,y=task(w,tid,seed);start=time.perf_counter()
    if method=='always_private': torch.linalg.lstsq(x,y).solution;ops=32*D*D if False else 8192
    elif method=='always_mirror': fit_view(x,y);ops=16*D*2
    elif method=='oracle':
     if kind=='out':torch.linalg.lstsq(x,y).solution;ops=8192
     else:fit_view(x,y);ops=16*D*2
    else:
     code=fit_view(x,y);val=(((x[16:,:2]@code-y[16:]).square().mean().sqrt())/(y[16:].square().mean().sqrt()+1e-9)).item()
     flag=val>th if method=='simple_threshold' else torch.sigmoid(torch.tensor(wg*val+bi)).item()>=.5
     if flag:torch.linalg.lstsq(x,y).solution;ops=8192
     else:ops=16*D*2
    ts.append(time.perf_counter()-start);mac.append(ops)
   out.append({'world':w,'seed':seed,'method':method,'tasks':40,'adaptation_wall_mean_s':statistics.mean(ts),'adaptation_wall_median_s':statistics.median(ts),'adaptation_MAC_mean':statistics.mean(mac)})
with open(A/'runtime.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
