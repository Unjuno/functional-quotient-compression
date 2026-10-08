#!/usr/bin/env python3
"""Exact whole-bank serializers for N task codes across both domains."""
import csv,hashlib,io,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';s=torch.load(PAY/'schedules.pt',weights_only=True);out=[]
def dump(x):b=io.BytesIO();torch.save(x,b);return b.getvalue()
for w in [44710,44711,44712]:
 for seed in [0,1,2]:
  codes={};
  for d in [0,1]:
   for m in ['adam','unconditioned','conditioned','separate']:
    f=PAY/f'{w}_{seed}_{d}_{m}_4.pt';codes[(d,m)]=torch.load(f,weights_only=False)['task_codes']
  for n in [1,20,100]:
   for method in ['adam','unconditioned','conditioned','separate']:
    counts=[(n+1)//2,n//2];all_codes=torch.cat([torch.cat([codes[(d,method)] for _ in range((count+9)//10)],0)[:count] for d,count in enumerate(counts) if count],0)
    if method=='adam':policy={'lr':.1}
    elif method=='unconditioned':policy={'schedule':s['unconditioned'][0]}
    elif method=='conditioned':
     r0=torch.log(torch.expm1(s['conditioned'][0].clamp_min(1e-8)));r1=torch.log(torch.expm1(s['conditioned'][1].clamp_min(1e-8)));policy={'base':(r0+r1)/2,'modulation':(r1-r0)/2,'domain_codes':torch.tensor([-1,1])}
    else:policy={'domain0_schedule':s['separate'][0],'domain1_schedule':s['separate'][1]}
    obj={'format':'ma447-frontier-v1','method':method,'policy':policy,'domain_task_codes':all_codes,'N':n,'domain_code_per_task':torch.tensor([0,1]).repeat_interleave(torch.tensor(counts))}
    data=dump(obj);path=PAY/f'frontier_{w}_{seed}_{method}_N{n}.pt';path.write_bytes(data)
    out.append({'world':w,'seed':seed,'method':method,'N':n,'serialized_bytes':len(data),'bytes_per_task':len(data)/n,'sha256':hashlib.sha256(data).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
with open(ART/'byte_frontier.csv','w',newline='') as f:wri=csv.DictWriter(f,fieldnames=out[0]);wri.writeheader();wri.writerows(out)
for m in ['adam','unconditioned','conditioned','separate']:
 print(m,[(n,round(sum(r['bytes_per_task'] for r in out if r['method']==m and r['N']==n)/9,2)) for n in [1,20,100]])
