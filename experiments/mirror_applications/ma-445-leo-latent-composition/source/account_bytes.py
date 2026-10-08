#!/usr/bin/env python3
"""A3 exact serialization accounting for shared skill banks plus task addresses."""
import csv, hashlib, io, json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; PAY=ROOT/'artifacts'/'payloads'; OUT=ROOT/'artifacts'
rows=[json.loads(x) for x in open(OUT/'fresh_runs.jsonl')]
methods=['taskvec','mirror','leo']; result=[]
def dump(obj):
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()
for w in [44530,44531,44532]:
 for seed in [0,1,2]:
  # Get shared state once from the first held-out pair, replicate and step.
  for method in methods:
   src=PAY/f'fresh_{w}_{seed}_0-1_r0_{method}_0.pt'; d=torch.load(src,weights_only=False)
   common={k:v for k,v in d.items() if k not in ('pair','format')}
   common['format']='ma445-shared-v1'; bank=dump(common)
   bankpath=PAY/f'shared_{w}_{seed}_{method}_a3.pt';bankpath.write_bytes(bank)
   for pair in [(0,1),(0,4),(1,3),(2,5),(3,4),(4,5)]:
    view=dump({'format':'ma445-view-v1','pair':pair})
    viewpath=PAY/f'view_{w}_{seed}_{pair[0]}-{pair[1]}_{method}_a3.pt';viewpath.write_bytes(view)
    for n in [1,20,100]:
     result.append({'world':w,'seed':seed,'pair':f'{pair[0]}-{pair[1]}','method':method,'shared_bytes':len(bank),'view_bytes':len(view),'N':n,'amortized_bytes':len(bank)+n*len(view),'shared_sha256':hashlib.sha256(bank).hexdigest(),'view_sha256':hashlib.sha256(view).hexdigest(),'shared_path':str(bankpath.relative_to(ROOT.parents[2])),'view_path':str(viewpath.relative_to(ROOT.parents[2]))})
with open(OUT/'serialized_accounting.csv','w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=result[0]);wr.writeheader();wr.writerows(result)
for method in methods:
 for n in [1,20,100]:
  a=[r['amortized_bytes'] for r in result if r['method']==method and r['N']==n]
  print(method,n,round(sum(a)/len(a),1))
