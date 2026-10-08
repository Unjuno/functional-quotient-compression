#!/usr/bin/env python3
"""A1: charge the complete shared conditioned-policy basis."""
import json,hashlib,io
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';sch=torch.load(PAY/'schedules.pt',weights_only=True)
def invsoftplus(x):return torch.log(torch.expm1(x.clamp_min(1e-8)))
r0=invsoftplus(sch['conditioned'][0]);r1=invsoftplus(sch['conditioned'][1]);basis={'base':(r0+r1)/2,'modulation':(r1-r0)/2}
rows=[json.loads(x) for x in open(ART/'fresh_runs.jsonl')]
for r in rows:
 if r['method']!='conditioned':continue
 p=ROOT.parents[2]/r['path'];d=torch.load(p,weights_only=False);d['policy_basis']=basis;d['policy']=None;d['config']['domain_code']=r['domain'];b=io.BytesIO();torch.save(d,b);data=b.getvalue();p.write_bytes(data);r['payload_bytes']=len(data);r['bytes_per_task']=len(data)/10;r['hash']=hashlib.sha256(data).hexdigest()
(ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('updated conditioned payload count',sum(r['method']=='conditioned' for r in rows))
