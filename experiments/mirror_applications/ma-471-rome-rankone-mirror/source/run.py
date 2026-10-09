#!/usr/bin/env python3
import csv, hashlib, io, json, math, statistics, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts'; PAY=ART/'payloads'
DEV=[47100,47101]; FRESH=[47110,47111,47112]; SEEDS=[0,1,2]; D=8; E=64

def make_world(world,seed):
    g=torch.Generator().manual_seed(world*100003+seed*7919+19)
    key_basis=torch.eye(D)[:2].T.contiguous()
    val_basis=torch.eye(D)[4:6].T.contiguous()
    ka=torch.rand(E,generator=g)*2*math.pi
    va=torch.rand(E,generator=torch.Generator().manual_seed(world*997+seed*31+7))*2*math.pi
    keys=torch.stack([ka.cos(),ka.sin()],1)@key_basis.T
    vals=1.25*(torch.stack([va.cos(),va.sin()],1)@val_basis.T)
    return key_basis,val_basis,ka,va,keys,vals

def make_payload(method,kb,vb,ka,va,keys,vals,n):
    obj={'format':'ma471-v1','method':method,'n':n}
    if method=='rome': obj.update(keys=keys[:n].clone(),values=vals[:n].clone())
    elif method=='generic_coeff': obj.update(key_basis=kb,value_basis=vb,key_coeff=torch.stack([ka[:n].cos(),ka[:n].sin()],1),value_coeff=1.25*torch.stack([va[:n].cos(),va[:n].sin()],1))
    elif method=='mirror': obj.update(key_basis=kb,value_basis=vb,key_angle=ka[:n].clone(),value_angle=va[:n].clone(),value_scale=1.25)
    b=io.BytesIO();torch.save(obj,b);return b.getvalue()

def decode(method,payload):
    obj=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
    if method=='rome': return obj['keys'],obj['values']
    if method=='generic_coeff': return obj['key_coeff']@obj['key_basis'].T,obj['value_coeff']@obj['value_basis'].T
    k=torch.stack([obj['key_angle'].cos(),obj['key_angle'].sin()],1)@obj['key_basis'].T
    v=obj['value_scale']*torch.stack([obj['value_angle'].cos(),obj['value_angle'].sin()],1)@obj['value_basis'].T
    return k,v

def metrics(keys,vals,khat,vhat):
    efficacy=[];specificity=[];g=torch.Generator().manual_seed(471991);q=torch.randn((32,D),generator=g)
    for k,v,kk,vv in zip(keys,vals,khat,vhat):
        dh=torch.outer(vv,kk)/(kk@kk)
        efficacy.append(float(((dh@k-v).norm())/(v.norm()+1e-12)))
        x=q-(q@k)[:,None]*k[None,:]/(k@k)
        specificity.append(float((x@dh.T).norm()/(x.norm()*v.norm()+1e-12)))
    return statistics.mean(efficacy),max(efficacy),statistics.mean(specificity),max(specificity)

def run(phase):
    worlds=DEV if phase=='development' else FRESH
    if phase=='development':
        (ART/'development_runs.json').write_text(json.dumps({'worlds':worlds,'seeds':SEEDS,'selection':'fixed exact 2D rotation orbit; no fresh-dependent changes'},indent=2)+'\n');print(json.dumps({'phase':phase,'worlds':len(worlds)*len(SEEDS)}));return
    rows=[]
    for world in worlds:
      for seed in SEEDS:
        kb,vb,ka,va,keys,vals=make_world(world,seed)
        for method in ('rome','generic_coeff','mirror'):
          for n in (1,20,64):
            payload=make_payload(method,kb,vb,ka,va,keys,vals,n);path=PAY/f'{world}_{seed}_{method}_N{n}.pt';path.write_bytes(payload)
            kh,vh=decode(method,payload);m=metrics(keys[:n],vals[:n],kh,vh);samples=[]
            for _ in range(50):
              t=time.perf_counter();make_payload(method,kb,vb,ka,va,keys,vals,n);samples.append(time.perf_counter()-t)
            q=torch.randn(D,generator=torch.Generator().manual_seed(world+seed*101+n))
            t0=time.perf_counter()
            for _ in range(100):
              for kk,vv in zip(kh,vh):
                _=torch.outer(vv,kk)/(kk@kk)@q
            query_seconds=(time.perf_counter()-t0)/(100*n)
            rows.append({'world':world,'seed':seed,'method':method,'n':n,'edit_efficacy_nrmse_mean':m[0],'edit_efficacy_nrmse_max':m[1],'specificity_drift_mean':m[2],'specificity_drift_max':m[3],'payload_bytes':len(payload),'bytes_per_edit':len(payload)/n,'factor_decode_MAC_per_fact':32 if method=='rome' else 24 if method=='generic_coeff' else 28,'rank_one_apply_MAC_per_query':2*D*D,'payload_encode_seconds_per_fact':statistics.median(samples)/n,'query_apply_seconds_per_edit':query_seconds,'hash':hashlib.sha256(payload).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
