#!/usr/bin/env python3
import csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads'
DEV=[47500,47501];FRESH=[47510,47511,47512];SEEDS=[0,1,2];D=16;E=64;SIGMA=.03;THRESH=.90;SCALE=1.2

def make_world(world,seed):
    g=torch.Generator().manual_seed(world*100003+seed*7919+47)
    keys=torch.randn(E,D,generator=g);keys=keys/keys.norm(dim=1,keepdim=True)
    q,_=torch.linalg.qr(torch.randn(D,2,generator=torch.Generator().manual_seed(world*31337+seed*101+5)))
    angles=torch.rand(E,generator=torch.Generator().manual_seed(world*997+seed*31+9))*2*math.pi
    values=SCALE*(torch.stack([angles.cos(),angles.sin()],1)@q.T)
    return keys,q,angles,values

def make_queries(world,seed,keys,values):
    g=torch.Generator().manual_seed(world*8713+seed*43+19)
    noise=torch.randn(E,4,D,generator=g)*SIGMA
    para=keys[:,None,:]+noise;para=para/para.norm(dim=-1,keepdim=True)
    neg=torch.randn(256,D,generator=torch.Generator().manual_seed(world*1811+seed*97+23));neg=neg/neg.norm(dim=-1,keepdim=True)
    return keys.clone(),para.reshape(-1,D),neg

def make_payload(method,keys,basis,angles,values,n):
    if method=='no_edit':return b''
    o={'format':'ma475-v1','method':method,'n':n,'fact_ids':torch.arange(n,dtype=torch.int32),'scope_threshold':THRESH}
    if method=='serac':o.update(keys=keys[:n].clone(),values=values[:n].clone())
    elif method=='generic_coeff':o.update(keys=keys[:n].clone(),value_basis=basis.clone(),value_coeff=torch.stack([angles[:n].cos(),angles[:n].sin()],1)*SCALE)
    elif method=='mirror':o.update(keys=keys[:n].clone(),value_basis=basis.clone(),value_angle=angles[:n].clone(),value_scale=SCALE)
    b=io.BytesIO();torch.save(o,b);return b.getvalue()

def decode(method,p):
    if method=='no_edit':return None,None,None
    o=torch.load(io.BytesIO(p),map_location='cpu',weights_only=False)
    if method=='serac':return o['keys'],o['values'],o['scope_threshold']
    if method=='generic_coeff':return o['keys'],o['value_coeff']@o['value_basis'].T,o['scope_threshold']
    vals=SCALE*torch.stack([o['value_angle'].cos(),o['value_angle'].sin()],1)@o['value_basis'].T
    return o['keys'],vals,o['scope_threshold']

def retrieve(query,keys,values,threshold):
    if keys is None:return torch.zeros(len(query),D),torch.zeros(len(query),dtype=torch.bool)
    sim=query@keys.T;best,idx=sim.max(dim=1);hit=best>=threshold;out=values[idx].clone();out[~hit]=0
    return out,hit

def evaluate(keys,values,qexact,qpara,qneg,khat,vhat,threshold):
    if khat is None:
        z=torch.zeros(len(qexact),D);p=torch.zeros(len(qpara),D);n=torch.zeros(len(qneg),D)
        _,h1=retrieve(qexact,None,None,None);_,h2=retrieve(qpara,None,None,None);_,h3=retrieve(qneg,None,None,None)
    else:
        z,h1=retrieve(qexact,khat,vhat,threshold);p,h2=retrieve(qpara,khat,vhat,threshold);n,h3=retrieve(qneg,khat,vhat,threshold)
    exact_target=values[:len(qexact)];para_target=values.repeat_interleave(4,dim=0)
    exact_err=float((z-exact_target).norm()/(exact_target.norm()+1e-12));para_err=float((p-para_target).norm()/(para_target.norm()+1e-12))
    neg_drift=float(n.norm()/(len(n)**.5*SCALE+1e-12))
    return float(h1.float().mean()),float(h2.float().mean()),float(h3.float().mean()),exact_err,para_err,neg_drift

def run(phase):
    worlds=DEV if phase=='development' else FRESH
    if phase=='development':
        (ART/'development_runs.json').write_text(json.dumps({'worlds':worlds,'seeds':SEEDS,'fixed_threshold':THRESH,'value_orbit':'fixed norm 1.2 in one shared 2D plane'},indent=2)+'\n');print(json.dumps({'phase':phase,'worlds':len(worlds)*len(SEEDS)}));return
    rows=[]
    for w in worlds:
      for s in SEEDS:
        keys,basis,ang,values=make_world(w,s);qe,qp,qn=make_queries(w,s,keys,values)
        for method in ('no_edit','serac','generic_coeff','mirror'):
          for n in (1,8,32,64):
            p=make_payload(method,keys,basis,ang,values,n);kh,vh,thr=decode(method,p)
            ex,pr,fp,ee,pe,nd=evaluate(keys[:n],values[:n],qe[:n],qp[:n*4],qn,kh,vh,thr)
            times=[]
            for _ in range(30):
                t=time.perf_counter();make_payload(method,keys,basis,ang,values,n);times.append(time.perf_counter()-t)
            # Repeated isolated probes reduce timer noise; median remains a
            # microbenchmark, not an end-to-end SERAC serving claim.
            lookup_samples=[]
            for _ in range(40):
                t=time.perf_counter()
                if kh is None:retrieve(qe[:n],None,None,None)
                else:retrieve(qe[:n],kh,vh,thr)
                lookup_samples.append((time.perf_counter()-t)/n)
            lookup=statistics.median(lookup_samples)
            path=PAY/f'{w}_{s}_{method}_N{n}.pt'
            if p:path.write_bytes(p)
            rows.append({'world':w,'seed':s,'method':method,'n':n,'exact_hit_recall':ex,'paraphrase_hit_recall':pr,'unrelated_false_trigger_rate':fp,'exact_value_nrmse':ee,'paraphrase_value_nrmse':pe,'unrelated_output_drift':nd,'payload_bytes':len(p),'bytes_per_edit':len(p)/n,'lookup_MAC_per_query':2*n*D,'value_decode_MAC_per_hit':0 if method in ('no_edit','serac') else 3*D,'payload_encode_seconds_per_edit':statistics.median(times)/n,'exact_lookup_seconds_per_query':lookup,'hash':hashlib.sha256(p).hexdigest(),'path':str(path.relative_to(ROOT.parents[2])) if p else 'empty'})
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
      wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['development','fresh'],required=True);run(a.parse_args().phase)
