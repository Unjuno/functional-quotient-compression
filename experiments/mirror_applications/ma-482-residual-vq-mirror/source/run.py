#!/usr/bin/env python3
import csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads'
DEV=[48200,48201];FRESH=[48220,48221,48222];SEEDS=[0,1,2];D=16;N=128;STAGES=(1,2,4);KS=(4,8,16,32);AMPS=torch.tensor([1.,.5,.25,.125])

def make_world(w,s):
    g=torch.Generator().manual_seed(w*100003+s*7919+41);theta=torch.rand(N,4,generator=g)*2*math.pi
    y=torch.zeros(N,D)
    for j,a in enumerate(AMPS):
        y[:,2*j]=a*theta[:,j].cos();y[:,2*j+1]=a*theta[:,j].sin()
    return theta,y

def collect_dev():
    return torch.cat([make_world(w,s)[1] for w in DEV for s in SEEDS],0)

def kmeans(x,k,seed,steps=40):
    g=torch.Generator().manual_seed(seed);idx=torch.randperm(len(x),generator=g)[:k];c=x[idx].clone()
    for _ in range(steps):
        d=torch.cdist(x,c);a=d.argmin(1);new=c.clone()
        for j in range(k):
            mask=a==j
            if mask.any():new[j]=x[mask].mean(0)
        if torch.allclose(new,c,atol=1e-7,rtol=0):c=new;break
        c=new
    return c

def fit_codebooks():
    x=collect_dev();out={}
    for k in KS:
      for r in STAGES:
        residual=x.clone();books=[]
        for j in range(r):
            cb=kmeans(residual,k,seed=482001+k*101+r*17+j);books.append(cb)
            residual=residual-cb[torch.cdist(residual,cb).argmin(1)]
        out[f'{k}_{r}']=torch.stack(books)
    torch.save(out,ART/'development_codebooks.pt');return out

def mirror_codes(y,k,r):
    # Infer phase only from the candidate target residual components; do not
    # read the synthetic teacher's generating angle.
    phases=torch.stack([torch.atan2(y[:,2*j+1],y[:,2*j]) for j in range(r)],1)
    return torch.remainder(torch.round(phases*k/(2*math.pi)).long(),k).to(torch.uint8)

def native_codes(y,books):
    residual=y.clone();codes=[]
    for cb in books:
        idx=torch.cdist(residual,cb).argmin(1);codes.append(idx.to(torch.uint8));residual-=cb[idx]
    return torch.stack(codes,1)

def make_payload(method,y,theta,k,r,books):
    if method=='dense':o={'format':'ma482-v1','method':method,'targets':y.clone()}
    elif method=='native_rvq':o={'format':'ma482-v1','method':method,'codebooks':books.clone(),'codes':native_codes(y,books),'K':k,'R':r,'N':len(y)}
    elif method=='generic_coeff_rvq':
        codes=mirror_codes(y,k,r);phases=torch.arange(k,dtype=torch.float32)*(2*math.pi/k)
        table=torch.stack([phases.cos(),phases.sin()],1)
        o={'format':'ma482-v1','method':method,'basis':torch.eye(D)[:,:2].clone(),'plane_ids':torch.arange(r,dtype=torch.int8),'scales':AMPS[:r].clone(),'coefficient_codebooks':table.clone(),'codes':codes,'K':k,'R':r,'N':len(y)}
    else:
        o={'format':'ma482-v1','method':'mirror_rvq','basis':torch.eye(D)[:,:2].clone(),'plane_ids':torch.arange(r,dtype=torch.int8),'scales':AMPS[:r].clone(),'codes':mirror_codes(y,k,r),'K':k,'R':r,'N':len(y)}
    b=io.BytesIO();torch.save(o,b);return b.getvalue()

def decode_obj(o,ablate=None):
    method=o['method'];n=o['N'] if method!='dense' else len(o['targets'])
    if method=='dense':return o['targets'].clone(),[]
    r=o['R'];stages=[]
    if method=='native_rvq':
        for j in range(r):stages.append(o['codebooks'][j][o['codes'][:,j].long()])
    elif method=='generic_coeff_rvq':
        t=o['coefficient_codebooks'][o['codes'].long()]
        for j in range(r):
            v=torch.zeros(n,D);a=o['scales'][j]*t[:,j,0];b=o['scales'][j]*t[:,j,1]
            v[:,2*j]=a;v[:,2*j+1]=b;stages.append(v)
    else:
        idx=o['codes'].long();phase=idx*(2*math.pi/o['K']);
        for j in range(r):
            v=torch.zeros(n,D);v[:,2*j]=o['scales'][j]*phase[:,j].cos();v[:,2*j+1]=o['scales'][j]*phase[:,j].sin();stages.append(v)
    y=torch.zeros(n,D)
    for j,v in enumerate(stages):
        if j!=ablate:y+=v
    return y,stages

def evaluate(y,o):
    recon,stages=decode_obj(o);err=float((recon-y).norm()/(y.norm()+1e-12));abl=[]
    for j in range(len(stages)):
        ar,_=decode_obj(o,ablate=j);abl.append(float((ar-y).norm()/(y.norm()+1e-12)))
    used=[]
    if o['method']!='dense':
        for j in range(o['R']):used.append(int(torch.unique(o['codes'][:,j]).numel()))
    return err,abl,used

def run(phase):
    if phase=='development':
        books=fit_codebooks();print(json.dumps({'phase':'development','codebook_sets':len(books),'fit_examples':len(collect_dev())}));return
    books=torch.load(ART/'development_codebooks.pt',map_location='cpu',weights_only=False);rows=[]
    for w in FRESH:
      for s in SEEDS:
        theta,y=make_world(w,s)
        for k in KS:
          for r in STAGES:
            bks=books[f'{k}_{r}']
            for method in ('dense','native_rvq','generic_coeff_rvq','mirror_rvq'):
              blob=make_payload(method,y,theta,k,r,bks);o=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
              err,abl,used=evaluate(y,o);encode=[]
              for _ in range(15):
                t=time.perf_counter();make_payload(method,y,theta,k,r,bks);encode.append(time.perf_counter()-t)
              decode=[]
              for _ in range(30):
                t=time.perf_counter();decode_obj(o);decode.append(time.perf_counter()-t)
              path=PAY/f'{w}_{s}_{method}_K{k}_R{r}.pt';path.write_bytes(blob)
              rows.append({'world':w,'seed':s,'method':method,'K':k,'stages':r,'normalized_rmse':err,'stage_ablation_rmse':json.dumps(abl),'used_codes_per_stage':json.dumps(used),'payload_bytes':len(blob),'bytes_per_function':len(blob)/N,'assignment_MAC_per_function':0 if method in ('dense','mirror_rvq','generic_coeff_rvq') else r*k*D,'phase_encode_ops_per_function':r if method in ('mirror_rvq','generic_coeff_rvq') else 0,'decode_MAC_per_function':0 if method=='dense' else r*D*2,'encode_seconds_per_function':statistics.median(encode)/N,'decode_seconds_per_function':statistics.median(decode)/N,'hash':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['development','fresh'],required=True);run(a.parse_args().phase)
