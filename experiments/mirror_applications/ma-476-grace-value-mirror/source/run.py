#!/usr/bin/env python3
import csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads'
DEV=[47600,47601];FRESH=[47610,47611,47612];SEEDS=[0,1,2];D=16;E=64;SIGMA=.03;SCALE=1.2;KS=(8,16,32,64)

def make_world(world,seed):
    g=torch.Generator().manual_seed(world*100003+seed*7919+61)
    keys=torch.randn(E,D,generator=g);keys=keys/keys.norm(dim=1,keepdim=True)
    radii=.87+.06*torch.rand(E,generator=torch.Generator().manual_seed(world*2777+seed*43+8))
    basis,_=torch.linalg.qr(torch.randn(D,2,generator=torch.Generator().manual_seed(world*31337+seed*101+5)))
    ang=torch.rand(E,generator=torch.Generator().manual_seed(world*997+seed*31+9))*2*math.pi
    values=SCALE*torch.stack([ang.cos(),ang.sin()],1)@basis.T
    return keys,radii,basis,ang,values

def make_queries(world,seed,keys):
    noise=torch.randn(E,4,D,generator=torch.Generator().manual_seed(world*8713+seed*43+19))*SIGMA
    para=keys[:,None,:]+noise;para=para/para.norm(dim=-1,keepdim=True)
    neg=torch.randn(256,D,generator=torch.Generator().manual_seed(world*1811+seed*97+23));neg=neg/neg.norm(dim=-1,keepdim=True)
    return keys.clone(),para.reshape(-1,D),neg

def payload(method,keys,radii,basis,ang,values,n,k=0):
    if method=='no_edit':return b''
    o={'format':'ma476-v1','method':method,'n':n,'fact_ids':torch.arange(n,dtype=torch.int32),'keys':keys[:n].clone(),'radii':radii[:n].clone()}
    if method=='grace_explicit':o['values']=values[:n].clone()
    elif method=='generic_coeff':o.update(value_basis=basis.clone(),value_coeff=torch.stack([ang[:n].cos(),ang[:n].sin()],1)*SCALE)
    elif method=='mirror':o.update(value_basis=basis.clone(),value_angle=ang[:n].clone(),value_scale=SCALE)
    elif method.startswith('vq'):
        k=int(method[2:]);centers=torch.arange(k,dtype=torch.float32)*(2*math.pi/k)
        cb=SCALE*torch.stack([centers.cos(),centers.sin()],1)@basis.T
        idx=torch.remainder(torch.round(ang[:n]*k/(2*math.pi)).to(torch.int64),k).to(torch.uint8)
        o.update(codebook_values=cb,code_indices=idx,codebook_size=k)
    b=io.BytesIO();torch.save(o,b);return b.getvalue()

def load_memory(method,p):
    if method=='no_edit':return None
    return torch.load(io.BytesIO(p),map_location='cpu',weights_only=False)

def selected_values(method,o,idx):
    if method=='grace_explicit':return o['values'][idx]
    if method=='generic_coeff':return o['value_coeff'][idx]@o['value_basis'].T
    if method=='mirror':
        a=o['value_angle'][idx];return o['value_scale']*torch.stack([a.cos(),a.sin()],1)@o['value_basis'].T
    if method.startswith('vq'):return o['codebook_values'][o['code_indices'][idx].long()]
    return torch.zeros(len(idx),D)

def retrieve(queries,method,o):
    if o is None:return torch.zeros(len(queries),D),torch.zeros(len(queries),dtype=torch.bool)
    sim=queries@o['keys'].T;best,idx=sim.max(1);hit=best>=o['radii'][idx]
    out=selected_values(method,o,idx);out[~hit]=0
    return out,hit

def evaluate(method,o,qe,qp,qn,values):
    ex,he=retrieve(qe,method,o);pa,hp=retrieve(qp,method,o);ng,hn=retrieve(qn,method,o)
    exact_err=float((ex-values[:len(qe)]).norm()/(values[:len(qe)].norm()+1e-12))
    target=values.repeat_interleave(4,0)[:len(qp)]
    para_err=float((pa-target).norm()/(target.norm()+1e-12))
    drift=float(ng.norm()/(len(ng)**.5*SCALE+1e-12))
    return float(he.float().mean()),float(hp.float().mean()),float(hn.float().mean()),exact_err,para_err,drift

def run(phase):
    worlds=DEV if phase=='development' else FRESH
    if phase=='development':
        (ART/'development_runs.json').write_text(json.dumps({'worlds':worlds,'seeds':SEEDS,'K_sweep':list(KS),'fixed_radius_rule':'per-entry radius Uniform[0.87,0.93]'},indent=2)+'\n');print(json.dumps({'phase':phase,'worlds':len(worlds)*len(SEEDS)}));return
    rows=[]
    for w in worlds:
      for s in SEEDS:
        keys,radii,basis,ang,values=make_world(w,s);qe,qp,qn=make_queries(w,s,keys)
        methods=['no_edit','grace_explicit','generic_coeff','mirror']+[f'vq{k}' for k in KS]
        for m in methods:
          for n in (1,8,32,64):
            k=int(m[2:]) if m.startswith('vq') else 0
            blob=payload(m,keys,radii,basis,ang,values,n,k);o=load_memory(m,blob)
            ex,pa,fp,ee,pe,dr=evaluate(m,o,qe[:n],qp[:n*4],qn,values[:n])
            et=[]
            for _ in range(30):
              t=time.perf_counter();payload(m,keys,radii,basis,ang,values,n,k);et.append(time.perf_counter()-t)
            lt=[]
            for _ in range(40):
              t=time.perf_counter();retrieve(qe[:n],m,o);lt.append((time.perf_counter()-t)/n)
            path=PAY/f'{w}_{s}_{m}_N{n}.pt'
            if blob:path.write_bytes(blob)
            vd=0 if m in ('no_edit','grace_explicit') or m.startswith('vq') else 4*D
            rows.append({'world':w,'seed':s,'method':m,'n':n,'exact_hit_recall':ex,'paraphrase_hit_recall':pa,'unrelated_false_trigger_rate':fp,'exact_value_nrmse':ee,'paraphrase_value_nrmse':pe,'unrelated_output_drift':dr,'payload_bytes':len(blob),'bytes_per_edit':len(blob)/n,'lookup_MAC_per_query':2*n*D,'value_decode_MAC_per_hit':vd,'payload_encode_seconds_per_edit':statistics.median(et)/n,'lookup_and_decode_seconds_per_query':statistics.median(lt),'hash':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2])) if blob else 'empty'})
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
