#!/usr/bin/env python3
import csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads'
DEV=[48100,48101];FRESH=[48110,48111,48112];SEEDS=[0,1,2];D=32;V=4;MAX_E=128;SIGMA=.01;RADIUS=.95;KS=(8,16,32,64,128)

def make_world(world,seed):
    basis,_=torch.linalg.qr(torch.randn(D,2,generator=torch.Generator().manual_seed(world*31337+seed*101+5)))
    i=torch.arange(MAX_E,dtype=torch.float32);jitter=(torch.rand(MAX_E,generator=torch.Generator().manual_seed(world*1117+seed*53+7))-.5)*.006
    angle=2*math.pi*i/MAX_E+jitter
    coeff=torch.stack([angle.cos(),angle.sin()],1);keys=coeff@basis.T
    values=torch.randn(MAX_E,V,generator=torch.Generator().manual_seed(world*997+seed*31+9))
    radii=torch.full((MAX_E,),RADIUS)
    return basis,angle,keys,values,radii

def make_queries(world,seed,keys):
    para=keys[:,None,:]+SIGMA*torch.randn(MAX_E,4,D,generator=torch.Generator().manual_seed(world*8713+seed*43+19))
    para=para/para.norm(dim=-1,keepdim=True)
    neg=torch.randn(512,D,generator=torch.Generator().manual_seed(world*1811+seed*97+23));neg=neg/neg.norm(dim=-1,keepdim=True)
    return keys.clone(),para.reshape(MAX_E*4,D),neg

def payload(method,basis,angle,keys,values,radii,n):
    if method=='no_edit':return b''
    o={'format':'ma481-v1','method':method,'n':n,'fact_ids':torch.arange(n,dtype=torch.int32),'radii':radii[:n].clone(),'values':values[:n].clone()}
    if method=='explicit':o['keys']=keys[:n].clone()
    elif method=='generic_coeff':o.update(key_basis=basis.clone(),key_coeff=torch.stack([angle[:n].cos(),angle[:n].sin()],1))
    elif method=='mirror':o.update(key_basis=basis.clone(),key_angle=angle[:n].clone())
    elif method.startswith('vq'):
        k=int(method[2:]);centers=torch.arange(k,dtype=torch.float32)*(2*math.pi/k);cb=torch.stack([centers.cos(),centers.sin()],1)@basis.T
        idx=torch.remainder(torch.round(angle[:n]*k/(2*math.pi)).long(),k).to(torch.uint8)
        o.update(codebook_keys=cb,key_indices=idx,codebook_size=k)
    b=io.BytesIO();torch.save(o,b);return b.getvalue()

def load_mem(p):return None if not p else torch.load(io.BytesIO(p),map_location='cpu',weights_only=False)

def decode_keys(method,o):
    if method=='no_edit':return None
    if method=='explicit':return o['keys']
    if method=='generic_coeff':return o['key_coeff']@o['key_basis'].T
    if method=='mirror':
        a=o['key_angle'];return torch.stack([a.cos(),a.sin()],1)@o['key_basis'].T
    return o['codebook_keys'][o['key_indices'].long()]

def retrieve(query,method,o,decoded):
    if o is None:return torch.zeros(len(query),V),torch.zeros(len(query),dtype=torch.bool),torch.full((len(query),),-1,dtype=torch.long)
    sims=query@decoded.T;best,idx=sims.max(1);hit=best>=o['radii'][idx]
    out=o['values'][idx].clone();out[~hit]=0
    return out,hit,idx

def metrics(method,o,decoded,qe,qp,qn,targets,n):
    ex,he,ie=retrieve(qe[:n],method,o,decoded);pa,hp,ip=retrieve(qp[:n*4],method,o,decoded);neg,hn,_=retrieve(qn,method,o,decoded)
    exok=he&(ie==torch.arange(n));pa_expected=torch.arange(n).repeat_interleave(4);paok=hp&(ip==pa_expected)
    ee=float((ex-targets[:n]).norm()/(targets[:n].norm()+1e-12));pt=targets[:n].repeat_interleave(4,0);pe=float((pa-pt).norm()/(pt.norm()+1e-12));dr=float(neg.norm()/(len(neg)**.5*(targets[:n].norm(dim=1).mean()+1e-12)))
    return float(exok.float().mean()),float(paok.float().mean()),float(hn.float().mean()),ee,pe,dr

def run(phase):
    worlds=DEV if phase=='development' else FRESH
    if phase=='development':
        (ART/'development_runs.json').write_text(json.dumps({'worlds':worlds,'seeds':SEEDS,'fixed_scope_threshold':RADIUS,'fixed_vq_K_sweep':list(KS),'address_jitter_radians':.003},indent=2)+'\n');print(json.dumps({'phase':phase,'worlds':len(worlds)*len(SEEDS)}));return
    rows=[]
    for w in worlds:
      for s in SEEDS:
        basis,angle,keys,values,radii=make_world(w,s);qe,qp,qn=make_queries(w,s,keys)
        methods=['no_edit','explicit','generic_coeff','mirror']+[f'vq{k}' for k in KS]
        for m in methods:
          for n in (8,32,64,128):
            blob=payload(m,basis,angle,keys,values,radii,n);o=load_mem(blob);decoded=decode_keys(m,o)
            a,b,c,ee,pe,dr=metrics(m,o,decoded,qe,qp,qn,values,n)
            dect=[]
            for _ in range(30):
                t=time.perf_counter();decode_keys(m,o);dect.append(time.perf_counter()-t)
            qlat=[]
            for _ in range(40):
                t=time.perf_counter();retrieve(qe[:n],m,o,decoded);qlat.append((time.perf_counter()-t)/n)
            unique=0 if decoded is None else int(torch.unique(decoded.round(decimals=6),dim=0).shape[0])
            path=PAY/f'{w}_{s}_{m}_N{n}.pt'
            if blob:path.write_bytes(blob)
            rows.append({'world':w,'seed':s,'method':m,'n':n,'exact_correct_address_recall':a,'paraphrase_correct_address_recall':b,'unrelated_false_trigger_rate':c,'exact_output_nrmse':ee,'paraphrase_output_nrmse':pe,'unrelated_output_drift':dr,'unique_decoded_addresses':unique,'address_collision_fraction':0 if n==0 else 1-unique/n,'payload_bytes':len(blob),'bytes_per_address':len(blob)/n,'lookup_MAC_per_query':2*n*D,'address_decode_MAC_per_key':0 if m in ('no_edit','explicit') or m.startswith('vq') else (2*D if m=='generic_coeff' else 3*D),'bank_decode_seconds_per_address':statistics.median(dect)/n,'lookup_seconds_per_query':statistics.median(qlat),'hash':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2])) if blob else 'empty'})
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
