#!/usr/bin/env python3
import csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads'
DEV=[47800,47801];FRESH=[47810,47811,47812];SEEDS=[0,1,2];D=16;E=64;SIGMA=.03;SCALE=1.2;ALPHA=.35;TAUS=(.05,.1,.2,.3)

def make_world(world,seed):
    keys=torch.randn(E,D,generator=torch.Generator().manual_seed(world*100003+seed*7919+1));keys=keys/keys.norm(dim=1,keepdim=True)
    radii=.87+.06*torch.rand(E,generator=torch.Generator().manual_seed(world*2777+seed*43+8))
    basis,_=torch.linalg.qr(torch.randn(D,2,generator=torch.Generator().manual_seed(world*31337+seed*101+5)))
    ang=torch.rand(E,generator=torch.Generator().manual_seed(world*997+seed*31+9))*2*math.pi
    orbit=torch.stack([ang.cos(),ang.sin()],1)@basis.T
    perm=torch.randperm(E,generator=torch.Generator().manual_seed(world*1811+seed*97+23));aligned=torch.zeros(E,dtype=torch.bool);aligned[perm[:round(.70*E)]]=True
    off=torch.randn(E,D,generator=torch.Generator().manual_seed(world*7817+seed*59+11));off=off-(off@basis)@basis.T;off=off/off.norm(dim=1,keepdim=True)
    vals=torch.where(aligned[:,None],orbit,math.sqrt(1-ALPHA**2)*orbit+ALPHA*off)*SCALE
    return keys,radii,basis,ang,vals,aligned

def make_queries(world,seed,keys):
    noise=torch.randn(E,4,D,generator=torch.Generator().manual_seed(world*8713+seed*43+19))*SIGMA
    para=keys[:,None,:]+noise;para=para/para.norm(dim=-1,keepdim=True)
    neg=torch.randn(256,D,generator=torch.Generator().manual_seed(world*3137+seed*29+13));neg=neg/neg.norm(dim=-1,keepdim=True)
    return keys.clone(),para.reshape(-1,D),neg

def projection(values,basis):
    coeff=values@basis
    recon=coeff@basis.T
    residual=(values-recon).norm(dim=1)/(values.norm(dim=1)+1e-12)
    return coeff,recon,residual

def make_payload(method,keys,radii,basis,values,aligned,n,tau):
    if method=='no_edit':return b''
    o={'format':'ma478-v1','method':method,'n':n,'fact_ids':torch.arange(n,dtype=torch.int32),'keys':keys[:n].clone(),'radii':radii[:n].clone()}
    coeff,recon,resid=projection(values[:n],basis)
    if method=='explicit':o['values']=values[:n].clone()
    elif method=='mirror_only':
        o.update(value_basis=basis.clone(),angles=torch.atan2(coeff[:,1],coeff[:,0]),value_scale=SCALE)
    elif method=='generic_only':o.update(value_basis=basis.clone(),coefficients=coeff.clone())
    elif method in ('mirror_fallback','generic_fallback','oracle_fallback'):
        private=(~aligned[:n]) if method=='oracle_fallback' else resid>tau
        flags=private.to(torch.uint8)
        common={'value_basis':basis.clone(),'private_flags':flags.clone(),'private_indices':torch.where(private)[0].to(torch.int32),'private_values':values[:n][private].clone(),'fallback_threshold':tau}
        if method=='generic_fallback':common['coefficients']=coeff.clone()
        else:common.update(angles=torch.atan2(coeff[:,1],coeff[:,0]),value_scale=SCALE)
        o.update(common)
    b=io.BytesIO();torch.save(o,b);return b.getvalue()

def load_memory(p):return None if not p else torch.load(io.BytesIO(p),map_location='cpu',weights_only=False)

def select_values(method,o,idx):
    if method=='explicit':return o['values'][idx]
    if method=='mirror_only':
        a=o['angles'][idx];return o['value_scale']*torch.stack([a.cos(),a.sin()],1)@o['value_basis'].T
    if method=='generic_only':return o['coefficients'][idx]@o['value_basis'].T
    if method in ('mirror_fallback','generic_fallback','oracle_fallback'):
        flags=o['private_flags'][idx].bool();out=torch.zeros(len(idx),D)
        if (~flags).any():
            ix=idx[~flags]
            if method=='generic_fallback':out[~flags]=o['coefficients'][ix]@o['value_basis'].T
            else:
                a=o['angles'][ix];out[~flags]=o['value_scale']*torch.stack([a.cos(),a.sin()],1)@o['value_basis'].T
        if flags.any():
            ix=idx[flags];loc=(ix[:,None]==o['private_indices'][None,:]).long().argmax(1);out[flags]=o['private_values'][loc]
        return out
    return torch.zeros(len(idx),D)

def retrieve(q,method,o):
    if o is None:return torch.zeros(len(q),D),torch.zeros(len(q),dtype=torch.bool)
    sim=q@o['keys'].T;best,idx=sim.max(1);hit=best>=o['radii'][idx]
    out=select_values(method,o,idx);out[~hit]=0
    return out,hit

def evaluate(method,o,qe,qp,qn,values):
    ex,he=retrieve(qe,method,o);pa,hp=retrieve(qp,method,o);neg,hn=retrieve(qn,method,o)
    ee=float((ex-values[:len(qe)]).norm()/(values[:len(qe)].norm()+1e-12));target=values.repeat_interleave(4,0)[:len(qp)]
    pe=float((pa-target).norm()/(target.norm()+1e-12));dr=float(neg.norm()/(len(neg)**.5*SCALE+1e-12))
    return float(he.float().mean()),float(hp.float().mean()),float(hn.float().mean()),ee,pe,dr

def assess_threshold(tau,worlds):
    errors=[];fractions=[]
    for w in worlds:
      for s in SEEDS:
        keys,radii,basis,ang,vals,aligned=make_world(w,s);c,_,res=projection(vals,basis);priv=res>tau
        fractions.append(float(priv.float().mean()))
        errors.append(float((vals[~priv]-c[~priv]@basis.T).norm()/(vals[~priv].norm()+1e-12)) if (~priv).any() else 0.)
    return statistics.mean(errors),statistics.mean(fractions)

def run(phase):
    if phase=='development':
        sweep=[]
        for tau in TAUS:
            err,frac=assess_threshold(tau,DEV);sweep.append({'tau':tau,'projected_value_nrmse':err,'private_fraction':frac})
        valid=[x for x in sweep if x['projected_value_nrmse']<=1e-5]
        if not valid:raise RuntimeError('No tau meets development quality gate')
        chosen=max(x['tau'] for x in valid)
        (ART/'development_sweep.json').write_text(json.dumps(sweep,indent=2)+'\n')
        (ART/'development_selection.json').write_text(json.dumps({'tau':chosen,'rule':'largest threshold satisfying <=1e-5 development projection NRMSE','worlds':DEV,'seeds':SEEDS},indent=2)+'\n')
        print(json.dumps({'phase':phase,'selected_tau':chosen,'sweep':sweep}));return
    tau=json.loads((ART/'development_selection.json').read_text())['tau'];rows=[]
    for w in FRESH:
      for s in SEEDS:
        keys,radii,basis,ang,vals,aligned=make_world(w,s);qe,qp,qn=make_queries(w,s,keys)
        for method in ('no_edit','explicit','mirror_only','generic_only','mirror_fallback','generic_fallback','oracle_fallback'):
          for n in (1,8,32,64):
            blob=make_payload(method,keys,radii,basis,vals,aligned,n,tau);o=load_memory(blob)
            ex,pa,fp,ee,pe,dr=evaluate(method,o,qe[:n],qp[:n*4],qn,vals[:n])
            if method in ('mirror_fallback','generic_fallback','oracle_fallback'):frac=float(o['private_flags'].float().mean())
            elif method=='explicit':frac=1.0
            else:frac=0.0
            enc=[]
            for _ in range(25):
                t=time.perf_counter();make_payload(method,keys,radii,basis,vals,aligned,n,tau);enc.append(time.perf_counter()-t)
            lat=[]
            for _ in range(40):
                t=time.perf_counter();retrieve(qe[:n],method,o);lat.append((time.perf_counter()-t)/n)
            path=PAY/f'{w}_{s}_{method}_N{n}.pt'
            if blob:path.write_bytes(blob)
            rows.append({'world':w,'seed':s,'method':method,'n':n,'tau':tau,'exact_hit_recall':ex,'paraphrase_hit_recall':pa,'unrelated_false_trigger_rate':fp,'exact_value_nrmse':ee,'paraphrase_value_nrmse':pe,'unrelated_output_drift':dr,'fallback_fraction':frac,'payload_bytes':len(blob),'bytes_per_edit':len(blob)/n,'fallback_projection_MAC_per_new_edit':4*D,'lookup_MAC_per_query':2*n*D,'value_decode_MAC_per_hit':4*D,'payload_encode_seconds_per_edit':statistics.median(enc)/n,'lookup_decode_seconds_per_query':statistics.median(lat),'hash':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2])) if blob else 'empty'})
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'phase':phase,'tau':tau,'rows':len(rows)}))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
