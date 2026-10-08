#!/usr/bin/env python3
"""SFM003 preregistered, trained lightweight readouts on a fixed shared heavy trunk.

No GPU, natural language, native MoE, or trained shared trunk claim. User's core
question is ONE heavy forward -> MANY useful, nonidentical task outputs.
Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_sfm003.py --phase dev|fresh|all
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import statistics
import time
from itertools import combinations
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

SEEDS = {'dev': [11,12,13], 'fresh':[101,102,103,104,105]}
KS=(4,5)
BETAS=(0.,0.3,0.8)
DIN=16
HIDDEN=64
DOUT=8
PRIVATE_HIDDEN=32
TRAIN=768
DEV=256
TEST=512
STEPS=350
LR=0.04
torch.set_num_threads(1)


def fresh_generator(seed):
    return torch.Generator().manual_seed(int(seed)*10007+17)


def base(x,U,D):
    return F.gelu(x @ U) @ D


def rotate(y,angles):
    # Input [batch,d] or [K,batch,d], angles [K,d/2]
    p=y.reshape(y.shape[:-1]+(DOUT//2,2))
    c=torch.cos(angles)[...,None,:]
    s=torch.sin(angles)[...,None,:]
    a, b=p[...,0], p[...,1]
    return torch.stack((c*a-s*b,s*a+c*b),dim=-1).reshape(y.shape[:-1]+(DOUT,))


def rotate_stored(y,cos,sin):
    p=y.reshape(y.shape[:-1]+(DOUT//2,2))
    c=cos[...,None,:]
    s=sin[...,None,:]
    return torch.stack((c*p[...,0]-s*p[...,1],s*p[...,0]+c*p[...,1]),dim=-1).reshape(y.shape[:-1]+(DOUT,))


def world(seed,K,beta):
    g=fresh_generator(seed)
    U=torch.randn(DIN,HIDDEN,generator=g)/math.sqrt(DIN)
    D=torch.randn(HIDDEN,DOUT,generator=g)/math.sqrt(HIDDEN)
    # Separate matrix of independent private teachers; not available to fitting algorithms.
    P=torch.randn(K,DIN,PRIVATE_HIDDEN,generator=g)/math.sqrt(DIN)
    R=torch.randn(K,PRIVATE_HIDDEN,DOUT,generator=g)/math.sqrt(PRIVATE_HIDDEN)
    angles=(2*torch.rand(K,DOUT//2,generator=g)-1)*0.8
    xs=[torch.randn(n,DIN,generator=g) for n in (TRAIN,DEV,TEST)]
    zs=[base(x,U,D).detach() for x in xs]
    ys=[]
    # Fixed scaling estimated on train only. Per-role normalizes off-orbit difficulty.
    ptrain=torch.einsum('nd,kdh->knh',xs[0],P).tanh()
    private_train=torch.einsum('knh,kho->kno',ptrain,R)
    ratio=zs[0].std().clamp(min=1e-6)/private_train.std(dim=(1,2),keepdim=True).clamp(min=1e-6)
    for x,z in zip(xs,zs):
        priv=torch.einsum('knh,kho->kno',torch.einsum('nd,kdh->knh',x,P).tanh(),R)
        y=rotate(z.unsqueeze(0).expand(K,-1,-1),angles)+beta*priv*ratio
        ys.append(y.detach())
    return {'U':U,'D':D,'xs':xs,'zs':zs,'ys':ys,'true_angles':angles}


def predict(z,method,p):
    K=len(p[0])
    y0=z.unsqueeze(0).expand(K,-1,-1)
    if method=='mirror':
        return rotate(y0,p[0])
    if method=='diag':
        return y0*p[0][:,None,:]+p[1][:,None,:]
    if method=='linear':
        return torch.einsum('nd,kdo->kno',z,p[0])+p[1][:,None,:]
    raise ValueError(method)


def fit(method,trainz,trainy):
    K=trainy.shape[0]
    if method=='mirror':
        params=[torch.nn.Parameter(torch.zeros(K,DOUT//2))]
    elif method=='diag':
        params=[torch.nn.Parameter(torch.ones(K,DOUT)),torch.nn.Parameter(torch.zeros(K,DOUT))]
    elif method=='linear':
        params=[torch.nn.Parameter(torch.eye(DOUT).unsqueeze(0).expand(K,-1,-1).clone()),torch.nn.Parameter(torch.zeros(K,DOUT))]
    else:
        raise ValueError(method)
    optimizer=torch.optim.Adam(params,lr=LR,weight_decay=0)
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        out=predict(trainz,method,params)
        loss=F.mse_loss(out,trainy)
        loss.backward()
        optimizer.step()
    return [p.detach().clone() for p in params],float(loss.detach())


def nmse(pred,target):
    numerator=((pred-target)**2).mean(dim=(1,2))
    denominator=(target-target.mean(dim=1,keepdim=True)).square().mean(dim=(1,2)).clamp(min=1e-9)
    return (numerator/denominator).tolist()


def diversity_ratio(pred,target):
    ds=[]
    for i,j in combinations(range(target.shape[0]),2):
        t=float((target[i]-target[j]).square().mean().sqrt())
        p=float((pred[i]-pred[j]).square().mean().sqrt())
        ds.append(p/max(t,1e-9))
    return min(ds),statistics.median(ds)


def npz_size(tensors):
    f=io.BytesIO()
    np.savez(f,**{name:obj.detach().cpu().numpy() for name,obj in tensors.items()})
    return len(f.getvalue())


def model_bytes(w,method,pars):
    d={'U':w['U'],'D':w['D']}
    if method=='mirror': d['angles']=pars[0]
    elif method=='diag': d.update({'gains':pars[0],'bias':pars[1]})
    elif method=='linear': d.update({'heads':pars[0],'bias':pars[1]})
    else: raise ValueError(method)
    return npz_size(d)


def _last(out):
    return out.sum().item()


def benchmark(w,pars,method,K,B=128,reps=100,warmup=20):
    x=w['xs'][2][:B]
    U,D=w['U'],w['D']
    p=pars
    if method=='mirror':
        c=torch.cos(p[0]);s=torch.sin(p[0])
        def op():
            z=base(x,U,D)
            return rotate_stored(z.unsqueeze(0).expand(K,-1,-1),c,s)
    elif method=='linear':
        def op():
            z=base(x,U,D)
            return torch.einsum('nd,kdo->kno',z,p[0])+p[1][:,None,:]
    elif method=='diag':
        def op():
            z=base(x,U,D)
            return z.unsqueeze(0)*p[0][:,None,:]+p[1][:,None,:]
    elif method=='redundant':
        c=torch.cos(p[0]);s=torch.sin(p[0])
        def op():
            # Explicitly redundant K physical block passes, not a strong native multi-head baseline.
            return torch.stack([rotate_stored(base(x,U,D),c[k],s[k]) for k in range(K)])
    elif method=='shared_same':
        def op():
            z=base(x,U,D)
            return z.unsqueeze(0).expand(K,-1,-1)
    else: raise ValueError(method)
    with torch.inference_mode():
        for _ in range(warmup):_last(op())
        times=[]
        for _ in range(reps):
            before=time.perf_counter_ns();_last(op());after=time.perf_counter_ns()
            times.append((after-before)/1e6)
    return {'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)), 'min_ms':float(min(times))}


def one(seed,phase,K,beta,timings):
    w=world(seed,K,beta)
    train_z, dev_z, test_z=w['zs']
    train_y,dev_y,test_y=w['ys']
    vals={}
    keys=('mirror','diag','linear')
    result=[]
    for method in keys:
        params,loss=fit(method,train_z,train_y)
        train_pred=predict(train_z,method,params)
        dev_pred=predict(dev_z,method,params)
        pred=predict(test_z,method,params)
        errs=nmse(pred,test_y)
        distinct_min,distinct_med=diversity_ratio(pred,test_y)
        vals[method]=params
        row={'phase':phase,'seed':seed,'K':K,'beta':beta,'method':method,'train_mse':loss,
             'dev_nmse_mean':float(np.mean(nmse(dev_pred,dev_y))),
             'audit_nmse_mean':float(np.mean(errs)),'audit_nmse_worst':max(errs),
             'output_pair_sep_min_ratio':distinct_min,'output_pair_sep_median_ratio':distinct_med,
             'inference_npz_bytes':model_bytes(w,method,params),'code_scalars_per_task':sum(p.numel() for p in params)//K,
             'learned_head_updates':STEPS,'teacher_known_trunk':True}
        result.append(row)
    shared_pred=test_z.unsqueeze(0).expand(K,-1,-1)
    ds,dm=diversity_ratio(shared_pred,test_y)
    result.append({'phase':phase,'seed':seed,'K':K,'beta':beta,'method':'shared_same','train_mse':float('nan'),
        'dev_nmse_mean':float(np.mean(nmse(dev_z.unsqueeze(0).expand(K,-1,-1),dev_y))),
        'audit_nmse_mean':float(np.mean(nmse(shared_pred,test_y))),
        'audit_nmse_worst':max(nmse(shared_pred,test_y)),
        'output_pair_sep_min_ratio':ds,'output_pair_sep_median_ratio':dm,
        'inference_npz_bytes':npz_size({'U':w['U'],'D':w['D']}),'code_scalars_per_task':0,'learned_head_updates':0,'teacher_known_trunk':True})
    # Native ordinary Givens is math-identical. This must be explicitly called M0.
    y_mirror=predict(test_z,'mirror',vals['mirror'])
    y_native=rotate(test_z.unsqueeze(0).expand(K,-1,-1),vals['mirror'][0])
    if not torch.equal(y_mirror,y_native): raise AssertionError('native ordinary head parity failed')
    if beta==0:
        # An architecture that knows the exact real teacher View may skip code training.
        direct=rotate(test_z.unsqueeze(0).expand(K,-1,-1),w['true_angles'])
        assert float((direct-test_y).abs().max())==0
    if timings is not None and beta==0:
        # Timings only on beta 0; beta doesn't affect inference computation and code shape.
        for method in ('mirror','diag','linear','redundant','shared_same'):
            p=vals['mirror'] if method=='redundant' else vals.get(method,[])
            info=benchmark(w,p,method,K)
            timings.append({'phase':phase,'seed':seed,'K':K,'beta':beta,'method':method,**info})
    return result


def run(phase,output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    allrows=[];alltimes=[]
    for seed in SEEDS[phase]:
        for K in KS:
            for beta in BETAS:
                allrows.extend(one(seed,phase,K,beta,alltimes))
        print(f'done {phase} seed {seed}',flush=True)
    def write(name,rows):
        with (output/name).open('w',newline='',encoding='utf-8') as f:
            wr=csv.DictWriter(f,fieldnames=rows[0].keys())
            wr.writeheader();wr.writerows(rows)
    write(phase+'_quality.csv',allrows)
    write(phase+'_timing.csv',alltimes)
    return allrows,alltimes


def make_verification(output):
    path=Path(output)
    info={'scope':'stage0 synthetic, frozen source common heavy trunk, train only tiny task heads, no native LM claims',
        'software':{'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version()},
        'hardware':{'cpu':platform.processor(),'cuda':torch.cuda.is_available(),'threads':torch.get_num_threads()},
        'seed_sets':SEEDS,'K':KS,'beta':BETAS,'steps':STEPS,'lr':LR,'train_examples':TRAIN,
        'serialization':'named float32 np.savez arrays, real file byte count; no broad tokenizer/LLM metadata',
        'main_boundary':'ordinary paired-Givens head is exactly equivalent to Mirror head (M0); native linear multi-head shares trunk too',
        'files':{}}
    for f in sorted(path.glob('*.csv')):
        info['files'][f.name]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size}
    info['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (path/'VERIFICATION.json').write_text(json.dumps(info,indent=2,default=str)+'\n',encoding='utf-8')
    return info


if __name__=='__main__':
    arg=argparse.ArgumentParser()
    arg.add_argument('--phase',choices=['dev','fresh','all'],default='dev')
    arg.add_argument('--output',default='../results')
    args=arg.parse_args()
    phases=['dev','fresh'] if args.phase=='all' else [args.phase]
    for p in phases: run(p,args.output)
    print(json.dumps(make_verification(args.output),indent=2))
