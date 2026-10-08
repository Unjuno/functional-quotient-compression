#!/usr/bin/env python3
"""MAT02: follow-up on distinct seeds after MAT01, same frozen data protocol.
Frozen protocol commit: c7b7b517c66d757930baf4b7054535c13a0dcf2a.
Source-only U/V shared; test paid private LoRA r2/r1 fallbacks vs native and linear.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from run_mat01 import (make_world,fit_base,source_family,fit_source_basis,target_support,role_batch,
                       core_from_code,actual_npz_bytes,WorldData,
                       TARGET_ROLES,SOURCE_ROLES,D,H,C,R,SOURCE_STEPS,TARGET_STEPS,
                       BASE_STEPS,FrozenBackbone)

DEV=(21,22,23)
FRESH=(201,202,203,204,205)
METHODS=('base','native_lora4','native_lora2','mirror4','linear4','mirror4_private2','linear4_private2','fullcore16_private1')


def freeze_target(method:str,base:FrozenBackbone,w:WorldData,basis,role:str,seed:int,task_num:int,support_ids):
    x=role_batch(w,support_ids,role,seed,'target_train')
    test=role_batch(w,w.audit_ids,role,seed,'target_audit')
    with torch.no_grad():
        h=base.features(x); q=base.down(h)
        ht=base.features(test); qt=base.down(ht)
    labels=torch.tensor(w.labels[support_ids]); yt=torch.tensor(w.labels[w.audit_ids])
    U,V,C0,dic=basis
    if method=='base':
        with torch.no_grad():out=qt
        return {'code':None,'a':None,'b':None,'nll':float(F.cross_entropy(out,yt)),
                'acc':float((out.argmax(-1)==yt).float().mean())}
    native=method in ('native_lora4','native_lora2')
    private_rank=4 if method=='native_lora4' else 2 if method in ('native_lora2','mirror4_private2','linear4_private2') else 1 if method=='fullcore16_private1' else 0
    ctype='mirror4' if method.startswith('mirror4') else 'linear4' if method.startswith('linear4') else 'fullcore16' if method.startswith('fullcore16') else None
    n=R*R if ctype=='fullcore16' else R
    if ctype:
        code=nn.Parameter(torch.zeros(n))
        if ctype=='mirror4':
            with torch.no_grad():code.copy_(torch.tensor([.002,-.002,.003,-.001]))
    else:code=None
    if private_rank:
        g=torch.Generator().manual_seed((seed*191+task_num)*1949+139)
        a=nn.Parameter(.02*torch.randn(H,private_rank,generator=g))
        b=nn.Parameter(torch.zeros(private_rank,C))
    else:a=b=None
    params=(([code] if code is not None else [])+([a,b] if a is not None else []))
    optimizer=torch.optim.AdamW(params,lr=.045,weight_decay=.0001)
    stream=torch.Generator().manual_seed((seed*191+task_num)*7919+269)
    for _ in range(TARGET_STEPS):
        i=torch.randint(len(labels),(96,),generator=stream)
        logits=q[i]
        if ctype:logits=logits+h[i]@U@core_from_code(ctype,code,C0,dic)@V
        if private_rank:logits=logits+h[i]@a@b
        loss=F.cross_entropy(logits,labels[i]);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    with torch.no_grad():
        logits=qt
        if ctype:logits=logits+ht@U@core_from_code(ctype,code,C0,dic)@V
        if private_rank:logits=logits+ht@a@b
        nll=float(F.cross_entropy(logits,yt));acc=float((logits.argmax(-1)==yt).float().mean())
    return {'code':code.detach().clone() if code is not None else None,
            'a':a.detach().clone() if a is not None else None,
            'b':b.detach().clone() if b is not None else None,'nll':nll,'acc':acc}


def arrays(base:FrozenBackbone,w:WorldData,basis,method:str,trained):
    out={f'base_{k}':p.detach().numpy().astype(np.float32) for k,p in base.state_dict().items()}
    out['scaler_mean']=w.scaler_mean;out['scaler_scale']=w.scaler_scale
    if method not in ('base','native_lora2','native_lora4'):
        U,V,C0,dic=basis
        out.update(source_U=U.numpy(),source_V=V.numpy(),source_C0=C0.numpy())
        if method.startswith('linear4'):out['source_linear_dictionary']=dic.numpy()
    for idx,res in enumerate(trained):
        if res['code'] is not None:out[f'task_{idx}_m']=res['code'].numpy().astype(np.float32)
        if res['a'] is not None:
            out[f'task_{idx}_A']=res['a'].numpy().astype(np.float32)
            out[f'task_{idx}_B']=res['b'].numpy().astype(np.float32)
    metadata={"method":method,"roles":list(TARGET_ROLES),"base_model":"64-64-GELU-10","source_roles":list(SOURCE_ROLES),"src_split":"stratified_75pct","scaler":"train_only", "stored_source":False}
    out['__metadata__']=np.frombuffer(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode('utf-8'),dtype=np.uint8)
    return out


def benchmark(base,w,basis,method,trained,seed):
    inp=role_batch(w,w.audit_ids[:128],'original',seed,'runtime')
    U,V,C0,dic=basis
    ctype='mirror4' if method.startswith('mirror4') else 'linear4' if method.startswith('linear4') else 'fullcore16' if method.startswith('fullcore16') else None
    def forward():
        h=base.features(inp);q=base.down(h)
        if method=='base':return q.unsqueeze(1).expand(-1,4,-1)
        outs=[]
        for res in trained:
            logits=q
            if ctype:logits=logits+h@U@core_from_code(ctype,res['code'],C0,dic)@V
            if res['a'] is not None:logits=logits+h@res['a']@res['b']
            outs.append(logits)
        return torch.stack(outs,dim=1)
    with torch.no_grad():
        for _ in range(20):forward()
        times=[]
        for _ in range(100):
            now=time.perf_counter_ns();forward();times.append((time.perf_counter_ns()-now)/1e6)
    return float(np.median(times)),float(np.quantile(times,.95))


def run_seed(seed,phase):
    w=make_world(seed);base=fit_base(w,seed)
    source=source_family(base,w,seed);basis=fit_source_basis(source)
    sourcehash=hashlib.sha256(source.tobytes()).hexdigest()
    support=target_support(w,seed)
    rows=[]
    for method in METHODS:
        trained=[freeze_target(method,base,w,basis,role,seed,i,support) for i,role in enumerate(TARGET_ROLES)]
        byte_count=actual_npz_bytes(arrays(base,w,basis,method,trained))
        p50,p95=benchmark(base,w,basis,method,trained,seed)
        for task,fit in zip(TARGET_ROLES,trained):
            rows.append({"phase":phase,"seed":seed,"role":task,"method":method,"nll":fit['nll'],
                         "accuracy":fit['acc'],"serialized_npz_bytes":byte_count,
                         "cpu_p50_ms":p50,"cpu_p95_ms":p95,"source_family_sha256":sourcehash,
                         "dataset_sha256":w.digest,"support_size":len(support),"audit_size":len(w.audit_ids),"train_steps":0 if method=='base' else TARGET_STEPS,"pretrained_base_steps":BASE_STEPS,"source_lora_steps":SOURCE_STEPS})
    return rows


def run(phase,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    start=time.perf_counter();rows=[]
    for seed in (DEV if phase=='dev' else FRESH):
        arr=run_seed(seed,phase);rows.extend(arr)
        summary={m:round(float(np.mean([z['nll'] for z in arr if z['method']==m])),4) for m in METHODS}
        print(json.dumps({"phase":phase,"seed":seed,"rows":len(arr),"nll":summary,"elapsed_s":round(time.perf_counter()-start,1)}),flush=True)
    with (dest/f'{phase}_raw.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('dev','fresh','all'),default='all');ap.add_argument('--out',default='results');args=ap.parse_args()
    if args.phase in ('all','dev'):run('dev',args.out)
    if args.phase in ('all','fresh'):run('fresh',args.out)
if __name__=='__main__':main()
