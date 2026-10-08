#!/usr/bin/env python3
"""MA-1183 CPU mechanism: a TabM/BatchEnsemble-*style* model vs role-code charts.

Not an author/native TabM reproduction. No natural TabReD metrics are claimed.
Pre-registered under separate branch before fresh outcomes (commit 5002b45a...).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import time
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

KINDS=('native_full_fast_weights','fixed_seed_linear_code4','learned_dictionary_linear_code4','structured_mirror_rotation4','no_member_shared')
SPLITS={'dev':(11,12,13),'fresh':(101,102,103,104,105)}
BETAS=(0.,.5,1.)
DIM=32
H=64
C=2
K=8
R=4
GENER_SEED=190328
UPDATES=300
BATCH=128


def get_pattern(seed=GENER_SEED):
    g=torch.Generator().manual_seed(seed)
    # Two native-independent bases: deterministic orthogonal linear dictionary and fixed rotation angle selector.
    full=torch.randn(96,R,generator=g)
    q,_=torch.linalg.qr(full,mode='reduced')
    signs=torch.sign(torch.randn(48,R,generator=g))
    return q.contiguous()*5.0,signs.contiguous()*0.75


class FastWeightMemberMLP(nn.Module):
    """One shared MLP with K role-dependent rank-one input/hidden fast weights.

    TabM-style mechanism, *not* exact TabM architecture or training recipe.
    """
    def __init__(self,method,K=K,base_seed=101):
        super().__init__()
        if method not in KINDS:raise ValueError(method)
        self.method=method;self.K=K
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(base_seed))
            self.up=nn.Linear(DIM,H)
            self.down=nn.Linear(H,C)
            if method=='native_full_fast_weights':
                self.r_in=nn.Parameter(torch.randn(K,DIM)*.08+1.)
                self.s_h=nn.Parameter(torch.randn(K,H)*.08+1.)
            elif method=='no_member_shared':
                pass
            else:
                # Shared 96-vector native fast-weight template paid once.
                self.template=nn.Parameter(torch.randn(DIM+H)*.11)
                self.codes=nn.Parameter(torch.randn(K,R)*.2)
                if method=='learned_dictionary_linear_code4':
                    self.dictionary=nn.Parameter(torch.randn(R,DIM+H)*.035)
                else:
                    linear,angle=get_pattern()
                    self.register_buffer('fixed_linear',linear,persistent=False)
                    self.register_buffer('fixed_angle',angle,persistent=False)
    def fast_weights(self):
        if self.method=='native_full_fast_weights':
            return self.r_in,self.s_h
        if self.method=='no_member_shared':
            return self.up.weight.new_ones(self.K,DIM),self.up.weight.new_ones(self.K,H)
        if self.method=='fixed_seed_linear_code4':
            v=self.template.unsqueeze(0)+self.codes @ self.fixed_linear.T
        elif self.method=='learned_dictionary_linear_code4':
            v=self.template.unsqueeze(0)+self.codes @ self.dictionary
        else:
            angles=self.codes @ self.fixed_angle.T   # K x 48 angles
            tpl=self.template.reshape(48,2)
            a=tpl[:,0].unsqueeze(0)
            b=tpl[:,1].unsqueeze(0)
            ca,sa=angles.cos(),angles.sin()
            v=torch.stack((ca*a-sa*b,sa*a+ca*b),dim=-1).reshape(self.K,DIM+H)
        return (v[:,:DIM]+1.,v[:,DIM:]+1.)
    def forward(self,x):
        if self.method=='no_member_shared':
            # Optimized no-role native: one true heavy call, then a zero-copy
            # broadcast of K identical predictions. Never charge it K GEMMs.
            single=self.down(F.gelu(self.up(x)))
            return single.unsqueeze(1).expand(-1,self.K,-1)
        r,s=self.fast_weights()
        # One set of W tensors; K true per-member input-weighted projections.
        h=F.gelu(F.linear(x[:,None,:]*r[None,:,:],self.up.weight,self.up.bias))
        h=h*s[None,:,:]
        return F.linear(h,self.down.weight,self.down.bias)
    def serialized_bytes(self):
        mem=io.BytesIO()
        variables={n: v.detach().cpu().numpy() for n,v in self.state_dict().items()}
        metadata={'method':self.method,'K':self.K,'input':DIM,'hidden':H,'classes':C,'pattern_seed':GENER_SEED,'float_dtype':'float32','codebook_derived_from_seed':True}
        variables['__metadata__']=np.frombuffer(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
        np.savez(mem,**variables)
        return len(mem.getvalue())
    def member_scalars(self):
        if self.method=='native_full_fast_weights':return self.K*(DIM+H)
        if self.method=='no_member_shared':return 0
        return (DIM+H)+self.K*R+(R*(DIM+H) if self.method=='learned_dictionary_linear_code4' else 0)


def generate(seed,beta):
    rng=np.random.default_rng(9437*seed+int(beta*100)+310)
    q,_=np.linalg.qr(rng.normal(size=(DIM,DIM)))
    # Across-world row dependence and a rotating nuisance axis, not a task ID.
    scale=np.linspace(.55,1.65,DIM)
    def features(n):
        x=rng.normal(size=(n,DIM))
        x=x@np.diag(scale)@(q*.48+np.eye(DIM)*.52)
        return x.astype(np.float32)
    u=rng.normal(size=(DIM,6)).astype(np.float32)/np.sqrt(DIM)
    v=rng.normal(size=(DIM,4)).astype(np.float32)/np.sqrt(DIM)
    w=rng.normal(size=6)
    def target(x):
        z=x@u
        r=x@v
        # A shared compositional structure plus off-orbit private interactions.
        score=1.0*z[:,0]-.8*z[:,1]+.70*np.sin(1.6*z[:,2]+z[:,3])+.65*z[:,4]*z[:,5]
        score += beta*(.95*r[:,0]*r[:,1]+.55*np.sin(2.1*r[:,2]-1.3*r[:,3])+0.25*np.cos(x[:,17]*x[:,18]))
        # Calibrate from training-only observations to keep class balance.
        return score
    tr=features(1024);val=features(256);test=features(512)
    trainlogit=target(tr)
    threshold=float(np.median(trainlogit))
    def emit(x):
        logits=(target(x)-threshold)/1.2
        prob=1/(1+np.exp(-np.clip(logits,-30,30)))
        return rng.binomial(1,prob).astype(np.int64)
    return (tr,emit(tr)),(val,emit(val)),(test,emit(test))


def evaluate(model,data):
    model.eval()
    x,y=data
    with torch.no_grad():
        logits=model(torch.from_numpy(x))
        prob=logits.softmax(-1).mean(1)[:,1].cpu().numpy()
        pred=logits.argmax(-1)
        labels=torch.from_numpy(y)
        member_ce=F.cross_entropy(logits.reshape(-1,2),labels.repeat_interleave(K)).item()
        member_logit_sd=(logits[:,:,1]-logits[:,:,0]).std(dim=1).mean().item()
    prob=np.clip(prob,1e-6,1-1e-6)
    nll=float(np.mean(-y*np.log(prob)-(1-y)*np.log(1-prob)))
    auc=float(roc_auc_score(y,prob)) if len(np.unique(y))>1 else float('nan')
    brier=float(np.mean((prob-y)**2))
    ece=0.
    for l,r in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(prob>=l)&(prob<r if r<1 else prob<=r)
        if mask.any():ece+=float(mask.mean()*abs(y[mask].mean()-prob[mask].mean()))
    return {'ensemble_nll':nll,'auc':auc,'brier':brier,'ece':ece,'member_ce':member_ce,'member_logit_sd':member_logit_sd}


def train(seed,beta,method):
    (tr_x,tr_y),val,test=generate(seed,beta)
    torch.manual_seed(seed+1811)
    model=FastWeightMemberMLP(method,base_seed=seed+708)
    optimiser=torch.optim.AdamW(model.parameters(),lr=.004,weight_decay=.0001)
    xt=torch.from_numpy(tr_x);yt=torch.from_numpy(tr_y)
    stream=torch.Generator().manual_seed(seed*7919+int(beta*29)+1093)
    model.train()
    for t in range(UPDATES):
        ids=torch.randint(len(xt),(BATCH,),generator=stream)
        logits=model(xt[ids])
        targets=yt[ids][:,None].expand(-1,K)
        loss=F.cross_entropy(logits.reshape(-1,C),targets.reshape(-1))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    val_metrics=evaluate(model,val)
    fresh_metrics=evaluate(model,test)
    return model,val_metrics,fresh_metrics


def benchmark(model,seed):
    rng=np.random.default_rng(98+seed)
    x=torch.from_numpy(rng.standard_normal((128,DIM),dtype=np.float32))
    model.eval();torch.set_num_threads(1)
    vals=[]
    with torch.no_grad():
        for _ in range(20):model(x)
        for _ in range(100):
            t=time.perf_counter_ns();model(x)
            vals.append((time.perf_counter_ns()-t)/1e6)
    return float(np.median(vals)),float(np.quantile(vals,.95))


def run(phase,dest):
    torch.set_num_threads(1)
    dest=Path(dest);dest.mkdir(exist_ok=True,parents=True)
    rows=[]
    start=time.perf_counter()
    for seed in SPLITS[phase]:
        for beta in BETAS:
            for method in KINDS:
                model,val,test=train(seed,beta,method)
                p50,p95=benchmark(model,seed)
                rows.append({'phase':phase,'seed':seed,'beta':beta,'method':method,'K':K,
                             **{f'val_{k}':v for k,v in val.items()},**{f'heldout_{k}':v for k,v in test.items()},
                             'serialized_npz_bytes':model.serialized_bytes(),'member_params':model.member_scalars(),'total_params':sum(v.numel() for v in model.parameters()),
                             'cpu_p50_ms':p50,'cpu_p95_ms':p95,'training_steps':UPDATES,'train_samples':1024,'test_samples':512})
        print(json.dumps({'phase':phase,'seed':seed,'done':len([r for r in rows if r['seed']==seed]),'elapsed_s':round(time.perf_counter()-start,2)}),flush=True)
    with (dest/f'{phase}_raw.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['dev','fresh','all'],default='all');p.add_argument('--output',default='results/ma1183');args=p.parse_args()
    if args.phase in ['dev','all']:run('dev',args.output)
    if args.phase in ['fresh','all']:run('fresh',args.output)

if __name__=='__main__':main()
