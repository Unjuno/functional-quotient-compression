#!/usr/bin/env python3
"""MA1183 second isolated pilot: two sklearn tabular datasets, native-style fast weights.

Independent preregistered protocol: MA1183_REAL_TABULAR_FROZEN_PROTOCOL.json.
This is NOT official TabM/TabReD, only TabM-/BatchEnsemble-style member fast weights.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path
import time
import numpy as np
from sklearn.datasets import load_breast_cancer,load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from ma1183_tabm_style import FastWeightMemberMLP,KINDS,K,BATCH,DIM,C,evaluate,benchmark

SEEDS=(301,302,303,304,305)
CHECKS=(25,50,100,150,200)
MAX_UPDATES=200
LR=.002


def real_dataset(name,seed):
    if name=='breast_cancer':
        bunch=load_breast_cancer()
        label=np.asarray(bunch.target,dtype=np.int64)
    elif name=='wine_binary':
        bunch=load_wine()
        label=(np.asarray(bunch.target)==0).astype(np.int64)
    else:raise ValueError(name)
    data=np.asarray(bunch.data,dtype=np.float64)
    dataset_sha256=hashlib.sha256(data.tobytes()+label.tobytes()).hexdigest()
    ids=np.arange(len(label))
    src,test=train_test_split(ids,test_size=.30,random_state=seed,stratify=label)
    tr,valid=train_test_split(src,test_size=.20,random_state=seed+17,stratify=label[src])
    assert not(set(tr)&set(test) or set(tr)&set(valid) or set(valid)&set(test))
    scaler=StandardScaler()
    xtrain=scaler.fit_transform(data[tr])
    xv=scaler.transform(data[valid]);xt=scaler.transform(data[test])
    if data.shape[1]>DIM:raise AssertionError('too many features')
    def padded(x):
        x=np.asarray(x,dtype=np.float32)
        if x.shape[1]<DIM:x=np.pad(x,((0,0),(0,DIM-x.shape[1])))
        return x
    return (padded(xtrain),label[tr]),(padded(xv),label[valid]),(padded(xt),label[test]),dataset_sha256


def fit_data(dataset,seed,method):
    train,valid,heldout,dh=real_dataset(dataset,seed)
    torch.manual_seed(seed+2811)
    m=FastWeightMemberMLP(method,base_seed=seed+708)
    optimizer=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=.0001)
    xt=torch.from_numpy(train[0]);yt=torch.from_numpy(train[1])
    stream=torch.Generator().manual_seed(seed*7919+381)
    best=float('inf');best_step=None;best_state=None
    trace=[]
    for step in range(1,MAX_UPDATES+1):
        sample=torch.randint(len(xt),(64,),generator=stream)
        logits=m(xt[sample]);labels=yt[sample][:,None].expand(-1,K)
        loss=F.cross_entropy(logits.reshape(-1,C),labels.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward();optimizer.step()
        if step in CHECKS:
            met=evaluate(m,valid)
            trace.append((step,met['ensemble_nll']))
            if met['ensemble_nll']<best:
                best=met['ensemble_nll'];best_step=step
                best_state={k:v.detach().clone() for k,v in m.state_dict().items()}
    assert best_state is not None
    m.load_state_dict(best_state)
    val=evaluate(m,valid);test=evaluate(m,heldout)
    p50,p95=benchmark(m,seed)
    assert val['ensemble_nll']-best<1e-8
    return {'dataset':dataset,'seed':seed,'method':method,'train_samples':len(train[0]),'val_samples':len(valid[0]),'test_samples':len(heldout[0]),'dataset_sha256':dh,
            'selected_step':best_step,'max_steps':MAX_UPDATES,'validation_nll':val['ensemble_nll'],
            **{f'test_{k}':v for k,v in test.items()},'serialized_npz_bytes':m.serialized_bytes(),'member_params':m.member_scalars(),'cpu_p50_ms':p50,'cpu_p95_ms':p95,'validation_history':';'.join(f'{step}:{loss:.7g}' for step,loss in trace)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/ma1183_real');args=ap.parse_args()
    path=Path(args.output);path.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    out=[];start=time.perf_counter()
    for name in ('breast_cancer','wine_binary'):
        for seed in SEEDS:
            for meth in KINDS:
                out.append(fit_data(name,seed,meth))
            print(f'{name} seed={seed} completed 5 methods elapsed={time.perf_counter()-start:.2f}s',flush=True)
    with (path/'real_raw.csv').open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(out[0]));wr.writeheader();wr.writerows(out)
    print('output',path/'real_raw.csv','rows',len(out))

if __name__=='__main__':main()
