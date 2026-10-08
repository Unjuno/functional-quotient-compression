#!/usr/bin/env python3
"""MAT03: true common-input one-forward five-task readout study.

FROZEN before any development/fresh: GitHub protocol commit
52afafbae5b4f1888e632488f85c430e01d749fa.
Not a native TabM, MIMO, or LoGo paper reproduction.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as Fn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

METHODS=("shared","native_multihead","native_fastweight","mirror_hadamard4",
         "native_hadamard_linear4","native_direct_diagonal4","native_learned_dictionary4",
         "mirror_private1","native_linear_private1")
SEEDS={'dev':(31,32,33),'fresh':(301,302,303,304,305)}
TASKS=('digit_parity','digit_ge5','left_right_balance','upper_lower_balance','center_density')
K=5
STEPS=400
BATCH=128
F=32
CODE=4


def hadamard(n:int)->torch.Tensor:
    assert n>0 and (n&(n-1))==0
    x=np.asarray([[1.]],dtype=np.float64)
    while x.shape[0]<n:x=np.block([[x,x],[x,-x]])
    return torch.from_numpy((x/np.sqrt(n)).astype(np.float32))


def patterns(seed:int=14051):
    g=torch.Generator().manual_seed(seed)
    B=torch.randn(F,CODE,generator=g)
    Q,_=torch.linalg.qr(B,mode='reduced')
    return hadamard(F),Q*3.0


def labels_from_images(raw:np.ndarray,digit_labels:np.ndarray,train_ids:np.ndarray):
    """Five distinct supervised targets on the same physical image input."""
    p=raw.reshape(-1,8,8)
    mid=p[:,:,0:4].sum((1,2))-p[:,:,4:8].sum((1,2))
    vert=p[:,0:4,:].sum((1,2))-p[:,4:8,:].sum((1,2))
    center=p[:,2:6,2:6].sum((1,2))
    vals=(mid,vert,center)
    thresholds=[float(np.median(t[train_ids])) for t in vals]
    ys=[digit_labels%2, (digit_labels>=5).astype(np.int64)]
    ys.extend((x>tr).astype(np.int64) for x,tr in zip(vals,thresholds))
    return np.stack(ys,axis=1).astype(np.int64),thresholds


def world(seed:int):
    d=load_digits();raw=np.asarray(d.data,dtype=np.float32)/16.
    y=np.asarray(d.target,dtype=np.int64)
    a,b=train_test_split(np.arange(len(y)),test_size=.25,random_state=seed,stratify=y)
    a,b=np.asarray(a),np.asarray(b)
    target,thresholds=labels_from_images(raw,y,a)
    mean=raw[a].mean(axis=0).astype(np.float32)
    sd=(raw[a].std(axis=0)+.15).astype(np.float32)
    x=(raw-mean)/sd
    digest=hashlib.sha256(raw.tobytes()+y.tobytes()).hexdigest()
    assert not set(a)&set(b)
    return torch.from_numpy(x[a].copy()),torch.from_numpy(target[a].copy()),torch.from_numpy(x[b].copy()),torch.from_numpy(target[b].copy()),digest,thresholds,mean,sd


class MultiHeadMirror(nn.Module):
    def __init__(self,method:str,seed:int):
        super().__init__()
        if method not in METHODS:raise ValueError(method)
        self.method=method
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed*1439+31)
            self.trunk=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,F),nn.GELU())
            self.common=nn.Linear(F,2)
            if method=='native_multihead':
                self.native_heads=nn.Parameter(self.common.weight.detach().clone().unsqueeze(0).repeat(K,1,1))
                self.native_bias=nn.Parameter(self.common.bias.detach().clone().unsqueeze(0).repeat(K,1))
            if method=='native_fastweight':
                self.gate=nn.Parameter(torch.zeros(K,F));self.task_bias=nn.Parameter(torch.zeros(K,2))
            if method in ('mirror_hadamard4','native_hadamard_linear4','native_direct_diagonal4',
                          'native_learned_dictionary4','mirror_private1','native_linear_private1'):
                self.codes=nn.Parameter(torch.randn(K,CODE)*.005)
                if method=='native_learned_dictionary4':
                    self.learned_dictionary=nn.Parameter(torch.randn(CODE,F)*.03)
                if method in ('mirror_private1','native_linear_private1'):
                    self.A=nn.Parameter(torch.randn(K,F,1)*.02)
                    self.B_private=nn.Parameter(torch.zeros(K,1,2))
        H,B=patterns()
        self.register_buffer('fixed_H',H,persistent=False)
        self.register_buffer('fixed_pattern',B,persistent=False)

    def features(self,x):return self.trunk(x)

    def forward(self,x):
        h=self.trunk(x) # IMPORTANT EXACTLY ONE TRUNK CALL FOR ALL K TASKS
        if self.method=='shared':
            return self.common(h).unsqueeze(1).expand(-1,K,-1)
        if self.method=='native_multihead':
            return torch.einsum('bf,kcf->bkc',h,self.native_heads)+self.native_bias.unsqueeze(0)
        if self.method=='native_fastweight':
            hrole=h.unsqueeze(1)*torch.exp(self.gate).unsqueeze(0)
            return Fn.linear(hrole,self.common.weight,self.common.bias)+self.task_bias.unsqueeze(0)
        if self.method=='native_direct_diagonal4':
            scale=torch.ones(K,F,dtype=h.dtype,device=h.device)
            scale=scale.scatter(1,torch.tensor([0,8,16,24],device=h.device).unsqueeze(0).expand(K,-1),
                                torch.exp(self.codes))
            hrole=h.unsqueeze(1)*scale.unsqueeze(0)
        else:
            H=self.fixed_H
            B=self.fixed_pattern
            # Hadamard feature chart is computed ONCE from the shared hidden vector.
            chart=h@H
            if self.method in ('mirror_hadamard4','mirror_private1'):
                scale=torch.exp(self.codes@B.T)
            elif self.method in ('native_hadamard_linear4','native_linear_private1'):
                scale=1.0+self.codes@B.T
            elif self.method=='native_learned_dictionary4':
                scale=1.0+self.codes@self.learned_dictionary
            else:raise RuntimeError(self.method)
            hrole=(chart.unsqueeze(1)*scale.unsqueeze(0))@H.T
        logits=Fn.linear(hrole,self.common.weight,self.common.bias)
        if self.method in ('mirror_private1','native_linear_private1'):
            residual=torch.einsum('bf,kfr,krc->bkc',h,self.A,self.B_private)
            logits=logits+residual
        return logits

    def serialized_bytes(self,means,sd,thresholds):
        tensors={k:v.detach().cpu().numpy().astype(np.float32) for k,v in self.state_dict().items()}
        # The unused common head is not part of the *native_multihead* deployed inference.
        if self.method=='native_multihead':
            tensors.pop('common.weight',None);tensors.pop('common.bias',None)
        tensors['train_only_mean']=means.astype(np.float32)
        tensors['train_only_sd']=sd.astype(np.float32)
        meta={'method':self.method,'K':K,'F':F,'pattern_seed':14051,'size':'64-128-32-2',
              'thresholds_train_only':[float(v) for v in thresholds],
              'classification_target_names':list(TASKS),'same_input':True}
        tensors['__meta__']=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
        bf=io.BytesIO();np.savez(bf,**tensors)
        return len(bf.getvalue())

    def member_variables(self):
        if self.method=='shared':return 0
        if self.method=='native_multihead':return K*(F*2+2)
        if self.method=='native_fastweight':return K*(F+2)
        extra=K*CODE
        if self.method=='native_learned_dictionary4':extra+=F*CODE
        if self.method in ('mirror_private1','native_linear_private1'):extra+=K*(F+2)
        return extra


def metrics(model,x,y):
    with torch.no_grad():
        logits=model(x)
        assert logits.shape==(len(x),K,2)
        losses=Fn.cross_entropy(logits.transpose(1,2),y,reduction='none') # [B,K]
        predicts=logits.argmax(-1)
        pertask=losses.mean(0).cpu().numpy()
        pacc=(predicts==y).float().mean(0).cpu().numpy()
        agreement=float((predicts[:,0,None]==predicts).float().mean())
        diversity=float(torch.std(torch.softmax(logits,dim=-1)[:,:,1],dim=1).mean())
    return {'mean_nll':float(pertask.mean()),'worst_task_nll':float(pertask.max()),'accuracy':float(pacc.mean()),
            'per_task_nll':pertask,'per_task_accuracy':pacc,'out_prob_std':diversity,'agreement_to_task0':agreement}


def benchmark(model,x):
    xx=x[:128]
    times=[]
    model.eval()
    counts={'n':0}
    def hook(_,__,___):counts['n']+=1
    with torch.no_grad():
        handle=model.trunk.register_forward_hook(hook)
        _=model(xx)
        handle.remove()
        assert counts['n']==1,(model.method,counts)
        for _ in range(20):model(xx)
        for _ in range(100):
            t=time.perf_counter_ns();model(xx);times.append((time.perf_counter_ns()-t)/1e6)
    return float(np.median(times)),float(np.quantile(times,.95)),counts['n']


def train_one(seed,method):
    trainx,trainy,testx,testy,digest,thresholds,mean,sd=world(seed)
    torch.set_num_threads(1)
    model=MultiHeadMirror(method,seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    stream=torch.Generator().manual_seed(seed*7919+41)
    for i in range(STEPS):
        idx=torch.randint(len(trainx),(BATCH,),generator=stream)
        logits=model(trainx[idx])
        loss=Fn.cross_entropy(logits.reshape(-1,2),trainy[idx].reshape(-1))
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    model.eval()
    train_stats=metrics(model,trainx,trainy)
    test_stats=metrics(model,testx,testy)
    b50,b95,calls=benchmark(model,testx)
    row={'phase':None,'seed':seed,'method':method,'K':K,'train_count':len(trainx),'test_count':len(testx),
         'train_mean_nll':train_stats['mean_nll'],'fresh_mean_nll':test_stats['mean_nll'],
         'fresh_worst_task_nll':test_stats['worst_task_nll'],
         'fresh_accuracy':test_stats['accuracy'],'out_prob_std':test_stats['out_prob_std'],
         'agreement_to_task0':test_stats['agreement_to_task0'],
         'per_task_fresh_nll':';'.join(f'{v:.9g}' for v in test_stats['per_task_nll']),
         'per_task_fresh_accuracy':';'.join(f'{v:.9g}' for v in test_stats['per_task_accuracy']),
         'serialized_npz_bytes':model.serialized_bytes(mean,sd,thresholds),'member_scalars':model.member_variables(),
         'cpu_p50_ms':b50,'cpu_p95_ms':b95,'trunk_invocations_per_forward':calls,
         'dataset_sha256':digest,'training_steps':STEPS,'input_same_across_roles':True}
    return row


def run(phase,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    allrows=[];start=time.perf_counter()
    for seed in SEEDS[phase]:
        for m in METHODS:
            r=train_one(seed,m);r['phase']=phase;allrows.append(r)
        print(json.dumps({'phase':phase,'seed':seed,'completed':len(METHODS),'nll':{r['method']:round(r['fresh_mean_nll'],4) for r in allrows if r['seed']==seed},'elapsed_s':round(time.perf_counter()-start,1)}),flush=True)
    with (out/f'{phase}_raw.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(allrows[0]));wr.writeheader();wr.writerows(allrows)
    return allrows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('dev','fresh','all'),default='all');ap.add_argument('--out',default='results');args=ap.parse_args()
    torch.set_num_threads(1)
    if args.phase in ('all','dev'):run('dev',args.out)
    if args.phase in ('all','fresh'):run('fresh',args.out)
if __name__=='__main__':main()
