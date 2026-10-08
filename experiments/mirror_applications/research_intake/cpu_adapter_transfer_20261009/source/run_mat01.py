#!/usr/bin/env python3
"""MAT01: source-trained LoRA bank -> source-only Mirror/non-Mirror task cores.

Preregistered on GitHub under MAT01_FROZEN_PROTOCOL.json at
5d4c37a1ba70b162fa09bbe2182a4e2d206f5eac, before dev/fresh data.
An *adapter bank compression screen*, not an implementation of LoGo routing.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter

DEV=(11,12,13)
FRESH=(101,102,103,104,105)
SOURCE_ROLES=('left1','right1','up1','down1','blur065','contrast07')
TARGET_ROLES=('upper_left1','lower_right1','noise025','dropout020')
METHODS=('base','native_lora4','mirror4','linear4','diag4','fullcore16')
D,H,C,R=64,64,10,4
BASE_STEPS=320
SOURCE_STEPS=145
TARGET_STEPS=175
ACTIONS=((0,1),(2,3),(0,2),(1,3))


def world_rng(seed:int, role:str, phase:str)->np.random.Generator:
    seedtext=f'{seed}|{role}|{phase}'.encode()
    digest=int.from_bytes(hashlib.sha256(seedtext).digest()[:8],'little')
    return np.random.default_rng(digest)


def spatial_shift(x:np.ndarray, dy:int, dx:int)->np.ndarray:
    """Zero-boundary shift; never circular wrap pixels."""
    x=np.asarray(x).reshape(-1,8,8)
    y=np.zeros_like(x)
    y0,y1=max(0,dy),min(8,8+dy)
    x0,x1=max(0,-dy),min(8,8-dy)
    z0,z1=max(0,dx),min(8,8+dx)
    w0,w1=max(0,-dx),min(8,8-dx)
    y[:,y0:y1,z0:z1]=x[:,x0:x1,w0:w1]
    return y.reshape(-1,D)


def transform(x:np.ndarray,role:str,rng:np.random.Generator)->np.ndarray:
    x=np.asarray(x,dtype=np.float32)
    if role=='left1':return spatial_shift(x,0,-1)
    if role=='right1':return spatial_shift(x,0,1)
    if role=='up1':return spatial_shift(x,-1,0)
    if role=='down1':return spatial_shift(x,1,0)
    if role=='upper_left1':return spatial_shift(x,-1,-1)
    if role=='lower_right1':return spatial_shift(x,1,1)
    if role=='blur065':return gaussian_filter(x.reshape(-1,8,8),(0,.65,.65)).reshape(-1,D)
    if role=='contrast07':return np.clip(.5+.7*(x-.5),0,1)
    if role=='noise025':return np.clip(x+rng.normal(0,.25,x.shape).astype(np.float32),0,1)
    if role=='dropout020':return x*(rng.random(x.shape)>.2)
    if role=='original':return x
    raise ValueError(role)


@dataclass
class WorldData:
    x_raw:np.ndarray
    labels:np.ndarray
    train_ids:np.ndarray
    audit_ids:np.ndarray
    scaler_mean:np.ndarray
    scaler_scale:np.ndarray
    digest:str


def make_world(seed:int)->WorldData:
    digits=load_digits()
    x=np.asarray(digits.data,dtype=np.float32)/16.
    y=np.asarray(digits.target,dtype=np.int64)
    ids=np.arange(len(y))
    a,b=train_test_split(ids,test_size=.25,random_state=seed,stratify=y)
    mean=x[a].mean(axis=0).astype(np.float32)
    scale=(x[a].std(axis=0)+.15).astype(np.float32)
    dat_sha=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
    assert len(set(a)&set(b))==0
    return WorldData(x,y,np.asarray(a),np.asarray(b),mean,scale,dat_sha)


def role_batch(w:WorldData, ids:np.ndarray, role:str, seed:int, phase:str)->torch.Tensor:
    raw=w.x_raw[ids]
    converted=transform(raw,role,world_rng(seed,role,phase)).astype(np.float32)
    normalized=np.asarray((converted-w.scaler_mean[None,:])/w.scaler_scale[None,:],dtype=np.float32)
    return torch.from_numpy(normalized)


class FrozenBackbone(nn.Module):
    def __init__(self,seed:int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.up=nn.Linear(D,H)
            self.down=nn.Linear(H,C)
    def features(self,x):
        return F.gelu(self.up(x))
    def forward(self,x):
        return self.down(self.features(x))


def fit_base(w:WorldData,seed:int):
    torch.manual_seed(seed*1009+3)
    model=FrozenBackbone(seed*1123+5)
    train=role_batch(w,w.train_ids,'original',seed,'base_train')
    labels=torch.from_numpy(w.labels[w.train_ids])
    optim=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
    stream=torch.Generator().manual_seed(seed*701+7)
    model.train()
    for _ in range(BASE_STEPS):
        idx=torch.randint(len(labels),(128,),generator=stream)
        loss=F.cross_entropy(model(train[idx]),labels[idx])
        optim.zero_grad(set_to_none=True);loss.backward();optim.step()
    model.eval()
    for param in model.parameters():param.requires_grad_(False)
    return model


def native_delta_fit(hidden:torch.Tensor,labels:torch.Tensor,base_logits:torch.Tensor,
                     seed:int,steps:int,learning_rate:float=.01)->tuple[torch.Tensor,torch.Tensor]:
    """Train only a rank-4 LoRA projection from pretrained hidden state."""
    gen=torch.Generator().manual_seed(seed*1949+139)
    A=nn.Parameter(.02*torch.randn(H,R,generator=gen))
    B=nn.Parameter(torch.zeros(R,C))
    optim=torch.optim.AdamW((A,B),lr=learning_rate,weight_decay=.0001)
    stream=torch.Generator().manual_seed(seed*7919+269)
    for _ in range(steps):
        ix=torch.randint(len(labels),(128 if steps==SOURCE_STEPS else 96,),generator=stream)
        logits=base_logits[ix]+hidden[ix]@A@B
        loss=F.cross_entropy(logits,labels[ix])
        optim.zero_grad(set_to_none=True);loss.backward();optim.step()
    return A.detach().clone(), B.detach().clone()


def source_family(base:FrozenBackbone,w:WorldData,seed:int):
    source=[]
    for j,role in enumerate(SOURCE_ROLES):
        src=role_batch(w,w.train_ids,role,seed,'source_train')
        with torch.no_grad():
            h=base.features(src); logits=base.down(h)
        a,b=native_delta_fit(h,torch.from_numpy(w.labels[w.train_ids]),logits,
                             seed*131+j+1,SOURCE_STEPS,.01)
        source.append((a@b).detach().numpy().astype(np.float64))
    return np.stack(source)


def fit_source_basis(deltas:np.ndarray):
    assert deltas.shape==(6,H,C)
    ul,_,_=np.linalg.svd(np.concatenate(list(deltas),axis=1),full_matrices=False)
    _,_,vr=np.linalg.svd(np.concatenate(list(deltas),axis=0),full_matrices=False)
    U=ul[:,:R].copy()
    V=vr[:R,:].copy()
    cores=np.stack([U.T@delta@V.T for delta in deltas])
    common=cores.mean(axis=0)
    centered=(cores-common[None,:,:]).reshape(6,R*R)
    _,_,vh=np.linalg.svd(centered,full_matrices=False)
    dictionary=vh[:R].reshape(R,R,R)
    for r in range(R):
        first=np.argmax(abs(dictionary[r]).reshape(-1))
        if dictionary[r].reshape(-1)[first]<0:dictionary[r]*=-1
    return (torch.tensor(U,dtype=torch.float32),torch.tensor(V,dtype=torch.float32),
            torch.tensor(common,dtype=torch.float32),torch.tensor(dictionary,dtype=torch.float32))


def rotate4(angles:torch.Tensor)->torch.Tensor:
    """Differentiable 4x4 Givens product. Inputs are angular coordinates, not gaugified output."""
    Q=torch.eye(R,dtype=angles.dtype,device=angles.device)
    for p,(a,b) in enumerate(ACTIONS):
        c,s=torch.cos(angles[p]),torch.sin(angles[p])
        # Avoid inplace mutation on differentiable output; construct each G from rows.
        G=Q.new_zeros((R,R))
        entries=[]
        for i in range(R):
            row=[]
            for j in range(R):
                if i==a and j==a:val=c
                elif i==b and j==b:val=c
                elif i==a and j==b:val=-s
                elif i==b and j==a:val=s
                else:val=torch.tensor(float(i==j),dtype=angles.dtype,device=angles.device)
                row.append(val)
            entries.append(torch.stack(row))
        G=torch.stack(entries)
        Q=Q@G
    return Q


def core_from_code(method:str, code:torch.Tensor, C0:torch.Tensor, dictionary:torch.Tensor):
    if method=='mirror4':
        rot=rotate4(code);return rot@C0@rot.T
    if method=='linear4':
        return C0+torch.einsum('k,kij->ij',code,dictionary)
    if method=='diag4':
        return C0+torch.diag(code)
    if method=='fullcore16':
        return C0+code.reshape(R,R)
    raise ValueError(method)


def target_support(w:WorldData,seed:int):
    # Shared target support ORIGINAL IDs, so no method receives extra labels.
    support,_=train_test_split(w.train_ids,train_size=300,stratify=w.labels[w.train_ids],
                               random_state=seed*13+7)
    return np.asarray(support)


def train_target(method:str,role:str,role_id:int,seed:int,model:FrozenBackbone,w:WorldData,
                 basis:tuple[torch.Tensor,...],support_ids:np.ndarray):
    xs=role_batch(w,support_ids,role,seed,'target_train')
    xe=role_batch(w,w.audit_ids,role,seed,'target_audit')
    y=torch.from_numpy(w.labels[support_ids]);ye=torch.from_numpy(w.labels[w.audit_ids])
    with torch.no_grad():
        h=model.features(xs);raw=model.down(h)
        he=model.features(xe);rawe=model.down(he)
    if method=='base':
        with torch.no_grad():nll=F.cross_entropy(rawe,ye).item();acc=(rawe.argmax(-1)==ye).float().mean().item()
        return {'nll':nll,'acc':acc,'code':None,'a':None,'b':None},he,rawe
    if method=='native_lora4':
        a,b=native_delta_fit(h,y,raw,seed*191+role_id,TARGET_STEPS,learning_rate=.045)
        with torch.no_grad():outs=rawe+he@a@b
        code=None
    else:
        U,V,C0,dictn=basis
        n=R*R if method=='fullcore16' else R
        init=torch.zeros(n)
        if method=='mirror4':
            init=init+torch.tensor([.002,-.002,.003,-.001])
        code=nn.Parameter(init)
        optim=torch.optim.AdamW((code,),lr=.045,weight_decay=.0001)
        stream=torch.Generator().manual_seed((seed*191+role_id)*7919+269)
        for _ in range(TARGET_STEPS):
            idx=torch.randint(len(y),(96,),generator=stream)
            core=core_from_code(method,code,C0,dictn)
            logits=raw[idx]+h[idx]@U@core@V
            loss=F.cross_entropy(logits,y[idx])
            optim.zero_grad(set_to_none=True);loss.backward();optim.step()
        with torch.no_grad():outs=rawe+he@U@core_from_code(method,code,C0,dictn)@V
        code=code.detach().clone();a=b=None
    nll=F.cross_entropy(outs,ye).item()
    acc=(outs.argmax(-1)==ye).float().mean().item()
    return {'nll':nll,'acc':acc,'code':code,'a':a if method=='native_lora4' else None,
            'b':b if method=='native_lora4' else None},he,rawe


def inference_arrays(base:FrozenBackbone,w:WorldData,method:str,basis:tuple[torch.Tensor,...],
                     results:list[dict])->dict[str,np.ndarray]:
    arrays={f'base_{n}':p.detach().numpy().astype(np.float32) for n,p in base.state_dict().items()}
    arrays['scaler_mean']=w.scaler_mean;arrays['scaler_scale']=w.scaler_scale
    if method=='native_lora4':
        for j,res in enumerate(results):
            arrays[f'task_{j}_A']=res['a'].numpy().astype(np.float32)
            arrays[f'task_{j}_B']=res['b'].numpy().astype(np.float32)
    elif method not in ('base',):
        U,V,C0,dictn=basis
        arrays['source_U']=U.numpy();arrays['source_V']=V.numpy();arrays['source_C0']=C0.numpy()
        if method=='linear4':arrays['source_dictionary']=dictn.numpy()
        for j,res in enumerate(results):arrays[f'role_{j}_code']=res['code'].numpy()
    meta={'method':method,'K':len(TARGET_ROLES),'source_lo_ra_rank':R,'role_names':list(TARGET_ROLES),
          'base_dim':D,'hidden':H,'output':C,'source_roles':list(SOURCE_ROLES),
          'numpy_dtype':'float32','angles_order':ACTIONS if method=='mirror4' else [],'preproc':'train_only'}
    arrays['__metadata__']=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode('utf-8'),dtype=np.uint8)
    return arrays


def actual_npz_bytes(named:dict[str,np.ndarray])->int:
    by=io.BytesIO();np.savez(by,**named);return len(by.getvalue())


def benchmark(base:FrozenBackbone,w:WorldData,method:str,basis,results:list[dict],seed:int):
    inp=role_batch(w,w.audit_ids[:128],'original',seed,'runtime_fixture')
    U,V,C0,dictn=basis
    codes=[r['code'] for r in results]
    aa=[r['a'] for r in results];bb=[r['b'] for r in results]
    def forward():
        h=base.features(inp);q=base.down(h)
        outs=[]
        if method=='base':return q.unsqueeze(1).expand(-1,len(TARGET_ROLES),-1)
        for t in range(len(TARGET_ROLES)):
            if method=='native_lora4':d=aa[t]@bb[t]
            else:d=U@core_from_code(method,codes[t],C0,dictn)@V
            outs.append(q+h@d)
        return torch.stack(outs,dim=1)
    with torch.no_grad():
        for _ in range(20):forward()
        times=[]
        for _ in range(100):
            st=time.perf_counter_ns();forward();times.append((time.perf_counter_ns()-st)/1e6)
    return float(np.median(times)),float(np.quantile(times,.95))


def run_one(seed:int,phase:str):
    w=make_world(seed)
    base=fit_base(w,seed)
    deltas=source_family(base,w,seed)
    basis=fit_source_basis(deltas)
    ids=target_support(w,seed)
    rows=[]
    base_tensor_sha=hashlib.sha256(b''.join(v.numpy().tobytes() for v in basis)).hexdigest()
    for method in METHODS:
        outs=[]
        for j,role in enumerate(TARGET_ROLES):
            result,_,_=train_target(method,role,j,seed,base,w,basis,ids)
            outs.append(result)
            rows.append({'phase':phase,'seed':seed,'role':role,'method':method,
                         'nll':result['nll'],'accuracy':result['acc'],
                         'base_dataset_sha256':w.digest,'source_svd_sha256':base_tensor_sha,
                         'support_size':len(ids),'audit_size':len(w.audit_ids),
                         'source_lora_steps':SOURCE_STEPS,'target_steps':0 if method=='base' else TARGET_STEPS})
        bytes_npz=actual_npz_bytes(inference_arrays(base,w,method,basis,outs))
        p50,p95=benchmark(base,w,method,basis,outs,seed)
        for r in rows[-len(TARGET_ROLES):]:
            r.update(serialized_npz_bytes=bytes_npz,cpu_p50_ms=p50,cpu_p95_ms=p95)
    assert len(rows)==len(METHODS)*len(TARGET_ROLES)
    return rows


def run(phase:str,out:Path):
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    seeds=DEV if phase=='dev' else FRESH
    rows=[];start=time.perf_counter()
    for seed in seeds:
        new=run_one(seed,phase);rows+=new
        sm={r['method']:round(float(np.mean([x['nll'] for x in new if x['method']==r['method']])),5) for r in new}
        print(json.dumps({'phase':phase,'seed':seed,'rows':len(new),'mean_nll':sm,'elapsed_s':round(time.perf_counter()-start,1)},ensure_ascii=False),flush=True)
    with (out/f'{phase}_raw.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['dev','fresh','all'],default='all')
    parser.add_argument('--out',default='results')
    a=parser.parse_args();out=Path(a.out)
    if a.phase in ('dev','all'):run('dev',out)
    if a.phase in ('fresh','all'):run('fresh',out)

if __name__=='__main__':main()
