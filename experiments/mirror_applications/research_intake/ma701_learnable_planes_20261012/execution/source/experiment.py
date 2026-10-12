"""MA701-LP01 Phase A: joint shared-plane learning and native controls.
Synthetic teachers are positive/negative fixtures, not pretrained-model results.
"""
from __future__ import annotations
import json,time,hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from torch import nn,Tensor
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from .planes import apply_planes

METHODS=['shared','fixed','learned','native4','film','lora2','householder','multihead','private_ffn']
FAMILIES=['aligned','partial','unrelated','digits8','digits4']
FRESH=set(range(17111,17116))

class Model(nn.Module):
    def __init__(self,method:str,di:int,o:int,k:int,seed:int,sigma:float,digits:bool):
        super().__init__()
        if method not in METHODS:raise ValueError(method)
        self.spec=dict(method=method,di=di,o=o,k=k,seed=seed,sigma=sigma,digits=digits)
        self.method=method;self.k=k;self.d=32;self.o=o
        with torch.random.fork_rng():
            torch.manual_seed(seed+88000)
            self.trunk=nn.Sequential(nn.Linear(64,48),nn.GELU(),nn.Linear(48,32),nn.GELU()) if digits else nn.Identity()
        g=torch.Generator().manual_seed(seed+99000)
        def rand(*shape):return torch.randn(*shape,generator=g)
        self.up=nn.Parameter(rand(32,di)/di**.5)
        self.down=nn.Parameter(rand(32,32)/32**.5)
        head=rand(o,32)/32**.5
        self.bias=nn.Parameter(torch.zeros(k,o))
        if method in ('multihead','private_ffn'):
            self.task_head=nn.Parameter(head.unsqueeze(0).repeat(k,1,1))
        else:self.head=nn.Parameter(head)
        if method in ('fixed','learned'):
            a=torch.zeros(4,32);b=torch.zeros(4,32)
            for j in range(4):a[j,2*j]=1;b[j,2*j+1]=1
            self.a=nn.Parameter(a,requires_grad=method=='learned')
            self.b=nn.Parameter(b,requires_grad=method=='learned')
            self.code=nn.Parameter(rand(k,4)*sigma)
        elif method=='native4':
            self.basis=nn.Parameter(rand(4,32)/2)
            self.code=nn.Parameter(rand(k,4)*sigma)
        elif method=='film':self.affine=nn.Parameter(rand(k,2,32)*sigma)
        elif method=='lora2':
            self.lo_a=nn.Parameter(rand(k,2,di)/di**.5)
            self.lo_b=nn.Parameter(rand(k,32,2)*sigma)
        elif method=='householder':
            n=rand(k,1,32)
            self.normals=nn.Parameter(torch.cat([n,n+sigma*rand(k,1,32)],1))
        elif method=='private_ffn':self.delta_up=nn.Parameter(torch.zeros(k,32,di))

    def pick(self,v:Tensor,tid:Tensor|None)->Tensor:
        return v[tid].unsqueeze(1) if tid is not None else v.unsqueeze(0)

    def forward(self,x:Tensor,tid:Tensor|None=None)->Tensor:
        h=self.trunk(x);z=(h@self.up.T).unsqueeze(1)
        z=z.expand(-1,1 if tid is not None else self.k,-1)
        if self.method in ('fixed','learned'):
            z=apply_planes(z,self.a,self.b,self.pick(self.code,tid))
        elif self.method=='native4':z=z*(1+self.pick(self.code@self.basis,tid))
        elif self.method=='film':
            af=self.pick(self.affine,tid);z=z*(1+af[...,0,:])+af[...,1,:]
        elif self.method=='lora2':
            a=self.pick(self.lo_a,tid);b=self.pick(self.lo_b,tid)
            t=torch.einsum('bi,bkri->bkr',h,a.expand(h.shape[0],-1,-1,-1))
            z=z+torch.einsum('bkr,bkdr->bkd',t,b.expand(h.shape[0],-1,-1,-1))
        elif self.method=='householder':
            normals=self.pick(self.normals,tid)
            for j in range(2):
                n=F.normalize(normals[...,j,:],dim=-1)
                z=z-2*(z*n).sum(-1,keepdim=True)*n
        elif self.method=='private_ffn':
            delta=self.pick(self.delta_up,tid)
            z=z+torch.einsum('bi,bkdi->bkd',h,delta.expand(h.shape[0],-1,-1,-1))
        z=F.gelu(z)@self.down.T
        if self.method in ('multihead','private_ffn'):
            out=torch.einsum('bkd,bkod->bko',z,self.pick(self.task_head,tid).expand(z.shape[0],-1,-1,-1))
        else:out=z@self.head.T
        out=out+self.pick(self.bias,tid)
        return out[:,0] if tid is not None else out

@lru_cache(maxsize=64)
def make_world(seed:int,family:str,allow_fresh:bool=False)->dict:
    if seed in FRESH and not allow_fresh:raise ValueError('fresh seed is sealed')
    if family not in FAMILIES:raise ValueError(family)
    if family.startswith('digits'):
        raw=load_digits();labels=raw.target
        ids=np.arange(len(labels));tr,rest=train_test_split(ids,test_size=.4,stratify=labels,random_state=seed)
        va,te=train_test_split(rest,test_size=.5,stratify=labels[rest],random_state=seed+1)
        y=np.stack([labels%2==0,labels>=5,np.isin(labels,[2,3,5,7]),labels%3==0,np.isin(labels,[0,1,2]),np.isin(labels,[3,4,5]),np.isin(labels,[6,7,8]),np.isin(labels,[1,4,9])],1)
        if family=='digits4':y=np.stack([labels%2==0,labels>=5,labels%2!=0,labels<5],1)
        w=dict(seed=seed,family=family,di=32,o=1,k=y.shape[1],digits=True)
        for split,ix in [('train',tr),('val',va),('test',te)]:
            w['x_'+split]=torch.tensor(raw.data[ix]/16,dtype=torch.float32)
            w['y_'+split]=torch.tensor(y[ix,:,None],dtype=torch.float32)
            w[split+'_ids']=ix.tolist()
        w['scale']=None
        return w
    k=12;di=24;o=6
    teacher=Model('learned',di,o,k,seed+300000,.03,False)
    g=torch.Generator().manual_seed(seed+400000)
    with torch.no_grad():
        teacher.a.copy_(torch.randn(4,32,generator=g));teacher.b.copy_(torch.randn(4,32,generator=g))
        teacher.code.copy_(torch.randn(k,4,generator=g)*.9)
    tu=torch.randn(k,32,di,generator=g)/di**.5
    th=torch.randn(k,o,32,generator=g)/32**.5
    def labels(x,tid):
        with torch.no_grad():
            shared=teacher(x,tid)
            independent=torch.einsum('boh,bh->bo',th[tid],F.gelu(torch.einsum('bhi,bi->bh',tu[tid],x)))
            if family=='partial':shared[:,-1]=independent[:,-1]
            elif family=='unrelated':shared=independent
            return shared
    w=dict(seed=seed,family=family,di=di,o=o,k=k,digits=False)
    for split,n,offset in [('train',128,0),('val',128,10000),('test',512,20000)]:
        rg=torch.Generator().manual_seed(seed+500000+offset)
        x=torch.randn(k*n,di,generator=rg);tid=torch.arange(k).repeat_interleave(n)
        w['x_'+split]=x;w['tid_'+split]=tid;w['y_'+split]=labels(x,tid)
        w[split+'_ids']=list(range(offset,offset+k*n))
    center=w['y_train'].mean(0);scale=w['y_train'].std(0).clamp_min(1e-4)
    for split in ['train','val','test']:w['y_'+split]=(w['y_'+split]-center)/scale
    w['scale']=w['y_train'].reshape(k,128,o).square().mean((1,2)).clamp_min(1e-8)
    return w

@torch.no_grad()
def metrics(model:Model,w:dict,split:str='test')->dict:
    model.eval();x=w['x_'+split];y=w['y_'+split]
    out=[]
    for start in range(0,len(x),256):
        tid=None if w['digits'] else w['tid_'+split][start:start+256]
        out.append(model(x[start:start+256],tid))
    z=torch.cat(out)
    if w['digits']:
        loss=F.binary_cross_entropy_with_logits(z,y,reduction='none').mean((0,2))
        acc=((z>=0)==(y>=.5)).float().mean((0,2))
        return dict(loss=float(loss.mean()),worst=float(loss.max()),accuracy=float(acc.mean()),per_task=loss.tolist())
    loss=(z-y).square().reshape(w['k'],-1,w['o']).mean((1,2))/w['scale']
    return dict(loss=float(loss.mean()),worst=float(loss.max()),accuracy=None,per_task=loss.tolist())

def fit(model:Model,w:dict,cfg:dict,seed:int,checkpoints:tuple=(800,2400))->dict:
    torch.set_num_threads(1);model.train()
    axis=[model.a,model.b] if model.method=='learned' else []
    base=[p for p in model.parameters() if p.requires_grad and not any(p is q for q in axis)]
    groups=[dict(params=base,lr=cfg['lr'])]
    if axis:groups.append(dict(params=axis,lr=cfg['axis_lr'],weight_decay=0.))
    opt=torch.optim.AdamW(groups,weight_decay=1e-4)
    rng=torch.Generator().manual_seed(seed+600000)
    snapshots={};snapshot_times={};trace=[];begin=time.perf_counter();batch=128 if w['digits'] else 64
    for step in range(cfg['steps']):
        ix=torch.randint(len(w['x_train']),(batch,),generator=rng)
        x=w['x_train'][ix];y=w['y_train'][ix];tid=None if w['digits'] else w['tid_train'][ix]
        opt.zero_grad(set_to_none=True)
        pred=model(x,tid)
        loss=F.binary_cross_entropy_with_logits(pred,y) if w['digits'] else (pred-y).square().mean()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite loss')
        loss.backward()
        if axis and step<cfg['warmup']:
            for p in axis:p.grad=None
        opt.step()
        if step==0 or (step+1)%100==0:trace.append([step+1,float(loss.detach())])
        if step+1 in checkpoints:
            snapshots[step+1]={k:v.detach().clone() for k,v in model.state_dict().items()}
            snapshot_times[step+1]=time.perf_counter()-begin
    return dict(train_seconds=time.perf_counter()-begin,trace=trace,snapshots=snapshots,snapshot_times=snapshot_times)

def save_model(model:Model,path:Path)->dict:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    meta=json.dumps(model.spec,sort_keys=True,separators=(',',':')).encode()
    data={'_meta':np.frombuffer(meta,dtype=np.uint8)}
    data.update({k:v.detach().cpu().numpy() for k,v in sorted(model.state_dict().items())})
    np.savez(path,**data)
    return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def load_model(path:Path)->Model:
    with np.load(path,allow_pickle=False) as data:
        spec=json.loads(data['_meta'].tobytes());m=Model(**spec)
        m.load_state_dict({k:torch.from_numpy(data[k].copy()) for k in data.files if k!='_meta'})
    return m.eval()
