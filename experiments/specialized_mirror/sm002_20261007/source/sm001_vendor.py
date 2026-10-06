"""SM001 finite common/private retention pilot. CPU float32, no quantization.

Protocol SM01 pays for all inference weights and config in a canonical uncompressed
payload. Fixed one-hot encoder and splitmix64-v1 shear decoder are shared public
protocol, not data-trained state. This is an MLP/FFN pilot, not a Transformer.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import struct
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

METHODS={0:'dense',1:'shared_task',2:'assigned',3:'fixed_view',4:'independent_full',5:'independent_small'}

def canonical(obj:Any)->bytes:
    return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()

@dataclass(frozen=True)
class Config:
    width:int=64
    rho:float=1.0
    method:int=0
    tasks:int=4
    values:int=32
    common:int=4
    max_private:int=32
    generator_seed:int=481
    protocol:str='SM01-splitmix64-v1-gelu-tanh'
    def __post_init__(self):
        if self.width<2 or (self.width%2 and self.method in (1,2,3)) or self.method not in METHODS:
            raise ValueError('positive width, even Mirror width, and known method required')
        if (self.tasks,self.values,self.common,self.max_private)!=(4,32,4,32):
            raise ValueError('SM001 alphabet is frozen')
        if not math.isfinite(self.rho) or abs(self.rho)>4:raise ValueError('invalid rho')
    @property
    def input_dim(self)->int:return self.tasks+self.common+self.max_private+self.values

class SplitMix64:
    def __init__(self,seed:int):self.state=seed&((1<<64)-1)
    def next(self)->int:
        self.state=(self.state+0x9E3779B97F4A7C15)&((1<<64)-1)
        z=self.state
        z=((z^(z>>30))*0xBF58476D1CE4E5B9)&((1<<64)-1)
        z=((z^(z>>27))*0x94D049BB133111EB)&((1<<64)-1)
        return z^(z>>31)
    def shuffle(self,a:list[int])->None:
        for i in range(len(a)-1,0,-1):
            j=self.next()%(i+1);a[i],a[j]=a[j],a[i]

class Shear(nn.Module):
    """Disjoint signed source/destination pairs imply A^2=0, inverse I-rho*A."""
    def __init__(self,width:int,layer:int,seed:int,rho:float):
        super().__init__();self.rho=rho
        src=[];dst=[];sign=[]
        for view in range(4):
            rng=SplitMix64(seed+1009*layer+104729*view)
            perm=list(range(width));rng.shuffle(perm)
            src.append(perm[:width//2]);dst.append(perm[width//2:2*(width//2)])
            sign.append([1. if rng.next()%2 else -1. for _ in range(width//2)])
        self.register_buffer('src',torch.tensor(src,dtype=torch.long),persistent=False)
        self.register_buffer('dst',torch.tensor(dst,dtype=torch.long),persistent=False)
        self.register_buffer('sign',torch.tensor(sign,dtype=torch.float32),persistent=False)
    def transform(self,v:torch.Tensor,m:torch.Tensor,inverse:bool=False)->torch.Tensor:
        amount=(-self.rho if inverse else self.rho)*self.sign[m]*v.gather(1,self.src[m])
        return v.scatter_add(1,self.dst[m],amount)
    def forward(self,v:torch.Tensor,m:torch.Tensor)->torch.Tensor:
        return self.transform(F.gelu(self.transform(v,m),approximate='tanh'),m,inverse=True)

class World:
    """Full finite random permutations; no unseen random-label extrapolation."""
    def __init__(self,seed:int):
        self.seed=seed
        g=torch.Generator().manual_seed(seed)
        common=torch.stack([torch.randperm(32,generator=g) for _ in range(4)])
        self.labels=torch.empty((4,36,32),dtype=torch.long)
        self.labels[:,:4]=common
        for task in range(4):
            for rule in range(4,36):self.labels[task,rule]=torch.randperm(32,generator=g)
    def all_rows(self,n_private:int)->tuple[torch.Tensor,torch.Tensor]:
        if not 0<=n_private<=32:raise ValueError('invalid private count')
        rows=torch.cartesian_prod(torch.arange(4),torch.arange(4+n_private),torch.arange(32))
        return rows,self.targets(rows)
    def targets(self,rows:torch.Tensor)->torch.Tensor:
        return self.labels[rows[:,0],rows[:,1],rows[:,2]]
    @property
    def digest(self)->str:return sha(self.labels.numpy().astype('<i8').tobytes())

def stream(world:World,n_private:int,steps:int,batch:int,seed:int)->torch.Tensor:
    if steps<=0 or batch%8:raise ValueError('positive steps and batch multiple of 8 required')
    g=torch.Generator().manual_seed(seed)
    result=torch.empty((steps,batch,3),dtype=torch.long)
    result[:,:,0]=torch.arange(4).repeat_interleave(batch//4)
    # Each task has exactly half common and half private examples per update.
    if n_private:
        result[:,:,1]=torch.randint(4,(steps,batch),generator=g)
        mask=torch.arange(batch)%(batch//4)>=batch//8
        result[:,mask,1]=4+torch.randint(n_private,(steps,int(mask.sum())),generator=g)
    else:result[:,:,1]=torch.randint(4,(steps,batch),generator=g)
    result[:,:,2]=torch.randint(32,(steps,batch),generator=g)
    return result

def coverage(rows:torch.Tensor,n_private:int)->dict[str,Any]:
    r=rows.reshape(-1,3)
    idx=(r[:,0]*(4+n_private)+r[:,1])*32+r[:,2]
    counts=torch.bincount(idx,minlength=4*(4+n_private)*32)
    private=counts.reshape(4,4+n_private,32)[:,4:]
    return {'missing_rows':int((counts==0).sum()),'min_exposures':int(counts.min()),
            'max_exposures':int(counts.max()),'unique_rows':len(counts),
            'private_min_exposures':int(private.min()) if n_private else None}

def all_features(cfg:Config)->torch.Tensor:
    rows=torch.cartesian_prod(torch.arange(4),torch.arange(36),torch.arange(32))
    return torch.cat((F.one_hot(rows[:,0],4),F.one_hot(rows[:,1],36),F.one_hot(rows[:,2],32)),dim=1).float()

class Model(nn.Module):
    def __init__(self,config:Config):
        super().__init__();self.config=config
        w=config.width
        self.w1=nn.Parameter(torch.empty(w,config.input_dim));self.b1=nn.Parameter(torch.empty(w))
        self.w2=nn.Parameter(torch.empty(w,w));self.b2=nn.Parameter(torch.empty(w))
        self.w3=nn.Parameter(torch.empty(32,w));self.b3=nn.Parameter(torch.empty(32))
        for weight,bias in [(self.w1,self.b1),(self.w2,self.b2),(self.w3,self.b3)]:
            nn.init.kaiming_uniform_(weight,a=math.sqrt(5));nn.init.uniform_(bias,-1/math.sqrt(weight.shape[1]),1/math.sqrt(weight.shape[1]))
        self.shears=nn.ModuleList([Shear(w,i,config.generator_seed,config.rho) for i in range(2)])
        self.register_buffer('features',all_features(config),persistent=False)
    def view_ids(self,rows:torch.Tensor)->torch.Tensor:
        if self.config.method==1:return (rows[:,1]+rows[:,2])%4
        if self.config.method==2:return rows[:,0]
        return torch.zeros_like(rows[:,0])
    def forward(self,rows:torch.Tensor,view:torch.Tensor|None=None)->torch.Tensor:
        idx=(rows[:,0]*36+rows[:,1])*32+rows[:,2]
        h=F.linear(self.features[idx],self.w1,self.b1)
        m=self.view_ids(rows) if view is None else view
        if self.config.method==0:h=F.gelu(h,approximate='tanh')
        else:h=self.shears[0](h,m)
        h=F.linear(h,self.w2,self.b2)
        if self.config.method==0:h=F.gelu(h,approximate='tanh')
        else:h=self.shears[1](h,m)
        return F.linear(h,self.w3,self.b3)

class ExpertModel(nn.Module):
    """Four fully independent networks; exactly one expert evaluates each input."""
    def __init__(self,config:Config):
        super().__init__();self.config=config
        parent=Model(Config(**{**asdict(config),'method':0}))
        for name,tensor in parent.state_dict().items():
            self.register_parameter(name,nn.Parameter(tensor.unsqueeze(0).repeat(4,*([1]*tensor.ndim))))
        self.register_buffer('features',all_features(config),persistent=False)
    def copy_parent(self,parent:Model)->None:
        with torch.no_grad():
            for name,tensor in parent.state_dict().items():
                getattr(self,name).copy_(tensor.unsqueeze(0).expand_as(getattr(self,name)))
    def forward(self,rows:torch.Tensor,view:torch.Tensor|None=None)->torch.Tensor:
        task=rows[:,0] if view is None else view
        idx=(rows[:,0]*36+rows[:,1])*32+rows[:,2]
        features=self.features[idx]
        # Balanced, task-sorted training and exhaustive evaluation use grouped bmm.
        b=len(rows);n=b//4
        balanced=(b%4==0 and torch.equal(task,torch.arange(4).repeat_interleave(n)))
        if balanced:
            h=torch.bmm(features.view(4,n,-1),self.w1.transpose(1,2))+self.b1[:,None,:]
            h=F.gelu(h,approximate='tanh')
            h=F.gelu(torch.bmm(h,self.w2.transpose(1,2))+self.b2[:,None,:],approximate='tanh')
            return (torch.bmm(h,self.w3.transpose(1,2))+self.b3[:,None,:]).reshape(b,32)
        result=torch.empty((b,32),dtype=features.dtype,device=features.device)
        for t in range(4):
            mask=task==t
            h=F.gelu(F.linear(features[mask],self.w1[t],self.b1[t]),approximate='tanh')
            h=F.gelu(F.linear(h,self.w2[t],self.b2[t]),approximate='tanh')
            result[mask]=F.linear(h,self.w3[t],self.b3[t])
        return result

def encode(model:nn.Module)->bytes:
    arrays=[];manifest=[]
    for name,t in sorted(model.state_dict().items()):
        if t.dtype!=torch.float32:raise ValueError('SM01 stores only float32')
        arr=t.detach().cpu().numpy().astype('<f4',copy=False)
        arrays.append(arr.tobytes(order='C'))
        manifest.append({'key':name,'shape':list(arr.shape),'dtype':'<f4','bytes':arr.nbytes})
    header=canonical({'config':asdict(model.config),'tensors':manifest})
    return b'SM01'+struct.pack('<Q',len(header))+header+b''.join(arrays)

def decode(payload:bytes)->nn.Module:
    if payload[:4]!=b'SM01' or len(payload)<12:raise ValueError('not an SM01 payload')
    length=struct.unpack('<Q',payload[4:12])[0]
    if length>100000 or length+12>len(payload):raise ValueError('invalid header')
    header=json.loads(payload[12:12+length]);cfg=Config(**header['config'])
    model=ExpertModel(cfg) if cfg.method in (4,5) else Model(cfg)
    offset=12+length;state={}
    for entry in header['tensors']:
        n=entry['bytes'];shape=entry['shape']
        if n!=4*math.prod(shape) or offset+n>len(payload):raise ValueError('invalid tensor')
        state[entry['key']]=torch.from_numpy(np.frombuffer(payload[offset:offset+n],dtype='<f4').copy().reshape(shape))
        offset+=n
    if offset!=len(payload):raise ValueError('trailing payload')
    model.load_state_dict(state,strict=True)
    return model

def model_parameter_hash(model:nn.Module)->str:
    return sha(b''.join(t.detach().cpu().numpy().tobytes() for _,t in sorted(model.state_dict().items())))

@torch.no_grad()
def evaluate(model:nn.Module,world:World,n_private:int,wrong_views:bool=True)->dict[str,Any]:
    model.eval();rows,targets=world.all_rows(n_private)
    logits=model(rows);correct=(logits.argmax(-1)==targets).reshape(4,4+n_private,32)
    ce=F.cross_entropy(logits,targets,reduction='none').reshape(4,4+n_private,32)
    common=float(correct[:,:4].float().mean());private=correct[:,4:]
    rules=private.float().mean(-1)
    out={'common_accuracy':common,'common_nll':float(ce[:,:4].mean()),
         'private_accuracy':float(private.float().mean()) if n_private else None,
         'private_nll':float(ce[:,4:].mean()) if n_private else None,
         'worst_task_private':float(private.float().mean((1,2)).min()) if n_private else None,
         'retained_private_rules_95':int((rules>=.95).sum()) if n_private else 0,
         'retained_private_rules_100':int((rules>=1.).sum()) if n_private else 0,
         'per_task_private':private.float().mean((1,2)).tolist() if n_private else [],
         'per_rule_private_accuracy':rules.tolist(),
         'common_per_rule_accuracy':correct[:,:4].float().mean((0,2)).tolist(),
         'prediction_hash':sha(logits.argmax(-1).numpy().astype('<i8').tobytes())}
    if wrong_views and n_private and model.config.method in (1,2,3):
        matrix=[]
        for view in range(4):
            log=model(rows,view=torch.full((len(rows),),view,dtype=torch.long))
            ok=(log.argmax(-1)==targets).reshape(4,4+n_private,32)
            matrix.append(ok[:,4:].float().mean((1,2)).tolist())
        mat=np.asarray(matrix).T
        out['task_by_view_private_accuracy']=mat.tolist()
        out['assigned_diag_minus_offdiag']=float(np.diag(mat).mean()-mat[~np.eye(4,dtype=bool)].mean())
        out['assigned_better_than_each_wrong_task_count']=int(sum(mat[t,t]>max(np.delete(mat[t],t)) for t in range(4)))
    model.train();return out

def active_forward_macs(config:Config)->int:
    w=config.width
    return config.input_dim*w+w*w+32*w+(2*w if config.method in (1,2,3) else 0)
