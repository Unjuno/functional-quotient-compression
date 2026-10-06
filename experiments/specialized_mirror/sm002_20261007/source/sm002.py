"""SM002 byte-capped, router-free conditional MLP controls.

SM02 fixes rank=2 for residual-linear experts, two hidden layers, four tasks.
All tensors are FP32. Fixed feature/gain/shear buffers are regenerated from paid
config and the versioned decoder, never from task labels. Low-rank bases AND the
shared trunk are trained: this is NOT a frozen-base LoRA replication.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import math
import struct
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import sm001_vendor as old

METHODS={0:'dense',1:'mirror',2:'fixed_gate',3:'film',4:'lowrank2',5:'shared_heads',
         6:'independent_small',7:'fixed_view',8:'input_partition',9:'independent_full',10:'private_ffn'}
PRIMARY=[0,1,2,3,4,5,10,6]
CAP=44155
canonical=old.canonical
sha=old.sha
World=old.World
stream=old.stream
coverage=old.coverage

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
    protocol:str='SM02-splitmix64-v1-gelu-tanh'
    def __post_init__(self):
        if self.method not in METHODS or not 2<=self.width<=256:raise ValueError('invalid architecture')
        if self.method in (1,7,8) and self.width%2:raise ValueError('shear requires even width')
        if not math.isfinite(self.rho) or abs(self.rho)>4:raise ValueError('invalid transform strength')
        if (self.tasks,self.values,self.common,self.max_private)!=(4,32,4,32):raise ValueError('fixed SM02 alphabet')
        if self.protocol!='SM02-splitmix64-v1-gelu-tanh':raise ValueError('unknown decoder protocol')
    @property
    def input_dim(self):return 72

def task_linear(x:torch.Tensor,w:torch.Tensor,b:torch.Tensor|None,m:torch.Tensor,balanced:bool):
    """One addressed expert per row; use grouped bmm when rows are task-sorted."""
    if w.ndim==2:return F.linear(x,w,b)
    if balanced:
        n=len(x)//4
        out=torch.bmm(x.reshape(4,n,-1),w.transpose(1,2))
        if b is not None:out=out+(b[:,None,:] if b.ndim==2 else b)
        return out.reshape(len(x),w.shape[1])
    out=x.new_empty((len(x),w.shape[1]))
    for task in range(4):
        idx=torch.nonzero(m==task,as_tuple=True)[0]
        out[idx]=F.linear(x[idx],w[task],None if b is None else (b[task] if b.ndim==2 else b))
    return out

class Model(nn.Module):
    def __init__(self,config:Config):
        super().__init__();self.config=config;c=config;w=c.width
        base=old.Model(old.Config(width=w,method=0))
        for name,p in base.named_parameters():
            repeat=(c.method in (6,9) or (c.method==5 and name=='w3') or (c.method==10 and name in ('w2','b2')))
            self.register_parameter(name,nn.Parameter(p.detach().clone().unsqueeze(0).repeat(4,*([1]*p.ndim))) if repeat else p)
        self.register_buffer('features',old.all_features(c),persistent=False)
        if c.method in (1,7,8):self.shears=nn.ModuleList([old.Shear(w,i,c.generator_seed,c.rho) for i in range(2)])
        if c.method==2:
            values=[]
            for layer in range(2):
                vv=[]
                for task in range(4):
                    rng=old.SplitMix64(c.generator_seed+1009*layer+104729*task)
                    vv.append([math.exp(c.rho*(1 if rng.next()%2 else -1)) for _ in range(w)])
                values.append(vv)
            self.register_buffer('gates',torch.tensor(values,dtype=torch.float32),persistent=False)
        if c.method==3:
            self.gamma=nn.Parameter(torch.ones(2,4,w));self.beta=nn.Parameter(torch.zeros(2,4,w))
        if c.method==4:
            for i,dim in ((1,72),(2,w)):
                a=torch.empty(4,2,dim);nn.init.kaiming_uniform_(a,a=math.sqrt(5))
                self.register_parameter(f'a{i}',nn.Parameter(a))
                self.register_parameter(f'd{i}',nn.Parameter(torch.zeros(4,w,2)))
    def copy_parent(self,parent:nn.Module):
        with torch.no_grad():
            for name,p in parent.state_dict().items():
                q=getattr(self,name)
                if q.shape==p.shape:q.copy_(p)
                elif q.shape[1:]==p.shape:q.copy_(p.unsqueeze(0).expand_as(q))
                else:raise ValueError(f'parent dimension mismatch {name}')
    def view_ids(self,rows):
        if self.config.method==7:return torch.zeros_like(rows[:,0])
        if self.config.method==8:return (rows[:,1]+rows[:,2])%4
        return rows[:,0]
    def forward(self,rows:torch.Tensor,view:torch.Tensor|None=None):
        m=self.view_ids(rows) if view is None else view
        idx=(rows[:,0]*36+rows[:,1])*32+rows[:,2]
        x=self.features[idx]
        needs_group=self.config.method in (4,5,6,9,10)
        balanced=(needs_group and len(rows)%4==0 and torch.equal(m,torch.arange(4,device=m.device).repeat_interleave(len(rows)//4)))
        h=task_linear(x,self.w1,self.b1,m,balanced)
        if self.config.method==4:
            h=h+task_linear(task_linear(x,self.a1,None,m,balanced),self.d1,None,m,balanced)
        h=self.activation(h,m,0)
        x=h;h=task_linear(x,self.w2,self.b2,m,balanced)
        if self.config.method==4:
            h=h+task_linear(task_linear(x,self.a2,None,m,balanced),self.d2,None,m,balanced)
        h=self.activation(h,m,1)
        return task_linear(h,self.w3,self.b3,m,balanced)
    def activation(self,h,m,layer):
        c=self.config
        if c.method in (1,7,8):return self.shears[layer](h,m)
        h=F.gelu(h,approximate='tanh')
        if c.method==2:return h*self.gates[layer,m]
        if c.method==3:return h*self.gamma[layer,m]+self.beta[layer,m]
        return h

def encode(model:Model)->bytes:
    parts=[];manifest=[]
    for key,t in sorted(model.state_dict().items()):
        if t.dtype!=torch.float32:raise ValueError('SM02 requires FP32')
        a=t.detach().cpu().numpy().astype('<f4',copy=False)
        parts.append(a.tobytes());manifest.append({'key':key,'shape':list(a.shape),'dtype':'<f4','bytes':a.nbytes})
    h=canonical({'config':asdict(model.config),'tensors':manifest})
    return b'SM02'+struct.pack('<Q',len(h))+h+b''.join(parts)

def decode(data:bytes)->Model:
    if len(data)<12 or data[:4]!=b'SM02':raise ValueError('not SM02')
    n=struct.unpack('<Q',data[4:12])[0]
    if n>100000 or n+12>len(data):raise ValueError('invalid header')
    h=json.loads(data[12:12+n]);m=Model(Config(**h['config']));state={};pos=12+n
    expected=m.state_dict()
    for e in h['tensors']:
        k=e['key'];s=e['shape'];size=e['bytes']
        if k in state or k not in expected or tuple(s)!=tuple(expected[k].shape) or e['dtype']!='<f4' or size!=4*math.prod(s) or pos+size>len(data):raise ValueError('bad tensor schema')
        state[k]=torch.from_numpy(np.frombuffer(data[pos:pos+size],dtype='<f4').copy().reshape(s));pos+=size
    if pos!=len(data) or set(state)!=set(expected):raise ValueError('payload length or keys')
    m.load_state_dict(state,strict=True);return m

def fit_config(method:int,cap:int=CAP)->Config:
    for w in range(64,1,-1):
        if method in (1,7,8) and w%2:continue
        cfg=Config(width=w,method=method,rho=.5 if method==2 else 1.)
        if len(encode(Model(cfg)))<=cap:return cfg
    raise ValueError('budget too small')

def active_macs(c:Config)->int:
    w=c.width
    # Linear MAC estimate; separately disclose elementwise / indexing / backward.
    n=72*w+w*w+32*w
    if c.method==4:n+=2*(72+3*w)
    return n

@torch.no_grad()
def evaluate(model:Model,world:World,n_private:int,wrong_views:bool=False):
    out=old.evaluate(model,world,n_private,wrong_views=False)
    if wrong_views and n_private and model.config.method not in (0,7,8):
        rows,y=world.all_rows(n_private);matrix=[]
        for v in range(4):
            logits=model(rows,view=torch.full_like(rows[:,0],v))
            ok=(logits.argmax(-1)==y).reshape(4,4+n_private,32)
            matrix.append(ok[:,4:].float().mean((1,2)).tolist())
        a=np.array(matrix).T
        out['task_by_view_private_accuracy']=a.tolist()
        out['assigned_diag_minus_offdiag']=float(np.diag(a).mean()-a[~np.eye(4,dtype=bool)].mean())
        out['assigned_better_than_each_wrong_task_count']=int(sum(a[t,t]>max(np.delete(a[t],t)) for t in range(4)))
    return out
