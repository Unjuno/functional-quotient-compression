"""Strict train/dev/fresh/OOD disjoint synthetic teacher datasets."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import new_model, D, EDGES

SPLIT_OFFSETS={'train':1101,'dev':2203,'fresh':3307,'fresh_ood':4409}
WORLD_OFFSETS={'aligned_sparse_mirror':100001,'independent_native_mlp':200003}

@torch.no_grad()
def teacher(world,seed):
    if world=='aligned_sparse_mirror':
        m=new_model('mirror',seed+44001)
        m.core.copy_(0.7*torch.randn(m.core.shape,generator=torch.Generator().manual_seed(seed+741)))
        m.address.copy_(1.2*torch.randn(m.address.shape,generator=torch.Generator().manual_seed(seed+743)))
        m.gates.fill_(1.2)
        rng=np.random.default_rng(seed+745)
        active=sorted(rng.choice(EDGES,size=8,replace=False).tolist())
        m.active.fill_(False);m.active[active]=True
        return m
    if world=='independent_native_mlp':
        m=new_model('native',seed+44003)
        return m
    raise ValueError(world)

@torch.no_grad()
def dataset(world,split,seed,count):
    assert split in SPLIT_OFFSETS
    rng=np.random.default_rng(WORLD_OFFSETS[world]+seed*997+SPLIT_OFFSETS[split])
    sigma=1.35 if split=='fresh_ood' else .9
    arr=rng.standard_normal((count,D),dtype=np.float32)*sigma
    if split=='fresh_ood':
        arr[:,1:]=0.8*arr[:,1:]+0.2*arr[:,:-1]
    x=torch.tensor(arr,dtype=torch.float32)
    model=teacher(world,seed)
    y=model(x).detach()
    return x,y

def batch_schedule(n,seed,updates,batch):
    rng=np.random.default_rng(552411+seed*401)
    return torch.tensor(rng.integers(0,n,size=(updates+100,batch),dtype=np.int64),dtype=torch.int64)