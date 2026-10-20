"""Synthetic *token sequence* causal language task, not natural-language corpus.

The key-value grammar forces a nontrivial target at the last token.  Targets at
random positions are counted in the ordinary LM NLL. IDs are never reused
between train/dev/fresh/fresh_ood splits.
"""
from __future__ import annotations
import numpy as np
import torch


def corpus(world:str,split:str,seed:int,count:int):
    assert world in ('balanced','shortcut')
    assert split in ('train','dev','fresh','fresh_ood')
    offset={'train':101,'dev':202,'fresh':303,'fresh_ood':404}[split]
    rng=np.random.default_rng(117300000+seed*997+offset+(0 if world=='balanced' else 70001))
    k0=rng.integers(5,11,size=count,dtype=np.int64)
    k1=rng.integers(5,10,size=count,dtype=np.int64)
    k1=k1+(k1>=k0)
    v0=rng.integers(11,17,size=count,dtype=np.int64)
    v1=rng.integers(11,17,size=count,dtype=np.int64)
    if world=='shortcut':
        prob=0.90 if split!='fresh_ood' else 0.10
    else:prob=0.50
    first=rng.random(count)<prob
    which_key=np.where(first,k0,k1)
    which_val=np.where(first,v0,v1)
    seq=np.stack((np.zeros(count,dtype=np.int64),np.ones(count,dtype=np.int64),k0,v0,
                  np.ones(count,dtype=np.int64),k1,v1,
                  np.full(count,2,dtype=np.int64),np.full(count,4,dtype=np.int64),
                  np.full(count,3,dtype=np.int64),which_key,which_val),axis=1)
    return torch.from_numpy(seq).long()


def batch_schedule(n:int, seed:int,updates:int,batch:int):
    rng=np.random.default_rng(991731+seed*31)
    return torch.from_numpy(rng.integers(0,n,(updates+100,batch),dtype=np.int64))
