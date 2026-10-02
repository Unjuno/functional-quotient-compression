# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Finite conditional-permutation task. Novel sequences, NOT novel rule-symbol pairs."""
from __future__ import annotations
import hashlib
import torch

def make_table(rules: int=16, symbols: int=32, seed: int=271828) -> torch.Tensor:
    if min(rules,symbols) < 1:
        raise ValueError('rules and symbols must be positive')
    rng=torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(symbols,generator=rng) for _ in range(rules)])

def dataset(n: int, seed: int, table: torch.Tensor, records: int=4):
    if min(n,records)<1:
        raise ValueError('n and records must be positive')
    if table.ndim!=2 or table.dtype!=torch.long or table.numel()==0:
        raise ValueError('table must be nonempty 2-D int64')
    rules,symbols=table.shape
    if (table<0).any() or (table>=symbols).any():
        raise ValueError('table values out of range')
    rng=torch.Generator().manual_seed(seed)
    modes=torch.randint(rules,(n,records),generator=rng)
    inputs=torch.randint(symbols,(n,records),generator=rng)
    seq=torch.empty(n,1+3*records,dtype=torch.long)
    seq[:,0]=rules+symbols
    seq[:,1::3]=modes+symbols
    seq[:,2::3]=inputs
    seq[:,3::3]=table[modes,inputs]
    x,y=seq[:,:-1].clone(),seq[:,1:].clone()
    y[:,torch.arange(y.shape[1])%3!=2]=-100
    return x,y

def tensor_hash(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def check_splits(train, dev, audit=None) -> dict:
    datasets=[train,dev]+([] if audit is None else [audit])
    sets=[{tuple(r.tolist()) for r in x} for x,y in datasets]
    overlaps=sum(len(sets[i]&sets[j]) for i in range(len(sets)) for j in range(i+1,len(sets)))
    if overlaps:
        raise ValueError(f'{overlaps} overlapping sequences')
    return {'overlap_sequences':overlaps,'unique_sequences':[len(s) for s in sets],
            'hashes':[{'x':tensor_hash(x),'y':tensor_hash(y)} for x,y in datasets]}
