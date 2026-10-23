"""Synthetic function teachers and disjoint split generation for MA-1176."""
from __future__ import annotations
import math
import torch
from models import D, RULES, householder

TEACHER_WORLDS = ('aligned_mirror_teacher', 'independent_mlp_teacher')


def teacher_parameters(world):
    assert world in TEACHER_WORLDS
    g = torch.Generator().manual_seed(77313 if world == TEACHER_WORLDS[0] else 77329)
    if world == TEACHER_WORLDS[0]:
        return {'up': torch.randn(8, D, generator=g) * 0.42,
                'down': torch.randn(D, 8, generator=g) * 0.30,
                'code': torch.randn(RULES, 8, generator=g)}
    return {'up': torch.randn(RULES, 8, D, generator=g) * 0.42,
            'down': torch.randn(RULES, D, 8, generator=g) * 0.30}


def teacher_output(x, rule, world):
    p = teacher_parameters(world)
    if world == TEACHER_WORLDS[0]:
        a = x @ p['up'].t()
        a = householder(a, p['code'][rule])
        a = householder(torch.tanh(a), p['code'][rule])
        return a @ p['down'].t()
    a = torch.einsum('bi,bhi->bh', x, p['up'][rule])
    return torch.einsum('bh,boh->bo', torch.tanh(a), p['down'][rule])


def loop_teacher(x):
    # No train or student seed enters the fixed teacher.
    g = torch.Generator().manual_seed(77413)
    w = torch.randn(D, D, generator=g) * 0.29
    o = torch.randn(D, D, generator=g) * 0.35
    h = x
    for _ in range(3):
        h = h + 0.42 * torch.tanh(h @ w.t())
    return (h - x) @ o.t()


def generate(world, seed, split, n, regime='clean_balanced'):
    if split not in ('train','dev','fresh','ood'):
        raise ValueError(split)
    if regime not in ('clean_balanced','skew','corrupted','oracle_filtered'):
        raise ValueError(regime)
    salt = {'train': 119, 'dev': 221, 'fresh': 332, 'ood': 443}[split]
    world_salt = 5 if world == TEACHER_WORLDS[0] else 41 if world == TEACHER_WORLDS[1] else 71
    # Separate generator for each split, size does not affect dev/fresh/ood content.
    g = torch.Generator().manual_seed(123231 + seed*1003 + salt + world_salt)
    x = torch.randn(n,D,generator=g)
    if split == 'ood': x = x * 1.4 + 0.55
    p = torch.tensor([0.8,0.1,0.1]) if (regime == 'skew' and split == 'train') else torch.ones(RULES)/RULES
    rule = torch.multinomial(p,n,replacement=True,generator=g)
    y = loop_teacher(x) if world == 'loop_teacher' else teacher_output(x,rule,world)
    good = torch.ones(n,dtype=torch.bool)
    if split == 'train' and regime in ('corrupted','oracle_filtered'):
        good = torch.rand(n,generator=g) >= 0.20
        perm = torch.randperm(n,generator=g)
        y = torch.where(good[:,None],y,y[perm])
        if regime == 'oracle_filtered':
            # This uses ground-truth metadata, NOT a deployable data-cleaning algorithm.
            x,rule,y = x[good],rule[good],y[good]
    return x.float(),rule.long(),y.float()
