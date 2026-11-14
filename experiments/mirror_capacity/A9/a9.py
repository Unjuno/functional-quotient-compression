# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A9 optimizer-reachability control, leveraging frozen A8 Student/serializer.

The canonical scaffold depends on the *family* (ramp) but never accesses
world-specific teacher thresholds or any audit target. It intentionally makes
a direct-shift baseline and a shear-Mirror algebraically identical at init.
"""
from __future__ import annotations
import hashlib

import torch
from torch.nn import functional as F
from experiments.mirror_capacity.A8 import a8

KINDS = ('mirror','pre','shift')
REGIMES = ('random_full4','random_code4','scaffold_full4',
           'scaffold_code4','scaffold_noisy4','random_wide')
CAP = 512


def initialize(kind: str, regime: str, world: int, seed: int) -> a8.Student:
    if kind not in KINDS or regime not in REGIMES or type(world) is not int or world < 0 or type(seed) is not int or seed < 0:
        raise ValueError('invalid kind/regime/world/seed')
    torch.manual_seed(510000+world*199+seed*797)
    m = a8.create(kind, CAP) if regime == 'random_wide' else a8.Student(kind,4)
    with torch.no_grad():
        # The same task-address initialization is used for all matched-width arms.
        gen=torch.Generator().manual_seed(129011+world*71+seed*151)
        m.code.weight.copy_((torch.rand(m.code.weight.shape,generator=gen)-0.5)*0.8)
        if regime.startswith('scaffold'):
            m.up.weight.zero_();m.up.bias.zero_()
            m.down.weight.zero_();m.down.bias.zero_()
            m.up.weight[0,0] = 1.; m.up.weight[2,0]=1.
            m.down.weight[0,0]=1.;m.down.weight[0,2]=-1.
            if kind in ('mirror','pre'):
                # Negative, inactive companion branch: (x,-1) for both pairs.
                m.up.bias[1]=-1.;m.up.bias[3]=-1.
            else:
                # Mirror's shear x + a*(-1) equals a direct hidden shift x - a.
                m.shift_w.zero_();m.shift_w[0,0]=-1.;m.shift_w[1,2]=-1.
            if regime=='scaffold_noisy4':
                noise_gen=torch.Generator().manual_seed(930017+world*523+seed*23+KINDS.index(kind)*53)
                for name,p in m.named_parameters():
                    if name=='code.weight':continue
                    p.add_(torch.randn(p.shape,generator=noise_gen)*0.1)
    if regime in ('random_code4','scaffold_code4'):
        for p in m.parameters():p.requires_grad_(False)
        m.code.weight.requires_grad_(True)
    if len(a8.serialize(m))>CAP:raise RuntimeError('budget exceeded')
    return m


def diagnostic_activity(m: a8.Student, x: torch.Tensor, task: torch.Tensor) -> float:
    with torch.no_grad():
        v=m.up(x)
        c=m.code(task)
        if m.kind in ('mirror','pre'):
            ab=c[:,torch.arange(m.width//2)%a8.RANK]
            pair=v.reshape(-1,m.width//2,2)
            z=torch.stack([pair[:,:,0]+ab*pair[:,:,1],pair[:,:,1]],dim=-1)
            return float((z>0).float().mean())
        z=v+c@m.shift_w
        return float((z>0).float().mean())


def fit(w:dict, kind:str, regime:str, seed:int, *, steps:int=1200, batch:int=256)->dict:
    if steps<=0 or batch<=0 or w['family']!='ramp':raise ValueError('invalid training request')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    m=initialize(kind,regime,int(w['world']),seed)
    x,y=w['train']
    gen=torch.Generator().manual_seed(951000+int(w['world'])*491+seed*37)
    opt=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=0.012,weight_decay=0.)
    activity=None;first_code_grad=None
    m.train()
    for step in range(steps):
        ix=torch.randint(len(x),(batch,),generator=gen)
        ids=torch.randint(a8.K,(batch,),generator=gen)
        if step==0:activity=diagnostic_activity(m,x[ix],ids)
        opt.zero_grad(set_to_none=True)
        loss=(m(x[ix],ids)-y[ix,ids]).square().mean()
        if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite loss')
        loss.backward()
        if step==0:first_code_grad=float(m.code.weight.grad.norm())
        norm=torch.nn.utils.clip_grad_norm_((p for p in m.parameters() if p.requires_grad),5.0)
        if not bool(torch.isfinite(norm)):raise RuntimeError('nonfinite gradient')
        opt.step()
    return {'model':m.eval(),'train_last_batch_mse':float(loss.detach()),
            'initial_code_grad_norm':first_code_grad,'initial_active_fraction':activity}
