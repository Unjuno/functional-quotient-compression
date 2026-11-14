# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A7 compositional spatial predicates on real digits; never use a Mirror teacher.

A7 intentionally imports the unchanged A6 model and FP32 serializer. The data
factory and task selection are separate, train-split-only, and deterministic.
"""
from __future__ import annotations
import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

A6_DIR=Path(__file__).resolve().parents[1] / 'A6'
if not (A6_DIR/'a6.py').is_file():
    raise FileNotFoundError('Expected sibling A6/a6.py, required source dependency')
sys.path.insert(0,str(A6_DIR))
import a6


def source_core():
    return a6


def task_templates() -> np.ndarray:
    """Eight fixed contrast templates, normalized once with no fitted parameters."""
    yy,xx=np.mgrid[0:8,0:8].astype(np.float32)
    u=xx-np.float32(3.5); v=yy-np.float32(3.5)
    templates=(u, v, u*v, u*u-v*v, u*u+v*v,
               np.sin((xx+1)*np.pi/4),np.cos((yy+1)*np.pi/4),
               np.sin((xx+yy+1)*np.pi/4))
    result=[]
    for pattern in templates:
        flat=np.asarray(pattern,dtype=np.float32).reshape(-1)
        flat=flat-flat.mean(dtype=np.float32)
        flat=flat/(np.linalg.norm(flat)+np.float32(1e-12))
        result.append(flat)
    return np.stack(result).astype(np.float32)


def make_world(world:int, family:str) -> dict:
    """Return image-disjoint splits and train-only-calibrated pairwise targets.

    No task mapping, source template or threshold is passed to the student.
    """
    if not isinstance(world,int) or world < 0 or family not in ('pair_and','pair_xor'):
        raise ValueError('invalid world or family')
    dat=load_digits()
    image=np.asarray(dat.data,dtype=np.float32)/16
    digit=np.asarray(dat.target,dtype=np.int64)
    ids=np.arange(len(image))
    train_ids,temp_ids=train_test_split(ids,test_size=0.4,stratify=digit,random_state=world*17+7)
    val_ids,test_ids=train_test_split(temp_ids,test_size=0.5,stratify=digit[temp_ids],random_state=world*17+13)
    templates=task_templates()
    projections=image@templates.T
    thresholds=np.median(projections[train_ids],axis=0).astype(np.float32)
    bits=projections>thresholds[None,:]
    pairs_all=list(itertools.combinations(range(8),2))
    rng=np.random.default_rng(world*11027 + (211 if family=='pair_and' else 419))
    rng.shuffle(pairs_all)
    chosen=[]
    for i,j in pairs_all:
        left,right=bits[train_ids,i],bits[train_ids,j]
        labels=(left&right) if family=='pair_and' else (left^right)
        prevalence=float(np.mean(labels))
        lower,upper=(0.18,0.40) if family=='pair_and' else (0.30,0.70)
        if lower<=prevalence<=upper:
            chosen.append((i,j))
        if len(chosen)==a6.TASKS:break
    if len(chosen)<a6.TASKS:
        raise ValueError(f'not enough balanced pairs in world {world}: {len(chosen)}')
    pairs=np.asarray(chosen,dtype=np.int64)
    def labels_from_ids(indices):
        value=np.stack([(bits[indices,i]&bits[indices,j]) if family=='pair_and' else
                        (bits[indices,i]^bits[indices,j]) for i,j in chosen],axis=0)
        return torch.tensor(value.astype(np.float32,copy=True))
    def features(indices,ood=False):
        img=image[indices].copy()
        if ood:
            noise=np.random.default_rng(world*313+900000).normal(0,.08,size=img.shape).astype(np.float32)
            img=np.clip(img+noise,0,1)
        return torch.tensor(img),labels_from_ids(indices)
    outputs={p:features(ix) for p,ix in [('train',train_ids),('val',val_ids),('iid',test_ids)]}
    outputs['ood']=features(test_ids,True)
    outputs['pairs']=pairs
    outputs['thresholds']=thresholds
    outputs['split_ids']={'train':train_ids.tolist(),'val':val_ids.tolist(),'iid':test_ids.tolist()}
    fingerprint_payload=(image.tobytes() + digit.tobytes()+templates.tobytes()+pairs.tobytes()+
                         thresholds.tobytes()+np.array([world],dtype='<i8').tobytes()+family.encode())
    outputs['fingerprint']=hashlib.sha256(fingerprint_payload).hexdigest()
    return outputs
