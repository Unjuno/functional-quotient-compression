# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""MNW v1: fixed 64-byte header + all FP32 parameters + 32-byte SHA-256.

Little-endian, no pickle, no external weights. Shared decoder/source is not
charged to either model; every model-specific numeric inference parameter is.
KV cache and temporary activations are runtime state, not this weight artifact.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import struct
import numpy as np
import torch
from model import TinyLM
MAGIC=b'MN002W01'
HEADER=struct.Struct('<8s10I16s')
KINDS=('dense','mirror','direct_gate','full_moe')

def expected_parameters(c):
    d,h,s,l,v,m=[c[k] for k in ('d','hidden','states','layers','vocab','max_length')]
    ff=2*d*h+h+d
    if c['kind']=='mirror':ff+=d*s+s+s*h
    elif c['kind']=='direct_gate':ff+=d*h+h
    elif c['kind']=='full_moe':ff=s*ff+d*s+s
    return v*d+m*d+2*d+d*v+v+l*(4*d*d+8*d+ff)

def save_model(model: TinyLM, path: Path):
    c=model.config
    state=model.state_dict()
    if any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('MNW v1 requires finite FP32 state')
    n=sum(v.numel() for v in state.values())
    if n!=expected_parameters(c):raise ValueError('unsupported model-state schema')
    header=HEADER.pack(MAGIC,1,KINDS.index(c['kind']),c['d'],c['hidden'],c['states'],
                       c['layers'],c['heads'],c['vocab'],c['max_length'],n,b'\0'*16)
    body=b''.join(v.detach().cpu().contiguous().numpy().astype('<f4',copy=False).tobytes() for v in state.values())
    raw=header+body
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(raw+hashlib.sha256(raw).digest())

def load_model(path: Path) -> TinyLM:
    path=Path(path)
    if not 96<=path.stat().st_size<=256_000_096:
        raise ValueError('invalid or oversized file')
    raw=path.read_bytes()
    if hashlib.sha256(raw[:-32]).digest()!=raw[-32:]:raise ValueError('hash mismatch')
    magic,version,kind,d,h,s,l,heads,v,m,n,reserved=HEADER.unpack(raw[:64])
    if magic!=MAGIC or version!=1 or kind>=len(KINDS) or reserved!=b'\0'*16:
        raise ValueError('unsupported or malformed header')
    if not (1<=d<=4096 and 1<=h<=16384 and 1<=s<=1024 and 1<=l<=64 and
            1<=heads<=d and d%heads==0 and 1<=v<=1_000_000 and 1<=m<=1_000_000):
        raise ValueError('dimensions out of range')
    c=dict(kind=KINDS[kind],d=d,hidden=h,states=s,layers=l,heads=heads,vocab=v,max_length=m)
    if n!=expected_parameters(c) or n>64_000_000 or len(raw)!=96+4*n:
        raise ValueError('payload size does not match configuration')
    values=np.frombuffer(raw,dtype='<f4',count=n,offset=64).astype(np.float32,copy=True)
    if not np.isfinite(values).all():raise ValueError('nonfinite parameter')
    model=TinyLM(**c).eval();state=model.state_dict();offset=0
    for key,value in state.items():
        length=value.numel();state[key]=torch.from_numpy(values[offset:offset+length]).reshape(value.shape)
        offset+=length
    assert offset==n
    model.load_state_dict(state)
    return model
