#!/usr/bin/env python3
"""Analytic DeepONet branch/trunk role-sharing mechanism screen."""
from __future__ import annotations
import csv, hashlib, io, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SEED=73401
D,RANK,DIM=8,4,8
PLANES=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
DOMAINS=8


def rotation(angles):
    r=np.eye(RANK)
    for theta,(i,j) in zip(angles,PLANES):
        c,s=np.cos(theta),np.sin(theta)
        g=np.eye(RANK);g[i,i]=g[j,j]=c;g[i,j]=-s;g[j,i]=s
        r=g@r
    return r


def npz_bytes(**arrays):
    b=io.BytesIO();np.savez_compressed(b,**arrays);return b.getvalue()


def nrmse(pred,target):
    return float(np.linalg.norm(pred-target)/max(np.linalg.norm(target),1e-30))


def run():
    rng=np.random.default_rng(SEED)
    w=rng.normal(0,.65,size=(RANK,D)).astype(np.float32)
    # Role-aligned synthetic teacher: distinct branch/trunk functions are generated
    # by applying small known rotations to one shared latent feature map.
    angles=rng.uniform(-.48,.48,size=(DOMAINS,2,len(PLANES))).astype(np.float32)
    xtrain=rng.normal(size=(DOMAINS,2,512,D)).astype(np.float32)
    xtest=rng.normal(size=(DOMAINS,2,256,D)).astype(np.float32)
    target_train=np.empty((DOMAINS,2,512,RANK),np.float32)
    target_test=np.empty((DOMAINS,2,256,RANK),np.float32)
    for d in range(DOMAINS):
      for role in range(2):
        rot=rotation(angles[d,role])
        target_train[d,role]=np.tanh(xtrain[d,role]@w.T)@rot.T
        target_test[d,role]=np.tanh(xtest[d,role]@w.T)@rot.T

    methods=[]
    # Independent-role upper control pays a full map and transform per logical role.
    wi=np.broadcast_to(w,(DOMAINS,2,RANK,D)).copy()
    methods.append(("untied_full",lambda d,r,x: np.tanh(x@wi[d,r].T)@rotation(angles[d,r]).T,
      npz_bytes(maps=wi,angles=angles,domain_ids=np.arange(DOMAINS,dtype=np.int16),
                role_ids=np.arange(2,dtype=np.int8))))
    methods.append(("hard_tied",lambda d,r,x: np.tanh(x@w.T),
      npz_bytes(shared_map=w,domain_ids=np.arange(DOMAINS,dtype=np.int16),role_ids=np.arange(2,dtype=np.int8))))
    # Output FiLM is fitted by least squares on training role examples only.
    gamma=np.empty((DOMAINS,2,RANK),np.float32);beta=np.empty_like(gamma)
    for d in range(DOMAINS):
      for role in range(2):
        h=np.tanh(xtrain[d,role]@w.T)
        for j in range(RANK):
          a,b=np.polyfit(h[:,j],target_train[d,role,:,j],1);gamma[d,role,j]=a;beta[d,role,j]=b
    methods.append(("shared_output_film",lambda d,r,x: np.tanh(x@w.T)*gamma[d,r]+beta[d,r],
      npz_bytes(shared_map=w,gamma=gamma,beta=beta,domain_ids=np.arange(DOMAINS,dtype=np.int16),role_ids=np.arange(2,dtype=np.int8))))
    direct_payload=npz_bytes(shared_map=w,angles=angles,domain_ids=np.arange(DOMAINS,dtype=np.int16),
      role_ids=np.arange(2,dtype=np.int8),plane_code=np.asarray(PLANES,np.int8))
    direct=lambda d,r,x: np.tanh(x@w.T)@rotation(angles[d,r]).T
    methods.append(("native_direct_givens",direct,direct_payload))
    methods.append(("mirror_role_view",direct,direct_payload))

    rows=[]
    for method,fn,payload in methods:
      fact=[];operator=[];wall=[]
      for d in range(DOMAINS):
        pred=[];ref=[]
        for role in range(2):
          t0=time.perf_counter();p=fn(d,role,xtest[d,role]);wall.append((time.perf_counter()-t0)*1e3)
          pred.append(p);ref.append(target_test[d,role])
          fact.append(nrmse(p,target_test[d,role]))
        # DeepONet rank-4 output is the branch/trunk factor inner product.
        yhat=pred[0]@pred[1].T; y=ref[0]@ref[1].T
        operator.append(nrmse(yhat,y))
      rows.append(dict(seed=SEED,method=method,domains=DOMAINS,serialized_bytes=len(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest(),branch_trunk_factor_nrmse=float(np.mean(fact)),
        composed_operator_nrmse=float(np.mean(operator)),active_multiplies=DOMAINS*2*256*D*RANK,
        wall_ms=float(np.sum(wall)),status_note="direct native alias" if method=="mirror_role_view" else ""))
    with (ROOT/"RESULTS_CORE.csv").open("w",newline="") as f:
      writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print("wrote",len(rows),"rows")

if __name__=="__main__":run()
