#!/usr/bin/env python3
"""Finite difference and exact tangent of Q/K gauge orbit with/without RoPE.
Registered hypothesis frozen before fresh seeds. This is an algebra audit,
not a learned Mirror capacity or compression experiment.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np

D_MODEL=4
D_K=2
THETA=.63
H=1e-5
REL_SVD_CUTOFF=1e-5
DEV=(11,12,13)
FRESH=(101,102,103,104,105)


def unpack(w):
    assert w.shape == (2*D_MODEL*D_K,)
    return w[:D_MODEL*D_K].reshape(D_MODEL,D_K), w[D_MODEL*D_K:].reshape(D_MODEL,D_K)


def rotation(theta):
    return np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]],dtype=np.float64)


def score(w, rope=False):
    q,k=unpack(w)
    c0=q@k.T
    if rope:
        return np.concatenate((c0.ravel(),(q@rotation(THETA)@k.T).ravel()))
    return c0.ravel()


def finite_jacobian(w, rope):
    ys=score(w,rope)
    j=np.empty((ys.size,w.size),np.float64)
    for col in range(w.size):
        inc=np.zeros_like(w);inc[col]=H
        j[:,col]=(score(w+inc,rope)-score(w-inc,rope))/(2*H)
    return j


def tangent_delta(q,k,x,rope=False):
    dq=q@x
    dk=-k@x.T
    c=dq@k.T+q@dk.T
    if rope:
        r=rotation(THETA)
        return np.concatenate((c.ravel(),(dq@r@k.T+q@r@dk.T).ravel()))
    return c.ravel()


def world(seed):
    rng=np.random.default_rng(seed)
    q=rng.standard_normal((D_MODEL,D_K))
    k=rng.standard_normal((D_MODEL,D_K))
    w=np.concatenate((q.ravel(),k.ravel()))
    t0=time.perf_counter_ns()
    j0=finite_jacobian(w,False)
    jr=finite_jacobian(w,True)
    elapsed_s=(time.perf_counter_ns()-t0)*1e-9
    sv0=np.linalg.svd(j0,compute_uv=False)
    svr=np.linalg.svd(jr,compute_uv=False)
    rank0=int(np.sum(sv0>REL_SVD_CUTOFF*sv0[0]))
    rankr=int(np.sum(svr>REL_SVD_CUTOFF*svr[0]))
    E=[]
    for i in range(2):
        for j in range(2):
            x=np.zeros((2,2),dtype=np.float64);x[i,j]=1
            E.append(x)
    i=np.eye(2)
    J=np.array([[0.,-1.],[1.,0.]])
    shear=np.array([[0.,1.],[0.,0.]])
    exact_no=max(np.linalg.norm(tangent_delta(q,k,x,False),ord=np.inf) for x in E)
    valid_ro=max(np.linalg.norm(tangent_delta(q,k,x,True),ord=np.inf) for x in [i,J])
    invalid_ro=max(np.linalg.norm(tangent_delta(q,k,x,True),ord=np.inf) for x in [shear,np.array([[1,0],[0,-1]])])
    g=np.array([[1.2,.3],[-.1,1.1]])
    g_valid=i+.2*J
    g_invalid=i+.35*shear
    c0=q@k.T
    c1=q@rotation(THETA)@k.T
    eq_exact=float(np.max(np.abs(c0-(q@g)@(k@np.linalg.inv(g).T).T)))
    ro_valid=float(np.max(np.abs(c1-(q@g_valid)@rotation(THETA)@(k@np.linalg.inv(g_valid).T).T)))
    ro_invalid=float(np.max(np.abs(c1-(q@g_invalid)@rotation(THETA)@(k@np.linalg.inv(g_invalid).T).T)))
    # finite Jacobian gauge tangent residual independent check
    eps_j0=max(float(np.max(np.abs(j0@np.concatenate(((q@x).ravel(),(-k@x.T).ravel()))))) for x in E)
    eps_jr=max(float(np.max(np.abs(jr@np.concatenate(((q@x).ravel(),(-k@x.T).ravel()))))) for x in [i,J])
    return dict(seed=seed,rank_no_rope=rank0,rank_rope=rankr,nullity_no_rope=w.size-rank0,nullity_rope=w.size-rankr,
       full_no_rope_gauge_tangent_inf=exact_no,valid_rope_gauge_tangent_inf=valid_ro,
       invalid_rope_tangent_inf=invalid_ro,
       finite_jac_no_rope_tangent_inf=eps_j0,finite_jac_rope_tangent_inf=eps_jr,
       no_rope_exact_full_GL_error=eq_exact,valid_rope_exact_error=ro_valid,invalid_rope_exact_error=ro_invalid,
       cond_Q=float(np.linalg.cond(q)),cond_K=float(np.linalg.cond(k)),
       nonzero_singval_no_rope=float(sv0[rank0-1]),discarded_singval_no_rope=float(sv0[rank0]),
       nonzero_singval_rope=float(svr[rankr-1]),discarded_singval_rope=float(svr[rankr]),
       jacobian_runtime_s=elapsed_s)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',required=True,choices=['dev','fresh'])
    ap.add_argument('--out',default='/mnt/data/mirror_gauge_stage0')
    args=ap.parse_args()
    seeds=DEV if args.phase=='dev' else FRESH
    rows=[world(i) for i in seeds]
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    target=root/f'{args.phase}_raw.csv'
    with target.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    verdict={
      'rank_12_without_rope':all(r['rank_no_rope']==12 for r in rows),
      'rank_14_with_rope':all(r['rank_rope']==14 for r in rows),
      'four_pure_gauge_tangents':all(r['full_no_rope_gauge_tangent_inf']<1e-10 for r in rows),
      'two_valid_rope_tangents':all(r['valid_rope_gauge_tangent_inf']<1e-10 for r in rows),
      'noncommuting_rope_detected':all(r['invalid_rope_tangent_inf']>1e-3 for r in rows),
      'full_GL_exact_invariant':all(r['no_rope_exact_full_GL_error']<1e-10 for r in rows),
      'RoPE_commutant_exact_invariant':all(r['valid_rope_exact_error']<1e-10 for r in rows),
      'RoPE_invalid_exact_detected':all(r['invalid_rope_exact_error']>1e-3 for r in rows)
    }
    summary={"phase":args.phase,"seeds":list(seeds),"quality":verdict,"system":{"numpy":np.__version__,"python":platform.python_version(),"device":"CPU x86_64", "float":"float64", "finite_difference_h":H,"rank_cutoff_relative":REL_SVD_CUTOFF},
     "ranges":{"condQ":[min(r['cond_Q'] for r in rows),max(r['cond_Q'] for r in rows)],
       "condK":[min(r['cond_K'] for r in rows),max(r['cond_K'] for r in rows)],
       "max_jac_null":max(r['finite_jac_rope_tangent_inf'] for r in rows),
       "min_invalid_rope_exact":min(r['invalid_rope_exact_error'] for r in rows)},
     "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     "raw_sha256":hashlib.sha256(target.read_bytes()).hexdigest()}
    (root/f'{args.phase}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
