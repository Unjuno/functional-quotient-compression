#!/usr/bin/env python3
"""Independent finite-dimensional audit of historical FQC/Mirror formulae.

Exploratory exact algebra and constructed counterexamples ONLY.  Not a natural
model training/quality/storage win, and not an independent replication of MA.
Contract: FORMULA_STAGE0_PROTOCOL.json committed before any fresh run.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import platform
from pathlib import Path

import numpy as np

DEV = (11, 12, 13)
FRESH = (101, 102, 103, 104, 105)
CHECKS = tuple(f'F{i:02d}' for i in range(1,13))


def gelu(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    erf = np.frompyfunc(math.erf, 1, 1)
    return .5 * arr * (1 + erf(arr / math.sqrt(2)).astype(np.float64))


def softmax(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    exp = np.exp(a-np.max(a))
    return exp / exp.sum()


def relative_error(a, b):
    return float(np.linalg.norm(a-b) / max(np.linalg.norm(a), 1e-12))


def algebra_checks(seed: int):
    rng = np.random.default_rng(seed)
    metrics = {}

    # F01: exact static-view fold, and a pure-permutation negative control.
    x = rng.normal(size=(12,3))
    wup = rng.normal(size=(4,3)); wdown = rng.normal(size=(2,4))
    bup = rng.normal(size=4); bdown = rng.normal(size=2)
    q = np.eye(4)+0.1*rng.normal(size=(4,4))
    qinv = np.linalg.inv(q)
    hidden = x@wup.T + bup
    folded = gelu(x@(q@wup).T + q@bup) @ (wdown@qinv).T + bdown
    mirror = gelu(hidden@q.T) @ qinv.T @ wdown.T + bdown
    fold_err = float(np.max(np.abs(folded-mirror)))
    perm = np.eye(4)[[2,0,3,1]]
    perm_err = float(np.max(np.abs(gelu(hidden@perm.T)@perm - gelu(hidden))))
    metrics['F01'] = max(fold_err,perm_err)
    assert fold_err<1e-11 and perm_err<1e-11

    # F02: exact weighted minimum J delta=b, gauge/rank qualification,
    # and correct direction of baseline-cost generalized eigenproblem.
    J = np.array([[1.,0.,0.],[0.,.1,0.]])
    R = np.diag([1.,1.,3.])
    W = J@np.linalg.solve(R,J.T)
    b = rng.normal(size=2)
    dstar=np.linalg.solve(R,J.T@np.linalg.solve(W,b))
    cal_cost=.5*float(b@np.linalg.solve(W,b))
    direct_cost=.5*float(dstar@R@dstar)
    perturbed_cost=.5*float((dstar+np.array([0.,0.,.3]))@R@(dstar+np.array([0.,0.,.3])))
    RB=np.linalg.pinv(W)
    eigen=np.linalg.eigvalsh(RB)
    rank_null=np.array([[1.,0.],[0.,0.]])
    target_null=np.array([0.,1.])
    null_unreachable=np.linalg.norm(rank_null@np.linalg.pinv(rank_null)@target_null-target_null)>1e-12
    assert perturbed_cost>=direct_cost and null_unreachable
    assert abs(eigen.max()-100.)<1e-11
    metrics['F02']=max(abs(cal_cost-direct_cost),np.linalg.norm(J@dstar-b),abs(eigen.max()-100.))
    assert metrics['F02'] < 1e-10

    # F03: exact finite-T residual recurrence for fixed linear least-squares.
    eta=.5; steps=100
    Js=np.diag([1.,.1]); Ws=Js@Js.T
    eb=rng.normal(size=2)
    dx=np.zeros(2)
    for _ in range(steps):
        dx += eta*Js.T@(eb-Js@dx)
    residual=eb-Js@dx
    prediction=np.linalg.matrix_power(np.eye(2)-eta*Ws,steps)@eb
    metrics['F03']=float(np.linalg.norm(residual-prediction))
    assert metrics['F03']<1e-12

    # F04: regular simplex with minimum K=r+1 mean-zero support.
    r=3; K=r+1
    H=np.eye(K)-np.ones((K,K))/K
    eigs,Vs=np.linalg.eigh(H)
    Vbasis=Vs[:,-r:]
    code=np.sqrt(K/r)*Vbasis
    mean_err=float(np.max(np.abs(np.mean(code,axis=0))))
    cov_err=float(np.max(np.abs(code.T@code/K-np.eye(r)/r)))
    min_support=np.eye(r)-np.ones((r,r))/r
    rank_bound=int(np.linalg.matrix_rank(min_support)) <= r-1
    metrics['F04']=max(mean_err,cov_err)
    assert rank_bound and metrics['F04']<1e-11

    # F05: zero-mean phase code does NOT cancel token-specific sensitivity.
    codes=np.array([1.,-1.]); beta=np.array([1.,2.])
    sum_first=float(beta@codes)
    metrics['F05']=abs(sum_first+1.)
    assert np.allclose(codes.mean(),0.) and sum_first !=0 and metrics['F05']<1e-12

    # F06: exact lazily transformed K/V read, only with unchanged prefix.
    kc=rng.normal(size=(7,4)); vc=rng.normal(size=(7,4)); query=rng.normal(size=4)
    A=np.eye(4)+.1*rng.normal(size=(4,4))
    B=np.eye(4)+.1*rng.normal(size=(4,4))
    explicit=softmax(query@(kc@A).T/math.sqrt(4))@(vc@B)
    lazy=(softmax((query@A.T)@kc.T/math.sqrt(4))@vc)@B
    difference=float(np.max(np.abs(explicit-lazy)))
    invalid=softmax((query@A.T)@(kc+.3*rng.normal(size=(7,4))).T/math.sqrt(4))@vc@B
    unsafe_diff=float(np.max(np.abs(invalid-explicit)))
    metrics['F06']=difference
    assert difference<1e-11 and unsafe_diff>1e-8

    # F07: task coupling majorizer; diagonal-only underestimates a witness.
    R0=rng.normal(size=(4,4)); G=R0.T@R0+.1*np.eye(4)
    err=rng.normal(size=4)
    exact=float(err@G@err)
    maj=float(sum((G[i,i]+sum(abs(G[i,j]) for j in range(4) if i!=j))*err[i]**2 for i in range(4)))
    Ge=np.array([[1.,.8],[.8,1.]])
    diag_underestimate=float(np.array([1.,1.])@Ge@np.array([1.,1.]) - np.trace(Ge))
    metrics['F07']=max(0.,exact-maj)
    assert diag_underestimate>1 and maj>=exact-1e-10

    # F08: exact serializer breaks naive raw-bit 80 bit eligibility.
    raw=76; overhead=5; allocated=8*math.ceil((raw+overhead)/8)
    metrics['F08']=abs(allocated-88)
    assert raw<=80 and allocated>80 and allocated==88

    # F09: equivalence requires decoded semantics AND decoder prerequisites.
    key_a=('decoded:one','requires:alpha')
    key_b=('decoded:one','requires:beta')
    key_a2=('decoded:one','requires:alpha')
    collapsed=len({key_a,key_b,key_a2})
    metrics['F09']=abs(collapsed-2)
    assert key_a!=key_b and key_a==key_a2 and collapsed==2

    # F10: known constructed complementarity (not natural Mirror gain).
    state={(0,0):10.,(0,1):11.,(1,0):12.,(1,1):5.}
    incumbent=(0,0); one_axis=[state[(1,0)],state[(0,1)]]
    optimum=min(state,key=state.get)
    metrics['F10']=abs(state[optimum]-5.)
    assert optimum==(1,1) and min(one_axis)>state[incumbent]

    # F11: local activation-weighted function error identity and rank reversal.
    Sigma=np.diag([100.,.01]); E1=np.diag([.2,0.]); E2=np.diag([0.,2.])
    trace=float(np.trace(E1@Sigma@E1.T))
    root=np.diag(np.sqrt(np.diag(Sigma)))
    norm=float(np.linalg.norm(E1@root,'fro')**2)
    metrics['F11']=abs(trace-norm)
    assert np.linalg.norm(E1,'fro')<np.linalg.norm(E2,'fro')
    assert trace>np.trace(E2@Sigma@E2.T)
    assert metrics['F11']<1e-10

    # F12: ordered rule action is not parallel-summed residual; commutator.
    A0=np.array([[0.,1.],[0.,0.]])
    B0=np.array([[0.,0.],[1.,0.]])
    C=(np.eye(2)+A0)@(np.eye(2)+B0)
    D=(np.eye(2)+B0)@(np.eye(2)+A0)
    bracket=A0@B0-B0@A0
    metrics['F12']=float(np.max(np.abs(C-D-bracket)))
    assert np.linalg.norm(C-D)>1 and metrics['F12']<1e-12

    if any(not np.isfinite(v) for v in metrics.values()):
        raise AssertionError('non-finite result')
    meta={'gauge_condition':float(np.linalg.cond(q)),
          'unsafe_cache_output_difference':unsafe_diff,
          'cross_block_diag_underestimate':diag_underestimate,
          'coupling_majorizer_gap':maj-exact,
          'reachability_Emax':float(eigen.max()),
          'k_support':K,
          'joint_only_gain':state[incumbent]-state[optimum],
          'weight_frobenius_misrank':True}
    return metrics,meta


def execute(outdir: Path):
    outdir.mkdir(parents=True,exist_ok=True)
    products={}
    for phase,seeds in [('dev',DEV),('fresh',FRESH)]:
        rows=[]
        for seed in seeds:
            vals,meta=algebra_checks(seed)
            rows.append(dict(phase=phase,seed=seed,
                             **{f'{k}_max_abs_error':vals[k] for k in CHECKS},
                             **meta))
        path=outdir/f'{phase}_raw.csv'
        with path.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
        products[phase]=rows
    verification={'protocol':'FORMULA_STAGE0_PROTOCOL.json',
                  'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'results_sha256':{x:hashlib.sha256((outdir/x).read_bytes()).hexdigest() for x in ['dev_raw.csv','fresh_raw.csv']},
                  'hardware':'CPU only, no GPU',
                  'numpy':np.__version__,
                  'platform':platform.platform(),
                  'dtype':'float64',
                  'dev_seeds':list(DEV),'fresh_seeds':list(FRESH),
                  'all_12_pass_each_world':True,
                  'capacity_or_adoption_claim':False,
                  'limits':'exact algebra + constructed examples only; no trained models, no natural task loss'}
    (outdir/'VERIFICATION.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'check_count':len(CHECKS),'fresh_worlds':len(FRESH),
                       'max_numerical_error':max(row[f'{k}_max_abs_error'] for row in products['fresh'] for k in CHECKS),
                       'min_invalid_cache_gap':min(row['unsafe_cache_output_difference'] for row in products['fresh']),
                       'exact_budget_criterion': '76 raw bits -> 88 serializer bits',
                       'complementarity_constructed_gain':5,
                       'source_sha256':verification['source_sha256']},indent=2))
    return products,verification


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    execute(parser.parse_args().output)
