#!/usr/bin/env python3
"""Frozen 2026-10-08 KQ-SVD native-control algebra screen; NOT Mirror capacity.
Requires: numpy. CPU float64 for algebra, float32 only for inference serialization.
Run: OPENBLAS_NUM_THREADS=1 python run_stage0.py --output ./results
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, platform
from pathlib import Path
import numpy as np

DEV=(11,12,13)
FRESH=(101,102,103,104,105)
DIM=16
TOKENS=128
QUERY_CAL=96
QUERY_SUPPORT=12
QUERY_AUDIT=96
RANK=4
CODE_DIM=2
SOURCE_ROLES=6


def rand_orth(rng,n,k):
    q,_=np.linalg.qr(rng.standard_normal((n,k)))
    return q


def build_world(seed):
    rng=np.random.default_rng(seed)
    U=rand_orth(rng,TOKENS,DIM)
    V=rand_orth(rng,DIM,DIM)
    sv=np.geomspace(4.0,0.12,DIM)
    K=(U*sv) @ V.T
    assert np.linalg.matrix_rank(K)==DIM
    return rng,K,V


def query(rng,V,role,count):
    # Fixed query-family design: emphasize directions poorly represented by key-only top singular vectors.
    idx=np.arange(DIM)
    center=5+((3*role) % 10)
    scale=0.16+2.4*np.exp(-((idx-center)/2.1)**2)
    return (rng.standard_normal((count,DIM))*scale) @ V.T


def lowrank(a,r):
    U,s,Vt=np.linalg.svd(a,full_matrices=False)
    return (U[:,:r]*s[:r]) @ Vt[:r,:]


def native_kq_projector(Q,K,rank=RANK):
    """Rank-r optimizes a calibration SCORE matrix; only full-column Q/K case."""
    assert Q.shape[1]==K.shape[1] and np.linalg.matrix_rank(Q)==DIM
    score=Q @ K.T
    target=lowrank(score,rank)
    P=np.linalg.pinv(Q) @ target @ np.linalg.pinv(K.T)
    return P, np.linalg.norm(Q@P@K.T-target)/max(np.linalg.norm(target),1e-12)


def key_projector(K,r=RANK):
    _,_,Vt=np.linalg.svd(K,full_matrices=False)
    Vr=Vt[:r].T
    return Vr @ Vr.T


def relscore(Q,K,P):
    score=Q@K.T
    return np.linalg.norm(score-Q@P@K.T)/max(np.linalg.norm(score),1e-12)


def bytes_npz(**tensors):
    buff=io.BytesIO()
    np.savez(buff,**{k:np.asarray(v,dtype=np.float32) for k,v in tensors.items()})
    return len(buff.getvalue())


def one_world(seed,phase):
    rng,K,V=build_world(seed)
    Pk=key_projector(K)
    sources=[]
    max_cal=0.0
    for role in range(SOURCE_ROLES):
        Q=query(rng,V,role,QUERY_CAL)
        P,e=native_kq_projector(Q,K)
        sources.append(P)
        max_cal=max(max_cal,e)
    # Source-only dictionary fit. Does not include held-out task calibration.
    bank=np.stack(sources)
    mean=bank.mean(axis=0)
    matrix=(bank-mean).reshape(SOURCE_ROLES,-1)
    _,_,vt=np.linalg.svd(matrix,full_matrices=False)
    bases=vt[:CODE_DIM].reshape(CODE_DIM,DIM,DIM)
    src_codes=np.array([np.linalg.lstsq(bases.reshape(CODE_DIM,-1).T,(p-mean).ravel(),rcond=None)[0] for p in bank])
    # A previously unseen role: source identity is not trained/calibrated.
    role=7+(seed % 5)
    Qsupport=query(rng,V,role,QUERY_SUPPORT)
    Qa=query(rng,V,role,QUERY_AUDIT)
    design=np.stack([(Qsupport@x@K.T).ravel() for x in bases],axis=1)
    target=(Qsupport@K.T-Qsupport@mean@K.T).ravel()
    reg=1e-6
    m=np.linalg.solve(design.T@design+reg*np.eye(CODE_DIM),design.T@target)
    Pcode=mean+np.einsum('i,ijk->jk',m,bases)
    # Source-trained mean projector and source-only target role KQ best oracle are not tuned from audit.
    target_support_P=np.linalg.pinv(Qsupport)@lowrank(Qsupport@K.T,RANK)@np.linalg.pinv(K.T)
    native_worst=relscore(Qa,K,target_support_P)
    oracle=lowrank(Qa@K.T,RANK)
    oracle_error=np.linalg.norm(Qa@K.T-oracle)/np.linalg.norm(Qa@K.T)
    key_error=relscore(Qa,K,Pk)
    # Score optimality: rank-r SVD is oracle lower bound for any rank-r approximation.
    assert oracle_error <= key_error+1e-10
    assert np.linalg.matrix_rank(Pk)==RANK
    assert max_cal<1e-9
    # The ordinary linear-coded bank is EXACTLY the code model used in this pilot.
    # Mirror-specific geometric advantage cannot be concluded.
    native_bank_bytes=bytes_npz(K=K, per_role_projectors=np.concatenate([bank,target_support_P[None]],axis=0))
    linear_bank_bytes=bytes_npz(K=K, mean=mean, code_bases=bases, source_codes=src_codes, support_code=m)
    return dict(phase=phase,seed=seed,query_role=role,native_calibration_max_error=max_cal,
        oracle_rank4_rel_score=oracle_error,key_svd_rel_score=key_error,
        source_mean_rel_score=relscore(Qa,K,mean),
        target_support_native_rel_score=native_worst,
        ordinary_linear_code_rel_score=relscore(Qa,K,Pcode),
        physical_key_bytes_npz=bytes_npz(K=K),
        native_full_bank_bytes_npz=native_bank_bytes,linear_code_bank_bytes_npz=linear_bank_bytes,
        rank_source_native=int(np.linalg.matrix_rank(sources[0])),
        ordinary_linear_code_equals_proposed_linear_m=True)


def run(output):
    output.mkdir(parents=True,exist_ok=True)
    for phase,seeds in (('dev',DEV),('fresh',FRESH)):
        rows=[one_world(seed,phase) for seed in seeds]
        path=output/(phase+'_raw.csv')
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
    checks={}
    for name in ('dev_raw.csv','fresh_raw.csv'):
        checks[name]=hashlib.sha256((output/name).read_bytes()).hexdigest()
    checks['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    checks['platform']=platform.platform()
    checks['numpy']=np.__version__
    checks['dtype']='float64/algebra; float32/NPZ; CPU'
    checks['target_audit_used_for_fit']=False
    checks['mirror_specific_result']='NOT_EVALUATED: ordinary linear code is native control'
    (output/'VERIFICATION.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    return checks

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results'))
    a=parser.parse_args()
    r=run(a.output)
    print(json.dumps(r,indent=2))
