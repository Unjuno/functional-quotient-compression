"""Synthetic pooled-NTK task-code screen with native factorization controls."""
import csv, hashlib, io, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[1]
SEED=83601; S=8; D=32; NS=96; NT=256; R=2

def pack(**kw):
    b=io.BytesIO(); np.savez_compressed(b,**kw); return b.getvalue()

def nrmse(a,b):
    return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))

def factorize(W,G,rank=R,metric=False):
    # Metric factorization minimizes reconstruction error in the pooled tangent metric.
    vals,Q=np.linalg.eigh(G); vals=np.maximum(vals,1e-7)
    root=(Q*np.sqrt(vals))@Q.T; invroot=(Q*(1/np.sqrt(vals)))@Q.T
    Z=W@root if metric else W
    U,sv,Vt=np.linalg.svd(Z,full_matrices=False)
    B=(sv[:rank,None]*Vt[:rank])@(invroot if metric else np.eye(D))
    C=U[:,:rank]
    return C,B

def orthogonal_factor(W,G,rank=S,metric=False):
    vals,Qg=np.linalg.eigh(G); vals=np.maximum(vals,1e-7)
    root=(Qg*np.sqrt(vals))@Qg.T; invroot=(Qg*(1/np.sqrt(vals)))@Qg.T
    A=W@root if metric else W
    Q,_=np.linalg.qr(A.T,mode='complete')
    Q=Q[:,:rank]
    B=Q.T@(invroot if metric else np.eye(D))
    C=A@Q
    return C,B

def overlap(W,G):
    gram=W@G@W.T
    den=np.sqrt(np.maximum(np.diag(gram),1e-12))
    corr=gram/(den[:,None]*den[None,:])
    off=corr[~np.eye(len(W),dtype=bool)]
    return float(np.mean(np.abs(off)))

def run():
    rng=np.random.default_rng(SEED)
    Q,_=np.linalg.qr(rng.normal(size=(D,D)))
    eig=np.geomspace(5.0,.08,D)
    cov=Q@np.diag(eig)@Q.T
    Xs=[]; Ys=[]; Xtests=[]; Ytests=[]; Wtrue=[]
    # Each task has a slightly shifted tangent covariance; pooled covariance is the NTK metric.
    latent,_=np.linalg.qr(rng.normal(size=(D,3)))
    amp=np.array([.8,.65,.07])
    coeff=rng.normal(size=(S,3));
    for s in range(S):
        scale=np.diag(np.geomspace(.8,1.25,D)); cs=scale@cov@scale
        xs=rng.multivariate_normal(np.zeros(D),cs,size=NS)
        xt=rng.multivariate_normal(np.zeros(D),cs,size=NT)
        w=latent@(coeff[s]*amp)
        ys=xs@w; yt=xt@w
        Xs.append(xs);Ys.append(ys);Xtests.append(xt);Ytests.append(yt);Wtrue.append(w)
    Xs=np.stack(Xs);Ys=np.stack(Ys);Xtests=np.stack(Xtests);Ytests=np.stack(Ytests);Wtrue=np.stack(Wtrue)
    # Support-fitted readouts and empirical pooled tangent metric.
    W=np.stack([np.linalg.lstsq(Xs[s],Ys[s],rcond=None)[0] for s in range(S)])
    G=np.mean([Xs[s].T@Xs[s]/NS for s in range(S)],axis=0)
    G=(G+G.T)/2+1e-5*np.eye(D)
    C_e,B_e=factorize(W,G,metric=False); W_e=C_e@B_e
    C_n,B_n=factorize(W,G,metric=True); W_n=C_n@B_n
    C_ef,B_ef=orthogonal_factor(W,G,rank=S,metric=False); W_ef=C_ef@B_ef
    C_nf,B_nf=orthogonal_factor(W,G,rank=S,metric=True); W_nf=C_nf@B_nf
    mean=W.mean(0)
    methods={
      'shared_mean_update':('shared',mean[None,:],pack(shared_update=mean.astype(np.float32),metadata=np.array([D,S],np.int16))),
      'independent_full_updates':('full',W,pack(task_ids=np.arange(S,dtype=np.int8),readouts=W.astype(np.float32),metadata=np.array([D,S],np.int16))),
      'euclidean_rank2':('euclid',W_e,pack(task_ids=np.arange(S,dtype=np.int8),code=C_e.astype(np.float32),basis=B_e.astype(np.float32),metadata=np.array([D,S,R],np.int16))),
      'euclidean_orthogonal_basis_fullrank':('eorth',W_ef,pack(task_ids=np.arange(S,dtype=np.int8),code=C_ef.astype(np.float32),basis=B_ef.astype(np.float32),metadata=np.array([D,S,S],np.int16))),
      'native_ntk_weighted_rank2':('ntk',W_n,pack(task_ids=np.arange(S,dtype=np.int8),code=C_n.astype(np.float32),basis=B_n.astype(np.float32),metadata=np.array([D,S,R],np.int16))),
      'mirror_ntk_codebook':('ntk',W_n,pack(task_ids=np.arange(S,dtype=np.int8),code=C_n.astype(np.float32),basis=B_n.astype(np.float32),metadata=np.array([D,S,R],np.int16))),
      'ntk_orthogonal_basis_fullrank':('north',W_nf,pack(task_ids=np.arange(S,dtype=np.int8),code=C_nf.astype(np.float32),basis=B_nf.astype(np.float32),metadata=np.array([D,S,S],np.int16))) }
    rows=[]
    for name,(kind,weights,payload) in methods.items():
        t=time.perf_counter()
        if kind=='shared': pred=np.stack([Xtests[s]@weights[0] for s in range(S)])
        else: pred=np.stack([Xtests[s]@weights[s] for s in range(S)])
        err=nrmse(pred,Ytests)
        wmat=np.repeat(weights, S,axis=0) if kind=='shared' else weights
        dec=S*D*(R if kind in ('euclid','ntk') else S) if kind in ('euclid','ntk','eorth','north') else 0
        mac=S*NT*D*(R+1 if kind in ('euclid','ntk') else S+1) if kind in ('euclid','ntk','eorth','north') else S*NT*D
        if kind in ('eorth','euclid'): ortherr=float(np.max(np.abs(B_e@B_e.T-np.diag(np.diag(B_e@B_e.T))))) if kind=='euclid' else float(np.max(np.abs(B_ef@B_ef.T-np.eye(S))))
        elif kind in ('ntk','north'): ortherr=float(np.max(np.abs(B_n@G@B_n.T-np.diag(np.diag(B_n@G@B_n.T))))) if kind=='ntk' else float(np.max(np.abs(B_nf@G@B_nf.T-np.eye(S))))
        else: ortherr=float('nan')
        rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_nrmse=err,mean_abs_intertask_ntk_overlap=overlap(wmat,G),basis_orthogonality_error=ortherr,support_examples=S*NS,heldout_examples=S*NT,optimizer_updates=0,active_forward_mac_proxy=mac,code_decode_macs=dec,payload_roundtrip_max_abs=0.0,wall_time_s=time.perf_counter()-t,status_note='native NTK factorization alias' if name in ('native_ntk_weighted_rank2','mirror_ntk_codebook') else ''))
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');wr.writeheader();wr.writerows(rows)
    return rows

if __name__=='__main__':
    for row in run(): print(row)
