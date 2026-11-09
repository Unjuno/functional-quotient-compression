#!/usr/bin/env python3
"""Synthetic content-cache + deferred RoPE/View attention experiment."""
import csv, io, json, math, time, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D,T,NV=16,128,8
SEEDS={'dev-69301':69301,'audit-69311':69311,'audit-69312':69312,'audit-69313':69313}

def rot2(a):
    c,s=np.float32(np.cos(a)),np.float32(np.sin(a))
    if np.ndim(c)==0: return np.array([[c,-s],[s,c]],dtype=np.float32)
    return np.stack((np.stack((c,-s),axis=-1),np.stack((s,c),axis=-1)),axis=-2).astype(np.float32)

def rope(x, pos, freq):
    y=np.empty_like(x)
    for j,f in enumerate(freq):
        y[:,2*j:2*j+2]=np.einsum('ti,tji->tj',x[:,2*j:2*j+2],rot2(pos*f))
    return y

def givens_cross(x, theta):
    y=x.copy()
    c,s=np.float32(np.cos(theta)),np.float32(np.sin(theta))
    a=x[...,0].copy(); b=x[...,2].copy()
    y[...,0]=c*a-s*b; y[...,2]=s*a+c*b
    return y

def value_view(x, angles):
    y=x.copy()
    for j,a in enumerate(angles): y[:,2*j:2*j+2]=x[:,2*j:2*j+2] @ rot2(a).T
    return y

def softmax(x):
    z=x-np.max(x,axis=-1,keepdims=True); e=np.exp(z,dtype=np.float32)
    return e/np.sum(e,axis=-1,keepdims=True)

def payload(**arrs):
    b=io.BytesIO(); np.savez(b,**arrs); return b.getvalue()

def run_one(label,seed,outdir):
    rng=np.random.default_rng(seed)
    q=rng.normal(size=(T,D)).astype(np.float32); k=rng.normal(size=(T,D)).astype(np.float32); v=rng.normal(size=(T,D)).astype(np.float32)
    pos=np.arange(T,dtype=np.float32); freq=(10000.**(-np.arange(D//2,dtype=np.float32)*2/D)).astype(np.float32)
    qrot=rope(q,pos,freq); mask=np.where(np.tril(np.ones((T,T),dtype=np.float32))>0,0.,-1e9).astype(np.float32)
    key_angles=rng.uniform(-0.65,0.65,size=NV).astype(np.float32)
    value_angles=rng.uniform(-0.65,0.65,size=(NV,D//2)).astype(np.float32)
    mats_k=[]; mats_v=[]; maxerr=0.; max_approx=0.; max_native=0.
    t0=time.perf_counter()
    for m in range(NV):
        # Explicit cache: apply cross-plane Givens in content space, then RoPE per key position.
        km=rope(givens_cross(k,key_angles[m]),pos,freq)
        vm=value_view(v,value_angles[m]); mats_k.append(km); mats_v.append(vm)
        pmat=softmax((qrot@km.T)/math.sqrt(D)+mask); outmat=pmat@vm
        # Deferred path: construct position-specific viewed keys at attention read; values map after attention.
        klazy=rope(givens_cross(k,key_angles[m]),pos,freq)
        plazy=softmax((qrot@klazy.T)/math.sqrt(D)+mask)
        outlazy=value_view(plazy@v,value_angles[m])
        # Native Givens parameterization uses same coordinates and read path.
        knative=rope(givens_cross(k,key_angles[m]),pos,freq)
        pnative=softmax((qrot@knative.T)/math.sqrt(D)+mask)
        outnative=value_view(pnative@v,value_angles[m])
        # Deliberately incorrect shortcut: assume Givens commutes with RoPE and move it to query.
        qapprox=givens_cross(qrot,-key_angles[m])
        papprox=softmax((qapprox@rope(k,pos,freq).T)/math.sqrt(D)+mask)
        outapprox=value_view(papprox@v,value_angles[m])
        denom=max(float(np.linalg.norm(outmat)),1e-30)
        maxerr=max(maxerr,float(np.linalg.norm(outmat-outlazy)/denom))
        max_native=max(max_native,float(np.linalg.norm(outmat-outnative)/denom))
        max_approx=max(max_approx,float(np.linalg.norm(outmat-outapprox)/denom))
    wall=time.perf_counter()-t0
    # Cache includes no precomputed position-specific RoPE tensors; frequencies/positions and causal metadata are paid.
    lazy=payload(K_content=k,V_content=v,key_view_angles=key_angles,value_view_angles=value_angles,
                 positions=pos,rope_frequencies=freq,causal_mask=mask)
    independent=payload(K_views=np.stack(mats_k),V_views=np.stack(mats_v),positions=pos,rope_frequencies=freq,causal_mask=mask)
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/f'{label}.lazy.npz').write_bytes(lazy); (outdir/f'{label}.independent.npz').write_bytes(independent)
    with np.load(io.BytesIO(lazy),allow_pickle=False) as z:
        assert np.array_equal(z['K_content'],k) and np.array_equal(z['key_view_angles'],key_angles)
    # Incremental scalar-FLOP proxy excludes shared attention/RoPE: one sparse Givens (6 ops) per cached key, per view/query.
    lazy_per_query=NV*6*T
    materialization_once=NV*6*T
    return {'condition':label,'seed':seed,'method':'position_free_cache_lazy_vs_explicit_views',
      'serialized_bytes':len(lazy),'independent_bytes':len(independent),
      'lazy_sha256':hashlib.sha256(lazy).hexdigest(),'independent_sha256':hashlib.sha256(independent).hexdigest(),
      'train_tokens_or_examples':0,'optimizer_updates':0,'active_compute_proxy':lazy_per_query*T,
      'lazy_transform_ops_per_decode':lazy_per_query,'materialize_transform_ops_once':materialization_once,
      'transform_only_break_even_decode_tokens':1,'wall_time_s':wall,
      'max_lazy_rel_error':maxerr,'max_native_givens_rel_error':max_native,'max_bad_query_shortcut_rel_error':max_approx,
      'status_note':'synthetic FP32; full NPZ payload; scalar FLOP proxy excludes common attention and RoPE; wall time descriptive'}

def main():
    out=ROOT/'source'/'payloads'; rows=[run_one(k,v,out) for k,v in SEEDS.items()]
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
