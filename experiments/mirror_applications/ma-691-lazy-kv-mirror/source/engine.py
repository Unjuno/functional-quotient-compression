from __future__ import annotations
import csv, json, math, time
from pathlib import Path
import torch

PREFIXES=(128,512,2048,8192,32768)
FRESH=(69101,69102,69103)
B,H,D=1,8,64
ANGLE_STD=.25
ERR_GATE=2e-6

def row_rot(x:torch.Tensor,theta:torch.Tensor)->torch.Tensor:
    p=x.reshape(*x.shape[:-1],x.shape[-1]//2,2)
    a,b=p[...,0],p[...,1]
    c,s=torch.cos(theta),torch.sin(theta)
    return torch.stack((c*a+s*b,-s*a+c*b),-1).reshape_as(x)

def attention(q,k,v):
    w=torch.softmax(torch.einsum('bhd,bhtd->bht',q,k)/math.sqrt(q.shape[-1]),-1)
    return torch.einsum('bht,bhtd->bhd',w,v)

def median_time(fn,it,warm=5):
    for _ in range(warm): fn()
    xs=[]
    for _ in range(it):
        t=time.perf_counter(); fn(); xs.append(time.perf_counter()-t)
    return sorted(xs)[len(xs)//2]

def iterations(T):
    return 20 if T<=2048 else (10 if T<=8192 else 4)

def run_seed(seed:int):
    torch.manual_seed(seed); torch.set_num_threads(1)
    angles_k=torch.randn(H,D//2)*ANGLE_STD
    angles_v=torch.randn(H,D//2)*ANGLE_STD
    rows=[]
    maxerr=0.0
    for T in PREFIXES:
        q=torch.randn(B,H,D)
        k=torch.randn(B,H,T,D)
        v=torch.randn(B,H,T,D)
        ak=angles_k[None,:,None,:]; av=angles_v[None,:,None,:]
        aq=angles_k[None,:,:]; ao=angles_v[None,:,:]
        km=row_rot(k,ak); vm=row_rot(v,av)
        native=attention(q,km,vm)
        lazy=row_rot(attention(row_rot(q,-aq),k,v),ao)
        err=float((native-lazy).abs().max()); maxerr=max(maxerr,err)
        it=iterations(T)
        def materialize(): return row_rot(k,ak),row_rot(v,av)
        def premat(): return attention(q,km,vm)
        def lazy_fn(): return row_rot(attention(row_rot(q,-aq),k,v),ao)
        tc=median_time(materialize,it)
        tn=median_time(premat,it)
        tl=median_time(lazy_fn,it)
        be=None if tl<=tn else tc/(tl-tn)
        cache_bytes=2*k.numel()*k.element_size()
        code_bytes=(angles_k.numel()+angles_v.numel())*angles_k.element_size()
        rows.append(dict(seed=seed,prefix=T,max_abs_error=err,
            canonical_cache_bytes=cache_bytes,per_view_code_bytes=code_bytes,
            materialize_once_ms=tc*1e3,prematerialized_attention_ms=tn*1e3,
            lazy_attention_ms=tl*1e3,break_even_generated_tokens=be))
    # RoPE commutation on same pair planes
    T=257
    k=torch.randn(B,H,T,D)
    pos=torch.arange(T,dtype=k.dtype)[:,None]
    freq=torch.exp(-torch.arange(D//2,dtype=k.dtype)[None,:]/(D//2)*7.0)
    rope=(pos*freq)[None,None,:,:]
    mirror=angles_k[None,:,None,:]
    rope_err=float((row_rot(row_rot(k,mirror),rope)-row_rot(row_rot(k,rope),mirror)).abs().max())
    # MLA absorption
    Tm,hm,dm,dc=1024,6,16,24
    C=torch.randn(B,Tm,dc)
    UK=torch.randn(hm,dc,dm)/math.sqrt(dc); UV=torch.randn(hm,dc,dm)/math.sqrt(dc)
    q=torch.randn(B,hm,dm)
    K=torch.einsum('btc,hcd->bhtd',C,UK); V=torch.einsum('btc,hcd->bhtd',C,UV)
    explicit=attention(q,K,V)
    qlat=torch.einsum('bhd,hcd->bhc',q,UK)
    w=torch.softmax(torch.einsum('bhc,btc->bht',qlat,C)/math.sqrt(dm),-1)
    clat=torch.einsum('bht,btc->bhc',w,C)
    absorbed=torch.einsum('bhc,hcd->bhd',clat,UV)
    mla_err=float((explicit-absorbed).abs().max())
    mla_full=2*K.numel()*K.element_size(); mla_lat=C.numel()*C.element_size()
    return rows,dict(seed=seed,lazy_max_abs_error=maxerr,rope_commutation_max_abs_error=rope_err,
        mla_absorption_max_abs_error=mla_err,mla_full_kv_bytes=mla_full,
        mla_latent_cache_bytes=mla_lat,mla_compression_ratio=mla_full/mla_lat)

def run_all(outdir:Path):
    outdir.mkdir(parents=True,exist_ok=True)
    rows=[]; checks=[]
    for seed in FRESH:
        r,c=run_seed(seed); rows+=r; checks.append(c)
    with (outdir/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (outdir/'CHECKS.json').write_text(json.dumps(checks,indent=2))
    verification={
      'experiment_id':'MA-691','fresh_seeds':list(FRESH),'torch':torch.__version__,
      'threads':torch.get_num_threads(),
      'lazy_exact_pass':all(c['lazy_max_abs_error']<=ERR_GATE for c in checks),
      'rope_commutation_pass':all(c['rope_commutation_max_abs_error']<=ERR_GATE for c in checks),
      'mla_absorption_pass':all(c['mla_absorption_max_abs_error']<=ERR_GATE for c in checks),
      'max_lazy_error':max(c['lazy_max_abs_error'] for c in checks),
      'max_rope_error':max(c['rope_commutation_max_abs_error'] for c in checks),
      'max_mla_error':max(c['mla_absorption_max_abs_error'] for c in checks),
      'runtime_boundary':'CPU one-thread eager PyTorch microbenchmark only'
    }
    (outdir/'VERIFICATION.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification,indent=2))

if __name__=='__main__':
    import sys
    run_all(Path(sys.argv[1] if len(sys.argv)>1 else '.'))
