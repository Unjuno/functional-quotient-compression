#!/usr/bin/env python3
"""MA-551 fused-kernel synthetic hierarchical FiLM/View screen."""
from __future__ import annotations
import argparse, ctypes, hashlib, json, subprocess, tempfile, time
from pathlib import Path
import numpy as np

D, N_FUNCS, RANK = 16, 32, 4
HELD = 1024


def build_kernel(src: Path):
    tmp=tempfile.TemporaryDirectory(prefix='ma551_')
    lib=Path(tmp.name)/'kernels.so'
    subprocess.run(['cc','-O3','-march=native','-shared','-fPIC',str(src),'-o',str(lib)],check=True)
    dll=ctypes.CDLL(str(lib)); ptr=ctypes.POINTER(ctypes.c_double)
    signatures={'film_apply':(4,2),'givens_fused':(6,2),'affine_apply':(4,2),'lowrank_apply':(6,3)}
    for name,(npointers,nints) in signatures.items():
        fn=getattr(dll,name);fn.argtypes=[ptr]*npointers+[ctypes.c_int]*nints;fn.restype=None
    return dll,tmp


def ptr(x): return x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def rotation(theta):
    r=np.eye(D,dtype=np.float64)
    for k,t in enumerate(theta):
        c,s=np.cos(t),np.sin(t);a,b=2*k,2*k+1
        r[a,a],r[a,b],r[b,a],r[b,b]=c,-s,s,c
    return r


def make_world(seed):
    rng=np.random.default_rng(seed);g=rng.uniform(.65,1.35,D);b=rng.normal(0,.15,D)
    theta=rng.uniform(-.6,.6,(N_FUNCS,D//2));xs=[];targets=[];affines=[];biases=[];films=[];lrus=[];lrvs=[]
    for i in range(N_FUNCS):
        r=rotation(theta[i]);a=r@np.diag(g);bi=r@b
        x=rng.normal(size=(HELD,D));y=x@a.T+bi
        xs.append(x);targets.append(y);affines.append(a);biases.append(bi)
        films.append((np.diag(a).copy(),bi.copy()))
        u,s,vt=np.linalg.svd(a-np.diag(g),full_matrices=False)
        lru=u[:,:RANK]*s[:RANK];lrv=vt[:RANK]
        lrus.append(lru);lrvs.append(lrv)
    return g,b,theta,np.stack(xs),np.stack(targets),np.stack(affines),np.stack(biases),np.stack(films),np.stack(lrus),np.stack(lrvs)


def npz(path,arrays):
    np.savez(path,**arrays);return path.stat().st_size


def run(seed,split,out,source):
    out.mkdir(parents=True,exist_ok=True);t_compile=time.perf_counter();kernel,tmp=build_kernel(source);compile_s=time.perf_counter()-t_compile
    t_reconstruct=time.perf_counter()
    g,b,theta,xs,ys,affines,biases,films,lr_u,lr_v=make_world(seed)
    cs=np.cos(theta);ss=np.sin(theta);reconstruct_s=time.perf_counter()-t_reconstruct
    common={'function_ids':np.arange(N_FUNCS,dtype=np.int16),'seed':np.array([seed],np.int64),'schema':np.array([551,1],np.int32)}
    payloads={
      'tied_film':{'gamma':g.astype(np.float32),'beta':b.astype(np.float32),**common},
      'independent_film':{'gamma':films[:,0].astype(np.float32),'beta':films[:,1].astype(np.float32),**common},
      'mirror_hierarchy':{'gamma':g.astype(np.float32),'beta':b.astype(np.float32),'angles':theta.astype(np.float32),'pairs':np.arange(D,dtype=np.int16).reshape(-1,2),**common},
      'native_givens':{'gamma':g.astype(np.float32),'beta':b.astype(np.float32),'angles':theta.astype(np.float32),'pairs':np.arange(D,dtype=np.int16).reshape(-1,2),**common},
      'lowrank_r4':{'gamma':g.astype(np.float32),'u':lr_u.astype(np.float32),'v':lr_v.astype(np.float32),'beta':biases.astype(np.float32),'function_ids':common['function_ids'],'seed':common['seed'],'schema':np.array([551,2],np.int32)},
      'full_affine':{'matrices':affines.astype(np.float32),'biases':biases.astype(np.float32),**common}}
    nbytes={m:npz(out/f'{m}.npz',p) for m,p in payloads.items()}
    errors={m:[] for m in payloads};timings={m:0. for m in payloads};throughput={}
    outbuf=[np.empty_like(ys[i]) for i in range(N_FUNCS)]
    def apply(method,i,x,y):
        if method=='tied_film': kernel.film_apply(ptr(x),ptr(g),ptr(b),ptr(y),ctypes.c_int(len(x)),ctypes.c_int(D))
        elif method=='independent_film': kernel.film_apply(ptr(x),ptr(films[i,0]),ptr(films[i,1]),ptr(y),ctypes.c_int(len(x)),ctypes.c_int(D))
        elif method in ('mirror_hierarchy','native_givens'):
            kernel.givens_fused(ptr(x),ptr(g),ptr(b),ptr(cs[i]),ptr(ss[i]),ptr(y),ctypes.c_int(len(x)),ctypes.c_int(D))
        elif method=='full_affine': kernel.affine_apply(ptr(x),ptr(affines[i]),ptr(biases[i]),ptr(y),ctypes.c_int(len(x)),ctypes.c_int(D))
        elif method=='lowrank_r4': kernel.lowrank_apply(ptr(x),ptr(g),ptr(biases[i]),ptr(lr_u[i]),ptr(lr_v[i]),ptr(y),ctypes.c_int(len(x)),ctypes.c_int(D),ctypes.c_int(RANK))
    methods=tuple(payloads)
    for method in methods:
        # Every generated input is held out from the analytically encoded operator state.
        for i in range(N_FUNCS):
            x=xs[i];truth=ys[i];y=np.empty_like(truth);apply(method,i,x,y)
            errors[method].append(float(np.linalg.norm(y-truth)/max(np.linalg.norm(truth),1e-12)))
        for _ in range(10):
            for i in range(N_FUNCS): apply(method,i,xs[i],outbuf[i])
        t0=time.perf_counter()
        for _ in range(30):
            for i in range(N_FUNCS): apply(method,i,xs[i],outbuf[i])
        timings[method]=(time.perf_counter()-t0)/30
        throughput[method]=N_FUNCS*HELD/timings[method]
    reference=throughput['independent_film']
    metrics={m:{'payload_bytes':nbytes[m],'mean_nrmse':float(np.mean(errors[m])),'max_function_nrmse':float(np.max(errors[m])),'per_function_nrmse':errors[m],'bank_seconds':timings[m],'throughput_examples_per_second':throughput[m],'throughput_ratio_to_film':throughput[m]/reference} for m in methods}
    report={'experiment_id':'MA-551','seed':seed,'split':split,'width':D,'functions':N_FUNCS,'heldout_vectors_per_function':HELD,'methods':metrics,'compute':{'timed_examples_per_method':N_FUNCS*HELD,'warmup_calls':10,'timed_calls':30,'optimizer_updates':0,'kernel_compile_seconds':compile_s,'code_reconstruction_and_trig_setup_seconds':reconstruct_s,'decode_flops_proxy_per_function':{'tied_film':2*D,'independent_film':2*D,'mirror_hierarchy':5*D,'native_givens':5*D,'lowrank_r4':2*D+4*D*RANK,'full_affine':2*D*D+2*D}},'exact_alias_audit':{'mirror_native_payload_arrays_equal':all(np.array_equal(payloads['mirror_hierarchy'][k],payloads['native_givens'][k]) for k in ('gamma','beta','angles','pairs','function_ids','seed')),'mirror_native_function_outputs_equal':errors['mirror_hierarchy']==errors['native_givens']}}
    (out/'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    manifest={'g':g.tolist(),'b':b.tolist(),'angles':theta.tolist(),'input_shape':[N_FUNCS,HELD,D],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    (out/'world.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'seed':seed,'split':split,'methods':{m:{k:v for k,v in d.items() if k!='per_function_nrmse'} for m,d in metrics.items()}},indent=2))
    tmp.cleanup()


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['dev','fresh'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,default=Path(__file__).with_name('kernels.c'));a=p.parse_args();run(a.seed,a.split,a.out,a.source)

if __name__=='__main__':main()
