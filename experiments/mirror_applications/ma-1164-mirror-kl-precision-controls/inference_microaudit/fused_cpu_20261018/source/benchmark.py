"""Reproducible CPU eager/Numba/dense K-way implicit Mirror benchmark.

Research-only performance diagnostic. All setup and JIT compilation are excluded
from the steady-state forward latency, but one-time startup cost is measured.
"""
import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from numba import set_num_threads, get_num_threads
import torch

HERE = Path(__file__).resolve().parents[1]
legacy_source = HERE.parent / 'source'
if not legacy_source.exists():
    legacy_source = Path(os.environ['MA1164_MICROAUDIT_SOURCE'])
sys.path.insert(0, str(legacy_source))
from online_operator import hadamard, signs, materialize, dense_forward, implicit_forward, direct_nonlinear_sandwich
from fused_online import forward


def med(fn, iterations=21, warmups=3):
    for _ in range(warmups):fn()
    values=[]
    for _ in range(iterations):
        t=time.perf_counter_ns()
        fn()
        values.append((time.perf_counter_ns()-t)*1e-6)
    return statistics.median(values), float(np.percentile(values,90)), float(np.percentile(values,10))


def nonlinear_old(x,u,code,K,R):
    h=x
    for r in range(R):
        z=torch.zeros_like(h)
        for k in range(K):
            z+=direct_nonlinear_sandwich(h,u,code[r*K+k])
        h=h+0.2*z/K
    return h


def cpu_label():
    try:
        lines=Path('/proc/cpuinfo').read_text().splitlines()
        return next(s.split(':',1)[1].strip() for s in lines if s.startswith('model name'))
    except Exception:
        return 'unknown'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default=str(HERE/'results'/'benchmark.csv'))
    parser.add_argument('--reps',type=int,default=21)
    args=parser.parse_args()
    torch.set_num_threads(1)
    set_num_threads(1)
    torch.manual_seed(20261018)
    old_threads = int(get_num_threads())
    dims=(64,128,256,512)
    modes=((1,1),(4,4))
    batches=(1,64)
    cases=[]
    for d in dims:
        H=hadamard(torch.eye(d,dtype=torch.float64))
        u=torch.randn(d,dtype=torch.float64)*0.045
        for K,R in modes:
            signs_t=[signs(r,k,d) for r in range(R) for k in range(K)]
            signs_np=np.stack([t.numpy().astype(np.int8) for t in signs_t]).reshape(R,K,d)
            W=torch.stack([materialize(u,s,H) for s in signs_t])
            for batch in batches:
                x=torch.randn(batch,d,dtype=torch.float64)
                x_np=x.numpy()
                with torch.no_grad():
                    actual=torch.from_numpy(forward(x_np,u.numpy(),signs_np))
                    ref=dense_forward(x,W,K,R)
                maxerr=(actual-ref).abs().max().item()
                assert maxerr<2e-11, (d,K,R,batch,maxerr)
                with torch.no_grad():
                    old_fn=lambda:implicit_forward(x,u,signs_t,K,R)
                    dense_fn=lambda:dense_forward(x,W,K,R)
                    old_ms,old_p90,_=med(old_fn,args.reps)
                    dense_ms,dense_p90,_=med(dense_fn,args.reps)
                set_num_threads(1)
                fast_fn=lambda:forward(x_np,u.numpy(),signs_np,parallel=False)
                fused_ms,fused_p90,_=med(fast_fn,args.reps)
                set_num_threads(4)
                par_fn=lambda:forward(x_np,u.numpy(),signs_np,parallel=True)
                parallel_ms,par_p90,_=med(par_fn,args.reps)
                row=dict(d=d,batch=batch,K=K,R=R,dtype='float64',
                    old_torch_ms_median=old_ms,old_torch_ms_p90=old_p90,
                    preexpanded_dense_ms_median=dense_ms,preexpanded_dense_ms_p90=dense_p90,
                    fused_serial_ms_median=fused_ms,fused_serial_ms_p90=fused_p90,
                    fused_parallel4_ms_median=parallel_ms,fused_parallel4_ms_p90=par_p90,
                    old_to_fused_speedup=old_ms/fused_ms,
                    dense_to_fused_speedup=dense_ms/fused_ms,
                    implicit_physical_bytes=u.numel()*8,
                    sign_code_bytes=signs_np.nbytes,
                    equivalent_dense_matrices_bytes=W.numel()*8,
                    virtual_weight_matrix_allocated_in_fused=False,
                    max_abs_error=maxerr,
                    timing_repetitions=args.reps,
                    torch_cpu_threads=1, numba_serial_threads=1,numba_parallel_threads=4)
                cases.append(row)
                print('d={d} B={batch} K={K} R={R} old={old_torch_ms_median:.4f}ms '
                      'fused={fused_serial_ms_median:.4f}ms par={fused_parallel4_ms_median:.4f}ms '
                      'dense={preexpanded_dense_ms_median:.4f}ms '
                      'speedup={old_to_fused_speedup:.1f}x'.format(**row),flush=True)
    # Genuine Q^-1 tanh(diag(u) Q x) nonlinear Mirror, output-recurrent K/R structure.
    nonlinear=[]
    for d,B,K,R in [(64,64,1,1),(128,64,4,4),(256,64,4,4),(512,64,4,4)]:
        x=torch.randn(B,d,dtype=torch.float64)
        u=torch.randn(d,dtype=torch.float64)*.1
        st=[signs(r,k,d) for r in range(R) for k in range(K)]
        sn=np.stack([s.numpy().astype(np.int8) for s in st]).reshape(R,K,d)
        with torch.no_grad():
            old=nonlinear_old(x,u,st,K,R)
            compiled=forward(x.numpy(),u.numpy(),sn,nonlinear_inside=True)
            maxerr=np.max(np.abs(compiled-old.numpy()))
            assert maxerr<2e-11
            old_ms,old_p90,_=med(lambda:nonlinear_old(x,u,st,K,R),args.reps)
        set_num_threads(1)
        fast_ms,fast_p90,_=med(lambda:forward(x.numpy(),u.numpy(),sn,nonlinear_inside=True),args.reps)
        nonlinear.append(dict(d=d,batch=B,K=K,R=R,max_error=float(maxerr),
                              old_ms=old_ms,fused_ms=fast_ms,speedup=old_ms/fast_ms,
                              old_p90=old_p90,fused_p90=fast_p90))
    output=Path(args.out)
    output.parent.mkdir(parents=True,exist_ok=True)
    nonlinear_path=output.with_name(output.stem+'_nonlinear.csv')
    meta_path=output.with_suffix('.meta.json')
    for dest in (output,nonlinear_path,meta_path):
        if dest.exists():raise FileExistsError('Never overwrite benchmark evidence: '+str(dest))
    with output.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(cases[0].keys()));wr.writeheader();wr.writerows(cases)
    nonlinear_path.write_text(
        '\n'.join([','.join(nonlinear[0].keys())]+[','.join(str(r[k]) for k in nonlinear[0]) for r in nonlinear])+'\n')
    metadata={'date':'2026-10-18','backend':'PyTorch eager 2.10.0+cpu + Numba nopython LLVM compiled',
        'python':sys.version,'numba':__import__('numba').__version__,
        'numpy':np.__version__,'torch':torch.__version__,'hardware':cpu_label(),
        'cpu_quota':os.cpu_count(),'cpu_torch_threads':1,'numba_threads':(1,4),
        'data':'deterministic synthetic FP64','reps':args.reps,
        'clocks':'uncontrolled container VM',
        'benchmark_scope':'steady-state forward only excluding JIT, code generation and transfers',
        'memory_scope':'theoretical per-weight arrays; no sampled peak RSS/alloc trace',
        'nonlinear_cases':nonlinear,'results_rows':len(cases),'absolute_error_max':max(z['max_abs_error'] for z in cases)}
    meta_path.write_text(json.dumps(metadata,indent=2,default=str))
    print('metadata',metadata['hardware'], 'torch',metadata['torch'],'numba',metadata['numba'])

if __name__=='__main__':main()
