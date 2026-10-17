"""Proof-of-execution for on-the-fly Mirror linear operators (not a trained model).

Implicit: W_{r,k} x = D_{r,k} H diag(u) H D_{r,k} x,
where H is normalized orthogonal Walsh-Hadamard, D is a deterministic sign map.
One shared physical coefficient vector u for all K x R heads.
No W is generated or stored in the implicit path. All data and signs unitless.
"""
import csv
import json
import math
import statistics
import time
from pathlib import Path
import torch

OUT=Path(__file__).resolve().parents[1] / 'results'
OUT.mkdir(parents=True, exist_ok=True)

def hadamard(x):
    n=x.shape[-1]
    assert n and (n&(n-1))==0
    h=1
    while h<n:
        y=x.reshape(*x.shape[:-1],n//(2*h),2,h)
        x=torch.stack((y[...,0,:]+y[...,1,:], y[...,0,:]-y[...,1,:]),dim=-2).reshape(*x.shape[:-1],n)
        h*=2
    return x/math.sqrt(n)


def signs(r,k,n):
    g=torch.Generator().manual_seed(104729+7919*r+113*k)
    return (torch.randint(0,2,(n,),generator=g,dtype=torch.int8).to(torch.float64)*2-1)


def direct(x,u,sign):
    return sign*hadamard(u*hadamard(x*sign))


def direct_nonlinear_sandwich(x,u,sign):
    # Q^-1 phi(diag(u) Qx); no dense Q or W is materialized.
    return sign*hadamard(torch.tanh(u*hadamard(x*sign)))


def dense_nonlinear_sandwich(x,u,sign,h):
    # Reference uses the explicit Q=H D and its transpose, unlike direct_nonlinear_sandwich.
    Q=h*sign[None,:]
    return torch.tanh((x@Q.T)*u)@Q


def materialize(u,sign,h):
    # W=D H diag(u) H D; our FWHT is symmetric and involutory.
    base = (h*u[None,:])@h
    return sign[:,None]*base*sign[None,:]


def dense_forward(x, weights, k, rounds):
    z=x
    for r in range(rounds):
        y=torch.zeros_like(z)
        for j in range(k):y += z@weights[r*k+j].T
        z = z+0.2*torch.tanh(y/k)
    return z


def implicit_forward(x,u,scode,k,rounds):
    z=x
    for r in range(rounds):
        y=torch.zeros_like(z)
        for j in range(k):y+=direct(z,u,scode[r*k+j])
        z=z+0.2*torch.tanh(y/k)
    return z


def med_ms(fn,n=19):
    with torch.no_grad():
        for _ in range(3):fn()
        t=[]
        for _ in range(n):
            a=time.perf_counter();fn();t.append(1000*(time.perf_counter()-a))
    return statistics.median(t)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(4242)
    dims=[64,128,256]
    modes=[(1,1),(4,4)]
    rows=[]
    for d in dims:
        # Hadamard matrix only used for reference; direct kernel needs NO dxd buffer.
        H=hadamard(torch.eye(d,dtype=torch.float64))
        physical=torch.randn(d,dtype=torch.float64)/math.sqrt(d)
        x=torch.randn(64,d,dtype=torch.float64)
        for k,r in modes:
            scode=[signs(i,j,d) for i in range(r) for j in range(k)]
            weights=torch.stack([materialize(physical,s,H) for s in scode])
            with torch.no_grad():
                a=implicit_forward(x,physical,scode,k,r)
                b=dense_forward(x,weights,k,r)
                max_diff=float((a-b).abs().max())
            assert max_diff<1e-11, (d,k,r,max_diff)
            direct_ms=med_ms(lambda:implicit_forward(x,physical,scode,k,r))
            dense_ms=med_ms(lambda:dense_forward(x,weights,k,r))
            # Optional sign codes can be generated from a 32-bit deterministic seed, plus K/R metadata.
            # Expanded tensors below are transient in reference only; not in implicit forward.
            row={'d':d,'batch':64,'K':k,'R':r,'physical_coeff_bytes_FP64':8*d,
                 'explicit_weights_bytes_FP64':weights.numel()*8,
                 'implicit_operator_temporary_virtual_W_bytes':0,
                 'direct_ms_median_CPU':round(direct_ms,5),'preexpanded_matmul_ms_median_CPU':round(dense_ms,5),
                 'implicit_over_dense_ratio':round(direct_ms/dense_ms,3),
                 'max_abs_output_difference':max_diff}
            rows.append(row)
            print(row,flush=True)
    with (OUT/'online_operator.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'online_operator_verification.json').write_text(json.dumps({'passed':len(rows),'rows':rows,
       'limitations':['Fixed Walsh-Hadamard generator plus signed addresses; cannot generate arbitrary dense trained weights',
                      'No direct supervised training, CPU eager FP64, clocks not fixed',
                      'Sign direction codes deterministic and generated from fixed seeds',
                      'Dense reference caches virtual weights; direct path avoids all W matrices',
                      'Activation function tanh and residual are in the recurrent computation, not a training experiment']},indent=2))

if __name__=='__main__':main()
