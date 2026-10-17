"""Fixed dyadic scalar vs ordinary binary64 for recurrent nonlinear Mirror views.

Non-trained mechanism probe, not a capacity/accuracy experiment.
"""
import csv, math, random, time
from pathlib import Path
import mpmath as mp

PVALUES=[16,32,53,64,128,256,800]
SEED=20261018
BITS=2048

def mirror_step(z, theta):
    c=mp.cos(theta); si=mp.sin(theta)
    # Q z, tanh, Q^-1, residual. All transformation parameters are fixed.
    a,b=c*z[0]-si*z[1], si*z[0]+c*z[1]
    a,b=mp.tanh(a+mp.mpf('0.13')),mp.tanh(b-mp.mpf('0.27'))
    v=(c*a+si*b, -si*a+c*b)
    return (mp.mpf('0.7')*z[0]+mp.mpf('0.3')*v[0],
            mp.mpf('0.7')*z[1]+mp.mpf('0.3')*v[1])

def mirror_program(s):
    z=(mp.mpf('0.18'),mp.mpf('-0.42'))
    for t in range(4):
        z=mirror_step(z,2*mp.pi*(mp.mpf(7+3*t)*s + mp.mpf(t+1)/19))
    return z[0] + mp.mpf('0.31')*z[1]

def run():
    rng=random.Random(SEED)
    n_star=rng.getrandbits(BITS)
    # exact rational x=n_star/2**BITS: nested dyadic approximations
    rows=[]
    for p in PVALUES:
        n=n_star>>(BITS-p)
        n_child=n_star>>(BITS-p-1)
        assert n_child in (2*n,2*n+1)
        with mp.workprec(max(128,p+125)):
            s=mp.mpf(n)/mp.power(2,p)
            neighbor=mp.mpf(n+1)/mp.power(2,p)
            child=mp.mpf(n_child)/mp.power(2,p+1)
            assert neighbor-s==mp.power(2,-p)
            assert child-s in (0,mp.power(2,-p-1))
            start=time.perf_counter()
            y=mirror_program(s)
            yp=mirror_program(neighbor)
            yc=mirror_program(child)
            wall=(time.perf_counter()-start)*1000
            abs_delta=mp.fabs(yp-y)
            child_delta=mp.fabs(yc-y)
            floats_equal=(float(s)==float(neighbor))
            rows.append(dict(fractional_bits=p,storage_bytes=math.ceil(p/8),
                scalar_step=mp.nstr(neighbor-s,12),
                prefix_extension_step=mp.nstr(child-s,12),
                mirror_response_neighbor=mp.nstr(abs_delta,12),
                mirror_response_prefix=mp.nstr(child_delta,12),
                local_sensitivity=mp.nstr(abs_delta/(neighbor-s),12),
                binary64_neighbor_collapses=floats_equal,
                three_forward_wall_ms=round(wall,4),
                program='4 x Q^-1(tanh(Qz+b)) residual, frequency 7,10,13,16'))
    root=Path(__file__).resolve().parent
    with (root/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for r in rows:
        print(f"p={r['fractional_bits']:4} dyadic-step={r['scalar_step']:<17} "
              f"out-delta={r['mirror_response_neighbor']:<19} "
              f"local-sensitivity={r['local_sensitivity']:<15} "
              f"FP64_collapsed={r['binary64_neighbor_collapses']}")
    return rows

if __name__=='__main__': run()
