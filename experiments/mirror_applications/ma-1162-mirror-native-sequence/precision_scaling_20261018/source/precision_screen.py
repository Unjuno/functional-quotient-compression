"""One-scalar arbitrary-precision nonlinear Mirror orbit, phase capacity and cost probe.

All model state is exactly ONE N/2**p. Mirror programs are fixed, non-learned
algorithms; no hidden per-head trainable parameters or bit-field optimizer.
Gauss-Newton uses a ONE-dimensional chain-rule gradient with p-bit rounding.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import mpmath as mp

FREQS = [1,2,3,5,7,11,13,17]
PHASE_ADDRESSES = [0,3,16,48,96,192,384,768,7000]
PRECISIONS = [16,32,53,64,128,256,800,8000]
TEACHER_BITS = 8192
TEACHER_SEED = 20261018


def exact_teacher():
    # Same denominator at every precision, no learned hidden coefficients.
    rng = random.Random(TEACHER_SEED)
    n = rng.getrandbits(TEACHER_BITS)
    return mp.mpf(n) / mp.power(2, TEACHER_BITS)


def quantize(s,p):
    n = int(mp.nint(s * mp.power(2,p)))
    n = min(max(n,0), (1<<p)-1)
    return mp.mpf(n) / mp.power(2,p), n


def mirror_rotate(z,theta):
    c = mp.cos(theta); si = mp.sin(theta)
    return (c*z[0]-si*z[1], si*z[0]+c*z[1])


def virtual_one_with_derivative(s, role, freq=None, phase=None):
    """One role's two nonlinear Mirror stages and EXACT chain-rule tangent."""
    if freq is None: freq=FREQS[role]
    if phase is None: phase=mp.mpf(role+1)/37
    z=(mp.mpf('0.37'), mp.mpf('-0.68'))
    dz=(mp.mpf(0),mp.mpf(0))
    theta_grad=2*mp.pi*freq
    for layer in (0,1):
        theta=2*mp.pi*(freq*s+phase+mp.mpf(layer)/19)
        c,si=mp.cos(theta),mp.sin(theta)
        v=(c*z[0]-si*z[1],si*z[0]+c*z[1])
        dv=(c*dz[0]-si*dz[1]+theta_grad*(-si*z[0]-c*z[1]),
            si*dz[0]+c*dz[1]+theta_grad*(c*z[0]-si*z[1]))
        bias=(mp.mpf('0.19')+(role%3)*mp.mpf('0.017'),mp.mpf('-0.13'))
        active=(mp.tanh(mp.mpf('1.15')*v[0]+bias[0]),mp.tanh(mp.mpf('1.15')*v[1]+bias[1]))
        da=(mp.mpf('1.15')*(1-active[0]**2)*dv[0],mp.mpf('1.15')*(1-active[1]**2)*dv[1])
        mirrored=(c*active[0]+si*active[1],-si*active[0]+c*active[1])
        dmirrored=(c*da[0]+si*da[1]+theta_grad*(-si*active[0]+c*active[1]),
                  -si*da[0]+c*da[1]+theta_grad*(-c*active[0]-si*active[1]))
        z=(mp.mpf('0.73')*z[0]+mp.mpf('0.27')*mirrored[0],
           mp.mpf('0.73')*z[1]+mp.mpf('0.27')*mirrored[1])
        dz=(mp.mpf('0.73')*dz[0]+mp.mpf('0.27')*dmirrored[0],
            mp.mpf('0.73')*dz[1]+mp.mpf('0.27')*dmirrored[1])
    return (z[0]+mp.mpf('0.17')*z[1],dz[0]+mp.mpf('0.17')*dz[1])


def virtual_one(s, role, freq=None, phase=None):
    return virtual_one_with_derivative(s,role,freq,phase)[0]


def virtual_vector(s):
    return [virtual_one(s,j) for j in range(len(FREQS))]


def target_offorbit(sstar, teacher):
    # Create an orthogonal perturbation to the tangent at the true teacher.
    tangent=[virtual_one_with_derivative(sstar,j)[1] for j in range(len(FREQS))]
    v=[mp.mpf((j*7)%9-4) for j in range(len(FREQS))]
    dot=sum(a*b for a,b in zip(tangent,v))
    dot2=sum(a*a for a in tangent)
    v=[b-a*dot/dot2 for a,b in zip(tangent,v)]
    norm=mp.sqrt(sum(b*b for b in v))
    return [a+mp.mpf('0.2')*b/norm for a,b in zip(teacher,v)]


def error(s,target):
    pred=virtual_vector(s)
    return sum((a-b)**2 for a,b in zip(pred,target))/len(pred)


def scalar_train(p, target, sstar, max_steps=24):
    # One-dimensional Gauss-Newton, not optimization of individual virtual weights.
    s, n = quantize(sstar+mp.mpf(1)/64,p)
    initial = error(s,target)
    accepted=0
    started=time.perf_counter()
    for step in range(max_steps):
        pred=virtual_vector(s)
        deriv=[virtual_one_with_derivative(s,j)[1] for j in range(len(FREQS))]
        g=sum((a-b)*d for a,b,d in zip(pred,target,deriv))
        curvature=sum(d*d for d in deriv)
        if curvature == 0:break
        delta=-g/curvature
        base=error(s,target)
        improved=False
        for shrink in range(17):
            sc,nc=quantize(s+delta/(2**shrink),p)
            if nc==n:
                continue
            ec=error(sc,target)
            if ec < base:
                s,n=sc,nc; improved=True; accepted+=1; break
        if not improved:
            break
    train_s=time.perf_counter()-started
    return {'steps_accepted':accepted,'initial_mse':initial,'final_mse':error(s,target),
            'distance_to_true_s':mp.fabs(s-sstar),'elapsed_s':train_s,'n':n}


def phase_value_from_fixed_integer(n,p,j,offset=0):
    # Equivalent to sin/cos(2pi*2^j*s) but avoids p-bit trig argument reduction.
    residue=((n<<j) & ((1<<p)-1))
    theta=2*mp.pi*(mp.mpf(residue)/mp.power(2,p)+mp.mpf(offset+1)/37)
    z=(mp.mpf('.37'),mp.mpf('-.68'))
    rotated=mirror_rotate(z,theta)
    active=(mp.tanh(rotated[0]+mp.mpf('.19')),mp.tanh(rotated[1]-mp.mpf('.13')))
    unrot=mirror_rotate(active,-theta)
    return unrot[0]+mp.mpf('.17')*unrot[1]


def sensitivity_rows(p,n):
    rows=[]
    for j in PHASE_ADDRESSES:
        y0=phase_value_from_fixed_integer(n,p,j,j%7)
        y1=phase_value_from_fixed_integer(n+1,p,j,j%7)
        diff=mp.fabs(y0-y1)
        rows.append({'precision_bits':p,'address_j':j,'phase_active':j<p,
                     'response_delta_one_ulp':mp.nstr(diff,13),
                     'exactly_zero':bool(diff==0),
                     'frequency_log2':j,'stored_model_scalars':1})
    return rows


def one_forward_time(p,n,repeats):
    # Pure mpmath CPU, fixed 8-role nonlinear Mirror program.
    with mp.workprec(max(96,p+40)):
        s=mp.mpf(n)/mp.power(2,p)
        for _ in range(2): virtual_vector(s)
        samples=[]
        for _ in range(repeats):
            start=time.perf_counter()
            virtual_vector(s)
            samples.append((time.perf_counter()-start)*1e3)
    return statistics.median(samples)


def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with mp.workprec(TEACHER_BITS+200):
        sstar=exact_teacher()
        teacher=virtual_vector(sstar)
        off=target_offorbit(sstar,teacher)
    runtime_rows=[]; training_rows=[]; phase_rows=[]
    for p in PRECISIONS:
            with mp.workprec(max(160,p+160)):
                sq,n=quantize(sstar,p)
                quantization_rmse=mp.sqrt(error(sq,teacher))
                # training uses the exact same target and architecture for all p
                for task,target in [('aligned',teacher),('off_orbit',off)]:
                    result=scalar_train(p,target,sstar)
                    training_rows.append({'precision_bits':p,'task':task,
                       'initial_mse':mp.nstr(result['initial_mse'],16),
                       'final_mse':mp.nstr(result['final_mse'],16),
                       'distance_to_true_s':mp.nstr(result['distance_to_true_s'],12),
                       'steps_accepted':result['steps_accepted'],
                       'train_seconds':round(result['elapsed_s'],5),
                       'trainable_scalars':1,'virtual_weights':len(FREQS)})
                phase_rows.extend(sensitivity_rows(p,n))
                print('optimized bits',p,flush=True)
            latency=one_forward_time(p,n,repeats=7 if p<=800 else 3)
            runtime_rows.append({'precision_bits':p, 'scalar_payload_bytes':(p+7)//8,
                      'ideal_scalar_kib':round((p+7)//8/1024,8),
                      'forward_median_ms_cpu':round(latency,5),
                      'virtual_roles':len(FREQS), 'quantization_teacher_rmse':mp.nstr(quantization_rmse,14),
                      'float64_roundtrip_same_as_53bit':float(sq)==float(quantize(sstar,53)[0]),
                      'math_backend':'mpmath CPU mpf, precision p+40 bits for timing',
                      'hardware':'CPU container, no CUDA'})
    for fn,rs in [('results/train.csv',training_rows),('results/phase_sensitivity.csv',phase_rows),('results/runtime.csv',runtime_rows)]:
            file=root/fn;file.parent.mkdir(parents=True,exist_ok=True)
            with open(file,'w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rs[0].keys()); writer.writeheader();writer.writerows(rs)
    meta={'precision_sweep':PRECISIONS,'virtual_roles':len(FREQS),'phase_j':PHASE_ADDRESSES,
              'teacher_bits':TEACHER_BITS,'true_scalar_hex_first_64bit':hex(int(sstar*2**64)),
              'backend':'CPU arbitrary precision mpmath 1.3.0','trainable_independent_coordinates':1,
              'note':'No GPU timing, dense RNN quality, or general capacity claims.'}
    (root/'results/metadata.json').write_text(json.dumps(meta,indent=2))
    return training_rows,runtime_rows,phase_rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    tr,rt,ph=run(a.out)
    print('RESULTS')
    for x in rt:print(x)
    for x in tr:print(x['precision_bits'],x['task'],x['final_mse'])
