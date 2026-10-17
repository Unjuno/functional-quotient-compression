"""Posthoc replay timing of trained-digits Mirror compiler with fused CPU kernel.

Uses an EXISTING old teacher checkpoint. Fits compressed m=768 coordinates to
frozen teacher weights (NO label/logit fitting); old posthoc world can be used
for exact functional replay, NOT promoted to fresh dataset generalization.
"""
import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
import numpy as np
import torch
from numba import set_num_threads
from fused_digits import make_program,decode,infer

HERE=Path(__file__).resolve().parents[1]
# Can use --native-source externally, or resolved sibling after GitHub checkout.

def med(fn,reps=21):
    with torch.no_grad():
        for _ in range(4):fn()
        times=[]
        for _ in range(reps):
            start=time.perf_counter_ns()
            fn()
            times.append((time.perf_counter_ns()-start)*1e-6)
    return statistics.median(times),float(np.percentile(times,90))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo-experiment-source', default=str(HERE.parents[1]/'source'))
    p.add_argument('--old-checkpoint',required=True,help='Existing MA-1163 native_11.pt teacher checkpoint')
    p.add_argument('--out',default=str(HERE/'results'/'learned_digits_speed.csv'))
    p.add_argument('--reps',type=int,default=21)
    a=p.parse_args()
    sys.path.insert(0,a.repo_experiment_source)
    from core import NativeMLP,native_flat,logits
    from run import fit_teacher_weights,world_data
    set_num_threads(1)
    torch.set_num_threads(1)
    teacher=NativeMLP().eval()
    teacher.load_state_dict(torch.load(a.old_checkpoint,weights_only=True,map_location='cpu')['state'])
    flat=native_flat(teacher)
    model,fit=fit_teacher_weights(flat,kind='mirror',m=768,k=8,r=4)
    model.eval()
    u=model.coords.detach().numpy().copy()
    ix=model.indices.numpy().astype(np.int64)
    t0=time.perf_counter()
    program=make_program(8,4)
    build_s=time.perf_counter()-t0
    xs,ys,split=world_data(618201)
    x_all=xs[split['test']]
    true_labels=ys[split['test']]
    with torch.no_grad():
        logits_old=model(x_all)
        q=decode(u,ix,program)
        target_weights=model.weights().numpy()
        weight_maxerr=float(np.abs(q-target_weights).max())
        outputs=infer(u,ix,program,x_all.numpy())
        logit_maxerr=float(np.abs(outputs-logits_old.numpy()).max())
        assert weight_maxerr<1e-12 and logit_maxerr<2e-12
        acc_old=float((logits_old.argmax(-1)==true_labels).double().mean())
        acc_fast=float((torch.from_numpy(outputs).argmax(-1)==true_labels).double().mean())
        assert acc_old==acc_fast
        assert np.array_equal(logits_old.numpy().argmax(-1),outputs.argmax(-1))
    rows=[]
    for B in (1,64,256):
        x=x_all[:B].contiguous();xx=x.numpy()
        # Numba was warmed during the full replay, so JIT excluded from measurements.
        old_ms,old_p90=med(lambda:model(x),a.reps)
        fused_ms,fused_p90=med(lambda:infer(u,ix,program,xx),a.reps)
        with torch.no_grad():refweights=model.weights()
        mat_ms,mat_p90=med(lambda:logits(refweights,x),a.reps)
        row={'batch':B,'K':8,'R':4,'physical_coefficients':len(u),'virtual_weight_elements':1210,
             'old_torch_ms':old_ms,'old_torch_p90':old_p90,
             'fused_numba_ms':fused_ms,'fused_numba_p90':fused_p90,
             'cached_native_matmul_ms':mat_ms,'cached_native_matmul_p90':mat_p90,
             'speedup_over_old':old_ms/fused_ms,'speedup_vs_cached_dense':mat_ms/fused_ms,
             'max_abs_weights_diff':weight_maxerr,'max_abs_logits_diff':logit_maxerr,
             'checkpoint_replayed_accuracy':acc_fast,'fitting_used_labels':False,
             'old_test_split_reopened':True,'warm_program_generation_s':build_s,'torch_threads':1,'numba_threads':1}
        rows.append(row)
        print(f'digits B{B}: old {old_ms:.4f}ms fused {fused_ms:.4f}ms native cached {mat_ms:.4f}ms '
             f'original speedup {old_ms/fused_ms:.1f}x, weight err {weight_maxerr:.2g}',flush=True)
    output=Path(a.out);output.parent.mkdir(parents=True,exist_ok=True)
    verification_path=output.with_suffix('.verification.json')
    for dest in (output,verification_path):
        if dest.exists():raise FileExistsError('Never overwrite posthoc evidence: '+str(dest))
    with output.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    verification={'scope':'posthoc, previously evaluated split and checkpoint, no fresh claim',
        'base_checkpoint':a.old_checkpoint,'mirrored_K':8,'rounds':4,'m':768,
        'fitted_to_weights_only':True,'fit_wall_s':fit['fit_seconds'],'fit_calls':fit['fit_calls'],
        'compile_program_ms':1000*build_s,
        'accuracy_replayed':acc_fast,'weight_diff_max':weight_maxerr,'logit_diff_max':logit_maxerr,
        'hardware':platform.platform(),'precision':'FP64','CPU_backend':'Numba JIT vs PyTorch eager',
        'repeat':a.reps,'rows':rows,'caveats':[
            'Fused digits kernel retains one 1210-element transient virtual weight vector per inference; unlike the matrix-free Hadamard kernel, it is NOT truly virtual-weight-free.',
            'No accuracy gain; no GPU/VRAM benchmark; no backward graph in Numba.',
            'JIT startup and program precompute excluded from inference steady-state timing but precompute measured.',
            'Fitted model uses past labels only for reporting old held-out accuracy, never for fitting mirror coefficients.'
        ]}
    verification_path.write_text(json.dumps(verification,ensure_ascii=False,indent=2,default=float))
    print('accuracy',acc_fast,'fits',fit,'program build ms',1000*build_s)

if __name__=='__main__':main()
