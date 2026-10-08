#!/usr/bin/env python3
"""MA-427 synthetic task-conditioned contractive DEQ screen."""
from __future__ import annotations
import argparse,csv,io,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; D=4

def fixed_point(W,U,b,x,tol=1e-10,max_iter=1000):
    h=np.zeros_like(x,dtype=np.float64)
    for i in range(1,max_iter+1):
        nxt=np.tanh((W@h.T).T+(U@x.T).T+b)
        if np.max(np.abs(nxt-h))<tol:return nxt,i
        h=nxt
    return h,max_iter

def world(seed):
    rng=np.random.default_rng(seed)
    q,_=np.linalg.qr(rng.normal(size=(D,D))); eig=np.array([.30,.24,.18,.12]); W=q@np.diag(eig)@q.T
    U=np.eye(D)*.45; base=np.zeros(D)
    direction=rng.normal(size=D); direction/=np.linalg.norm(direction)
    # Four development task vectors identify a single shared direction.
    dev_scales=np.array([-.24,-.08,.11,.27]); dev_b=dev_scales[:,None]*direction
    # Centered rank-1 span estimated from development task identities only.
    _,_,vt=np.linalg.svd(dev_b-dev_b.mean(axis=0),full_matrices=False); V=vt[0];
    if V@direction<0:V=-V
    aligned_scales=rng.uniform(-.32,.32,size=8); aligned=aligned_scales[:,None]*direction
    boundary=rng.normal(size=(4,D)); boundary=boundary/np.linalg.norm(boundary,axis=1,keepdims=True)*rng.uniform(.18,.34,size=(4,1))
    tasks=[]
    for kind,bs in [('aligned',aligned),('boundary',boundary)]:
        for b in bs:
            xs=rng.normal(size=(128,D)); hs,_=fixed_point(W,U,b,xs)
            xt=rng.normal(size=(64,D)); ht,_=fixed_point(W,U,b,xt)
            tasks.append((kind,b,xs,hs,xt,ht))
    return W,U,base,V,tasks

def infer_bias(W,U,x,h):
    clipped=np.clip(h,-1+1e-12,1-1e-12)
    z=np.arctanh(clipped)-(W@h.T).T-(U@x.T).T
    return z.mean(axis=0)

def pack(arrays):
    bio=io.BytesIO();np.savez(bio,**arrays);return len(bio.getvalue())

def evaluate(seed):
    t0=time.perf_counter();W,U,base,V,tasks=world(seed); elapsed_setup=time.perf_counter()-t0
    n=len(tasks); mirror_codes=[]; full_biases=[]; estimates=[]
    for _,true_b,xs,hs,_,_ in tasks:
        bhat=infer_bias(W,U,xs,hs); estimates.append(bhat)
        mirror_codes.append(float((bhat-base)@V)); full_biases.append(bhat)
    mirror_codes=np.asarray(mirror_codes,dtype=np.float32);full_biases=np.asarray(full_biases,dtype=np.float32)
    packages={
      'shared_no_code':{'W':W.astype(np.float32),'U':U.astype(np.float32),'base_bias':base.astype(np.float32)},
      'mirror_rank1':{'W':W.astype(np.float32),'U':U.astype(np.float32),'base_bias':base.astype(np.float32),'direction':V.astype(np.float32),'task_codes':mirror_codes},
      'native_rank1_bias':{'W':W.astype(np.float32),'U':U.astype(np.float32),'base_bias':base.astype(np.float32),'direction':V.astype(np.float32),'task_codes':mirror_codes.copy()},
      'independent_full_bias':{'W':W.astype(np.float32),'U':U.astype(np.float32),'task_biases':full_biases},
    }
    sizes={k:pack(v) for k,v in packages.items()}
    rows=[]
    for ti,(kind,true_b,xs,hs,xt,ht) in enumerate(tasks):
        bhat=estimates[ti]; mc=V*mirror_codes[ti]+base
        biases={'shared_no_code':base,'mirror_rank1':mc,'native_rank1_bias':mc,'independent_full_bias':bhat}
        for method,b in biases.items():
            start=time.perf_counter();pred,it=fixed_point(W,U,b,xt); solve_time=time.perf_counter()-start
            mse=float(np.mean((pred-ht)**2)); resid=float(np.max(np.abs(pred-np.tanh((W@pred.T).T+(U@xt.T).T+b))))
            jac=float(np.linalg.norm(W,2))
            rows.append({'condition':kind+'_'+method,'world_or_seed':seed,'method':method,'serialized_bytes':sizes[method],'train_tokens_or_examples':128,'optimizer_updates':0,'active_compute_proxy':'per DEQ iteration: 4x4 W and U products + tanh; contraction bound <= 0.30','wall_time_s':f'{solve_time:.9f}','primary_metric':'heldout_fixed_point_mse','primary_value':f'{mse:.12g}','secondary_metric':'max_fixed_point_residual;iterations;spectral_radius_bound','secondary_value':f'{resid:.12g};{it};{jac:.12g}','status_note':f'task={ti};support_fit=128;heldout=64;development_world=one fixed rank1 direction'})
    return rows,{'seed':seed,'aligned_tasks':8,'boundary_tasks':4,'mirror_bank_bytes':sizes['mirror_rank1'],'native_rank1_bank_bytes':sizes['native_rank1_bias'],'independent_full_bias_bank_bytes':sizes['independent_full_bias'],'shared_no_code_bytes':sizes['shared_no_code'],'mirror_native_identical_payload':sizes['mirror_rank1']==sizes['native_rank1_bias'],'setup_wall_s':elapsed_setup,'support_examples':12*128,'updates':0}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,default=[42701]);ap.add_argument('--out',default=str(ROOT/'RESULTS_CORE.csv'));ap.add_argument('--summary',default=str(ROOT/'source'/'summary.json'));a=ap.parse_args();rows=[];summ=[]
    for seed in a.seeds:r,s=evaluate(seed);rows+=r;summ.append(s)
    with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    Path(a.summary).write_text(json.dumps(summ,indent=2)+'\n');print(json.dumps(summ,indent=2))
if __name__=='__main__':main()
