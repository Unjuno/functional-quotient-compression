#!/usr/bin/env python3
"""Few-shot task composition of shared synthetic LoRA atoms (MA-669)."""
from __future__ import annotations
import argparse,csv,io,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; DIN=4; DOUT=4; K=6; NOISE=.03

def atoms_for_world(seed):
    rng=np.random.default_rng(seed); left,_=np.linalg.qr(rng.normal(size=(DIN,DIN))); right,_=np.linalg.qr(rng.normal(size=(DOUT,DOUT)))
    pairs=rng.choice(DIN*DOUT,size=K,replace=False)
    return np.stack([np.outer(right[:,int(p//DIN)],left[:,int(p%DIN)]) for p in pairs])

def sample_tasks(seed,atoms,n=12,support_n=32,query_n=128):
    rng=np.random.default_rng(seed); out=[]
    for _ in range(n):
        c=np.zeros(K); ids=rng.choice(K,size=int(rng.choice([2,3])),replace=False);c[ids]=rng.uniform(.35,.9,size=len(ids))
        xs=rng.normal(size=(support_n,DIN));xq=rng.normal(size=(query_n,DIN));delta=np.einsum('k,koi->oi',c,atoms)
        ys=xs@delta.T+rng.normal(scale=NOISE,size=(support_n,DOUT));yq=xq@delta.T
        # Support statistic per expert; feature dimension equals coefficient dimension.
        z=np.stack([np.mean((xs@a.T)*ys) for a in atoms])
        out.append((c,delta,xs,ys,xq,yq,z))
    return out

def learn_map(atoms):
    dev=sample_tasks(66901,atoms,n=12)
    Z=np.stack([t[-1] for t in dev]);C=np.stack([t[0] for t in dev])
    # Ridge-regularized linear support-statistic -> Mirror code map, fit only on dev tasks.
    W=np.linalg.solve(Z.T@Z+1e-4*np.eye(K),Z.T@C)
    b=C.mean(0)-Z.mean(0)@W
    return W,b,dev

def direct_fit(atoms,x,y):
    basis=np.concatenate([(x@a.T).reshape(-1,1) for a in atoms],axis=1)
    target=y.reshape(-1)
    return np.linalg.lstsq(basis,target,rcond=None)[0]

def random_search(atoms,x,y,seed,budget=1024):
    rng=np.random.default_rng(seed);cand=rng.uniform(-.1,1.1,size=(budget,K));responses=np.stack([(x@a.T) for a in atoms]) # [K,N,O]
    pred=np.einsum('ck,kno->cno',cand,responses);loss=np.mean((pred-y[None,:,:])**2,axis=(1,2));return cand[np.argmin(loss)],budget

def predict(atoms,c,x):return x@np.einsum('k,koi->oi',c,atoms).T

def pack(d):
    f=io.BytesIO();np.savez(f,**d);return len(f.getvalue())

def evaluate(seed):
    atoms=atoms_for_world(66901);W,b,dev=learn_map(atoms);fresh=sample_tasks(seed,atoms,n=12)
    mirror_codes=[];direct_codes=[];search_codes=[];rows=[];t0=time.perf_counter()
    for i,(true_c,delta,xs,ys,xq,yq,z) in enumerate(fresh):
        cm=z@W+b;cd=direct_fit(atoms,xs,ys);ch,steps=random_search(atoms,xs,ys,seed+i)
        mirror_codes.append(cm);direct_codes.append(cd);search_codes.append(ch)
        estimates={'mirror_support_map':cm,'direct_least_squares':cd,'lorahub_random_search':ch,'independent_delta':None}
        for method,c in estimates.items():
            if c is None:pred=xq@delta.T;active=0
            else:pred=predict(atoms,c,xq);active=K if method=='direct_least_squares' else (int(np.count_nonzero(np.abs(c)>1e-6)) if method=='mirror_support_map' else int(np.count_nonzero(np.abs(c)>1e-6)))
            mse=float(np.mean((pred-yq)**2))
            support_pred=xs@delta.T if c is None else predict(atoms,c,xs)
            rows.append({'condition':'fresh_task_'+method,'world_or_seed':seed,'method':method,'serialized_bytes':0,'train_tokens_or_examples':32,'optimizer_updates':0,'active_compute_proxy':f'{active} expert projections per query; objective_evaluations={1024 if method=="lorahub_random_search" else 1}','wall_time_s':f'{(time.perf_counter()-t0):.9f}','primary_metric':'heldout_query_mse','primary_value':f'{mse:.12g}','secondary_metric':'support_mse;objective_evaluations','secondary_value':f'{np.mean((support_pred-ys)**2):.12g};{steps if method=="lorahub_random_search" else 1}','status_note':f'task={i};support=32;query=128;whole-task heldout;noise_sd={NOISE}'})
    # Charge all inference state across the 12-task bank.
    bank=atoms.astype(np.float32);mc=np.asarray(mirror_codes,dtype=np.float32);dc=np.asarray(direct_codes,dtype=np.float32);hc=np.asarray(search_codes,dtype=np.float32)
    bytes_mirror=pack({'shared_experts':bank,'support_to_code_W':W.astype(np.float32),'support_to_code_b':b.astype(np.float32),'task_codes':mc})
    bytes_search=pack({'shared_experts':bank,'task_coefficients':hc})
    bytes_direct=pack({'shared_experts':bank,'task_coefficients':dc})
    bytes_full=pack({'independent_task_deltas':np.stack([t[1] for t in fresh]).astype(np.float32)})
    for r in rows:
        r['serialized_bytes']={'mirror_support_map':bytes_mirror,'lorahub_random_search':bytes_search,'direct_least_squares':bytes_direct,'independent_delta':bytes_full}[r['method']]
    summary={'seed':seed,'tasks':12,'support_per_task':32,'query_per_task':128,'mirror_bytes':bytes_mirror,'lorahub_bytes':bytes_search,'direct_bytes':bytes_direct,'independent_delta_bytes':bytes_full,'mirror_lorahub_ratio':bytes_mirror/bytes_search,'mirror_codes':mc.tolist(),'wall_s':time.perf_counter()-t0}
    return rows,summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[66901]);p.add_argument('--out',default=str(ROOT/'RESULTS_CORE.csv'));p.add_argument('--summary',default=str(ROOT/'source'/'summary.json'));a=p.parse_args();rows=[];summ=[]
    for s in a.seeds:r,m=evaluate(s);rows+=r;summ.append(m)
    with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    Path(a.summary).write_text(json.dumps(summ,indent=2)+'\n');print(json.dumps(summ,indent=2))
if __name__=='__main__':main()
