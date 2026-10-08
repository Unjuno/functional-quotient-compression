#!/usr/bin/env python3
"""Gauge-invariant synthetic LoRA update atom screen for MA-674."""
from __future__ import annotations
import argparse,csv,io,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=8;K=6

def make_atoms(rng):
    left,_=np.linalg.qr(rng.normal(size=(D,K))); right,_=np.linalg.qr(rng.normal(size=(D,K)))
    return np.stack([np.outer(left[:,i],right[:,i]) for i in range(K)])

def make_world(seed):
    rng=np.random.default_rng(seed);atoms=make_atoms(rng)
    # 12 development task identities include each atom and six mixtures; atom extraction is dev-only.
    dev=[]
    for j in range(K):dev.append(atoms[j].copy())
    for j in range(K):
        a=(j+1)%K;dev.append(atoms[j]+.6*atoms[a])
    # The shared atom bank is estimated from six single-atom development tasks.
    learned=np.stack([dev[j] for j in range(K)])
    fresh=[]
    for i in range(12):
        ids=rng.choice(K,size=2,replace=False);co=rng.uniform(.35,1.0,size=2);m=np.zeros(K);m[ids]=co
        fresh.append(('in_span',m,m@atoms.reshape(K,-1)))
    for i in range(4):
        l=rng.normal(size=D);r=rng.normal(size=D);l/=np.linalg.norm(l);r/=np.linalg.norm(r);mat=np.outer(l,r)*rng.uniform(.5,1.0)
        fresh.append(('out_of_span',None,mat.reshape(-1)))
    return learned.reshape(K,-1),fresh

def pack(d):
    b=io.BytesIO();np.savez(b,**d);return len(b.getvalue())

def evaluate(seed):
    start=time.perf_counter();atoms,fresh=make_world(seed);n=len(fresh)
    basis=atoms.T # [64,K]
    codes=[];mir_idx=[];mir_val=[];targets=[];kinds=[]
    for kind,c,t in fresh:
        if c is None:c=np.linalg.lstsq(basis,t,rcond=None)[0]
        codes.append(c);targets.append(t);kinds.append(kind)
        ix=np.argsort(np.abs(c))[-2:][::-1];mir_idx.append(ix);mir_val.append(c[ix])
    codes=np.asarray(codes);targets=np.asarray(targets);mir_idx=np.asarray(mir_idx,dtype=np.uint8);mir_val=np.asarray(mir_val,dtype=np.float32)
    atomf=atoms.reshape(K,D,D).astype(np.float32)
    top=np.zeros_like(codes)
    for q,(ix,v) in enumerate(zip(mir_idx,mir_val)):top[q,ix]=v
    rec_top=top@basis.T
    rec_dense=codes@basis.T
    residual=targets-rec_top
    packages={
      'mirror_top2':{'shared_atoms':atomf,'code_indices':mir_idx,'code_values':mir_val},
      'mole_top2':{'shared_atoms':atomf.copy(),'code_indices':mir_idx.copy(),'code_values':mir_val.copy()},
      'dense_atom_code':{'shared_atoms':atomf,'dense_codes':codes.astype(np.float32)},
      'independent_full':{'task_deltas':targets.astype(np.float32)},
      'mirror_private_residual':{'shared_atoms':atomf,'code_indices':mir_idx,'code_values':mir_val,'private_residuals':residual.reshape(n,D,D).astype(np.float32)}
    }
    sizes={m:pack(d) for m,d in packages.items()};rows=[]
    recon={'mirror_top2':rec_top,'mole_top2':rec_top.copy(),'dense_atom_code':rec_dense,'independent_full':targets,'mirror_private_residual':rec_top+residual}
    for i,kind in enumerate(kinds):
        for method,out in recon.items():
            mse=float(np.mean((out[i]-targets[i])**2)); rel=float(np.linalg.norm(out[i]-targets[i])/(np.linalg.norm(targets[i])+1e-12))
            rows.append({'condition':kind+'_'+method,'world_or_seed':seed,'method':method,'serialized_bytes':sizes[method],'train_tokens_or_examples':12,'optimizer_updates':0,'active_compute_proxy':'top2: two rank-1 atom updates; dense: six; independent: stored full matrix; private: top2 plus residual add','wall_time_s':f'{(time.perf_counter()-start):.9f}','primary_metric':'functional_delta_matrix_mse','primary_value':f'{mse:.12g}','secondary_metric':'relative_frobenius_error;active_atoms','secondary_value':f'{rel:.12g};{2 if method in ("mirror_top2","mole_top2","mirror_private_residual") else (6 if method=="dense_atom_code" else 0)}','status_note':f'task={i};domain={kind};whole-task dev split; oracle coefficients from full functional matrix'})
    summary={'seed':seed,'fresh_in_span':12,'fresh_out_of_span':4,'mirror_bytes':sizes['mirror_top2'],'mole_bytes':sizes['mole_top2'],'dense_code_bytes':sizes['dense_atom_code'],'independent_bytes':sizes['independent_full'],'private_residual_bytes':sizes['mirror_private_residual'],'mirror_mole_identical_bytes':sizes['mirror_top2']==sizes['mole_top2'],'wall_s':time.perf_counter()-start}
    return rows,summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,default=[67401]);ap.add_argument('--out',default=str(ROOT/'RESULTS_CORE.csv'));ap.add_argument('--summary',default=str(ROOT/'source'/'summary.json'));a=ap.parse_args();rows=[];summ=[]
    for seed in a.seeds:r,s=evaluate(seed);rows+=r;summ.append(s)
    with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    Path(a.summary).write_text(json.dumps(summ,indent=2)+'\n');print(json.dumps(summ,indent=2))
if __name__=='__main__':main()
