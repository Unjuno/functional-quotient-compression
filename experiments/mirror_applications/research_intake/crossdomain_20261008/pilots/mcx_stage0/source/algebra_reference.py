"""MCX preregistered finite-dimensional RC, BatchEnsemble and PDE composition audit.
Native TabM, Caduceus, DISCO models are NOT implemented by this algebraic preflight.
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python algebra_reference.py
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm

DEV = [11, 12, 13]
FRESH = [101, 102, 103, 104, 105]
DT = [0.05, 0.1, 0.2]
FIELDS = ["stage","seed","dt_s","rc_involution_max_abs","rc_parity_max_abs",
"rc_view_recovery_max_abs","generic_rc_parity_max_abs","base_collision_max_abs",
"input_gate_collision_l2","post_gate_exact_max_abs","commutator_norm_fro",
"lie_rel_fro","corrected_lie_rel_fro","strang_rel_fro","commuting_lie_max_abs",
"lie_vs_strang_ratio","lie_vs_corrected_ratio"]

def reverse_complement(length=16):
    complement = (3,2,1,0)
    perm = [(length-1-p)*4+complement[b] for p in range(length) for b in range(4)]
    return np.eye(length*4)[:,perm]

def strand_audit(seed):
    rng = np.random.default_rng(seed*10+1)
    j = reverse_complement()
    r = rng.normal(size=(64,12))/8.0
    we=(r+j@r)/2.0
    wo=(r-j@r)/2.0
    w=np.column_stack((we[:,:6],wo[:,:6]))
    p=np.diag([1.0]*6+[-1.0]*6)
    tok=rng.integers(0,4,size=(128,16))
    x=np.eye(4)[tok].reshape(128,64)
    h=x@w
    hr=(x@j)@w
    c=rng.normal(size=(12,3))
    g=rng.normal(size=(64,12))/8.0
    return (float(np.max(np.abs(j@j-np.eye(64)))),
            float(np.max(np.abs(hr-h@p))),
            float(np.max(np.abs(hr@c-h@p@c))),
            float(np.max(np.abs(x@j@g-x@g@p))))

def member_audit(seed):
    rng=np.random.default_rng(seed*10+2)
    w=rng.normal(size=(6,4))/np.sqrt(6)
    _,_,vh=np.linalg.svd(w.T,full_matrices=True)
    n=vh[4]; n=n/np.linalg.norm(n)
    x=rng.normal(size=6)
    x2=x+n
    r=rng.uniform(.4,1.6,size=6)
    s=rng.uniform(.4,1.6,size=4)
    y0=x@w; y1=x2@w
    return (float(np.max(np.abs(y0-y1))),
            float(np.linalg.norm((x2*r)@w-(x*r)@w)),
            float(np.max(np.abs(x@(w*s[None,:])-(x@w)*s))))

def pde_audit(seed,dt):
    rng=np.random.default_rng(seed*10+3)
    eye=np.eye(16)
    a=.5*(np.roll(eye,-1,axis=1)-np.roll(eye,1,axis=1))
    b=np.diag(-.25-rng.uniform(0.,1.,size=16))
    comm=a@b-b@a
    exact=expm(dt*(a+b))
    lie=expm(dt*a)@expm(dt*b)
    corrected=lie@expm(-.5*dt*dt*comm)
    strang=expm(.5*dt*a)@expm(dt*b)@expm(.5*dt*a)
    identity_b=-.5*eye
    commuting=expm(dt*a)@expm(dt*identity_b)
    commuting_exact=expm(dt*(a+identity_b))
    norm=np.linalg.norm(exact,"fro")
    l=float(np.linalg.norm(lie-exact,"fro")/norm)
    c=float(np.linalg.norm(corrected-exact,"fro")/norm)
    s=float(np.linalg.norm(strang-exact,"fro")/norm)
    return (float(np.linalg.norm(comm,"fro")),l,c,s,
            float(np.max(np.abs(commuting-commuting_exact))),l/s,l/c)

def rows_for(stage,seeds):
    rows=[]
    for seed in seeds:
        strand=strand_audit(seed)
        member=member_audit(seed)
        for dt in DT:
            row=[stage,seed,dt,*strand,*member,*pde_audit(seed,dt)]
            assert len(row)==len(FIELDS)
            vals=dict(zip(FIELDS,row))
            assert max(vals[k] for k in ["rc_involution_max_abs","rc_parity_max_abs",
                "rc_view_recovery_max_abs","base_collision_max_abs","post_gate_exact_max_abs",
                "commuting_lie_max_abs"])<1e-11
            assert vals["input_gate_collision_l2"]>1e-4
            assert vals["generic_rc_parity_max_abs"]>1e-3
            assert vals["corrected_lie_rel_fro"]<vals["lie_rel_fro"]
            assert vals["strang_rel_fro"]<vals["lie_rel_fro"]
            rows.append(vals)
    return rows

def main():
    out=Path(__file__).resolve().parent.parent/"results"
    out.mkdir(parents=True,exist_ok=True)
    for stage,seeds in [("dev",DEV),("fresh",FRESH)]:
        rows=rows_for(stage,seeds)
        dest=out/(stage+"_recomputed.csv")
        with dest.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=FIELDS)
            w.writeheader()
            for row in rows:
                w.writerow({k:format(v,".14g") if isinstance(v,float) else v for k,v in row.items()})
        subset=[r for r in rows if r["dt_s"]==.1]
        print(json.dumps(dict(stage=stage,worlds=len(seeds),rows=len(rows),
          parity_max=max(r["rc_parity_max_abs"] for r in rows),
          input_gate_collision_min=min(r["input_gate_collision_l2"] for r in rows),
          lie_dt01_median=float(np.median([r["lie_rel_fro"] for r in subset])),
          corrected_dt01_median=float(np.median([r["corrected_lie_rel_fro"] for r in subset])),
          strang_dt01_median=float(np.median([r["strang_rel_fro"] for r in subset])),
          original_hypothesis="mechanism only; no trained model or Mirror gain"),indent=2))
if __name__=="__main__":
    main()
