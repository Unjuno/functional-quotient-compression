from __future__ import annotations
import hashlib, io, time
from pathlib import Path
import numpy as np

D, O, T, R = 64, 32, 8, 4
RIDGE = 1e-3

def _solve(G, C):
    return np.linalg.solve(G + RIDGE*np.eye(G.shape[0]), C)

def _basis_solution(B, G, C):
    H=np.einsum('rdo,df,sfo->rs',B,G,B,optimize=True)
    rhs=np.einsum('rdo,do->r',B,C)
    a=np.linalg.solve(H+RIDGE*np.eye(B.shape[0]),rhs)
    return np.einsum('r,rdo->do',a,B)

def _objective(G,C,W):
    return float(np.sum(W*(G@W))-2*np.sum(C*W))

def _pack(arrays):
    bio=io.BytesIO(); np.savez_compressed(bio,**arrays); data=bio.getvalue()
    return len(data),hashlib.sha256(data).hexdigest()

def run(seed=71501):
    rng=np.random.default_rng(seed)
    raw,_=np.linalg.qr(rng.normal(size=(D*O,R)))
    B=raw.T.reshape(R,D,O)
    coeff=rng.normal(size=(T,R))
    private=np.zeros((T,D,O))
    for t in range(T):
        v=rng.normal(size=D*O); flatB=B.reshape(R,-1)
        v=v-flatB.T@(flatB@v)
        private[t]=0.1*v.reshape(D,O)/np.sqrt(D*O)
    deltas=np.einsum('tr,rdo->tdo',coeff,B)+private
    grams=[]; crosses=[]
    for t in range(T):
        X=rng.normal(size=(D+12,D)); Y=X@deltas[t]+0.03*rng.normal(size=(D+12,O))
        grams.append(X.T@X); crosses.append(X.T@Y)
    grams=np.asarray(grams); crosses=np.asarray(crosses)
    requests=[np.arange(k) for k in (2,4,6)]*4
    rows=[]
    full_outputs=[]; basis_outputs=[]; task_outputs=[]
    for ids in requests:
        G=grams[ids].sum(axis=0); C=crosses[ids].sum(axis=0)
        full_outputs.append(_solve(G,C)); basis_outputs.append(_basis_solution(B,G,C)); task_outputs.append(deltas[ids].mean(axis=0))
    full_outputs=np.stack(full_outputs); basis_outputs=np.stack(basis_outputs); task_outputs=np.stack(task_outputs)
    # Report distortion only on nonconstant part of the quadratic; this is a nonnegative excess objective.
    full_excess=[]; basis_excess=[]; task_excess=[]
    for ids,Wf,Wb,Wt in zip(requests,full_outputs,basis_outputs,task_outputs):
        G=grams[ids].sum(0); C=crosses[ids].sum(0)
        full_excess.append(max(0.0,_objective(G,C,Wf)-_objective(G,C,Wf)))
        basis_excess.append(max(0.0,_objective(G,C,Wb)-_objective(G,C,Wf)))
        task_excess.append(max(0.0,_objective(G,C,Wt)-_objective(G,C,Wf)))
    shared={'grams':grams,'crosses':crosses,'request_ids':np.array([np.pad(x,(0,6-len(x)),constant_values=-1) for x in requests],dtype=np.int16)}
    arrays_by_method={
      'full_RegMean': {**shared,'source_deltas':deltas,'outputs':full_outputs},
      'Mirror_basis': {**shared,'basis':B,'source_codes':coeff,'private_residuals':private,'outputs':basis_outputs},
      'native_projected_RegMean': {**shared,'basis':B,'source_codes':coeff,'private_residuals':private,'outputs':basis_outputs},
      'task_arithmetic': {'source_deltas':deltas,'request_ids':np.array([np.pad(x,(0,6-len(x)),constant_values=-1) for x in requests],dtype=np.int16),'outputs':task_outputs}}
    for method,arrays in arrays_by_method.items():
        n,digest=_pack(arrays)
        excess={'full_RegMean':full_excess,'Mirror_basis':basis_excess,'native_projected_RegMean':basis_excess,'task_arithmetic':task_excess}[method]
        for j,ids in enumerate(requests):
            rows.append({'seed':seed,'request':j,'tasks':len(ids),'method':method,'serialized_bytes':n,'payload_sha256':digest,'normalized_excess_objective':excess[j], 'nrmse_to_full_RegMean':float(np.linalg.norm(arrays['outputs'][j]-full_outputs[j])/(np.linalg.norm(full_outputs[j])+1e-12)), 'examples':int(len(ids)*(D+12)),'optimizer_updates':0,'active_solve_work_proxy':int(D**3+O*D*D if method=='full_RegMean' else R**3+R*D*O)})
    return rows

def save(rows, outdir):
    import csv
    p=Path(outdir); p.mkdir(parents=True,exist_ok=True)
    with (p/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=71501); ap.add_argument('--outdir',default='.'); a=ap.parse_args()
    start=time.perf_counter(); rows=run(a.seed); elapsed=time.perf_counter()-start
    for row in rows: row['solve_wall_s']=elapsed
    save(rows,a.outdir)
