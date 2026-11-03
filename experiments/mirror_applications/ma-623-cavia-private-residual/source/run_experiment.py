import argparse, csv, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
PAIRS=[(i,j) for i in range(4) for j in range(4)]
HELD={(i,i) for i in range(4)}
TRAIN=[p for p in PAIRS if p not in HELD]

def givens(seed, factor):
    rng=np.random.default_rng(seed*101+factor); angles=rng.uniform(-.7,.7,4); q=np.eye(8)
    for k,a in enumerate(angles):
        i,j=2*k,2*k+1; c,s=np.cos(a),np.sin(a); g=np.eye(8); g[i,i]=g[j,j]=c; g[i,j]=-s; g[j,i]=s; q=q@g
    return q,angles

def rot(a):
    q=np.eye(8)
    for k,x in enumerate(a):
        i,j=2*k,2*k+1; c,s=np.cos(x),np.sin(x); g=np.eye(8); g[i,i]=g[j,j]=c; g[i,j]=-s; g[j,i]=s; q=q@g
    return q

def nrmse(y,z): return float(np.sqrt(np.mean((y-z)**2))/max(np.sqrt(np.mean(y*y)),1e-12))
def estimate_context(x,y,w,b): return np.linalg.lstsq(x@b.T,y-x@w,rcond=None)[0]

def run(seed):
    rng=np.random.default_rng(seed); w=rng.normal(0,.3,16); b=rng.normal(0,.25,(8,16)); c0=rng.normal(0,.5,8)
    rs=[givens(seed,i) for i in range(4)]; ss=[givens(seed+79,i) for i in range(4)]
    base={p:rs[p[0]][0]@ss[p[1]][0]@c0 for p in PAIRS}
    # Private interaction lies in a two-dimensional task-private subspace.
    q,_=np.linalg.qr(rng.normal(size=(8,2))); coeff={p:rng.normal(0,.25,2) for p in PAIRS}
    worlds={'aligned':{p:base[p].copy() for p in PAIRS},'private_interaction':{p:base[p]+q@coeff[p] for p in PAIRS}}
    out=[]
    for world,targets in worlds.items():
        support={}; audit={}; observed={}
        for p in PAIRS:
            xs=rng.normal(size=(16,16)); xa=rng.normal(size=(128,16)); support[p]=xs; audit[p]=xa
            observed[p]=estimate_context(xs,xs@w+(xs@b.T)@targets[p],w,b)
        # Fit shared rotation coordinates on training task supports only.
        def dec(z,i,j): return rot(z[8+4*i:12+4*i])@rot(z[24+4*j:28+4*j])@z[:8]
        z0=np.zeros(40); z0[:8]=np.mean([observed[p] for p in TRAIN],axis=0)
        fit=least_squares(lambda z:np.concatenate([dec(z,*p)-observed[p] for p in TRAIN]),z0,max_nfev=250)
        fitted={p:dec(fit.x,*p) for p in PAIRS}
        residual={p:observed[p]-fitted[p] for p in TRAIN}
        mat=np.stack([residual[p] for p in TRAIN]); _,_,vh=np.linalg.svd(mat,full_matrices=False); pca=vh.T
        for rank in [0,1,2,4,8]:
            for method in ['mirror_pca_residual','native_pca_residual','direct_residual_coordinates']:
                if method=='direct_residual_coordinates': basis=np.eye(8)[:,:rank]
                else: basis=pca[:,:rank]
                codes={p:(basis.T@(observed[p]-fitted[p]) if p in HELD else basis.T@(observed[p]-fitted[p])) for p in PAIRS}
                pred={p:fitted[p]+basis@codes[p] for p in PAIRS}
                arrays={'w':w,'context_basis':b,'shared_seed':fit.x[:8],
                        'factor_angles':np.r_[fit.x[8:24],fit.x[24:40]],
                        'residual_basis':basis,'residual_codes':np.stack([codes[p] for p in PAIRS]),
                        'factor_pairs':np.array(PAIRS,dtype=np.int8)}
                if rank==0: arrays.pop('residual_basis'); arrays.pop('residual_codes')
                fn=ROOT/'source'/f'{world}_{seed}_r{rank}_{method}.npz'; np.savez_compressed(fn,**arrays)
                task_state=(arrays['shared_seed'].nbytes+arrays['factor_angles'].nbytes+arrays['factor_pairs'].nbytes)
                if rank: task_state+=arrays['residual_basis'].nbytes+arrays['residual_codes'].nbytes
                errs=[]
                for p in HELD:
                    x=audit[p]; y=x@w+(x@b.T)@targets[p]; yh=x@w+(x@b.T)@pred[p]; errs.append(nrmse(y,yh))
                # Independent full contexts adapted from each task support, including held-out supports.
                if method=='mirror_pca_residual':
                    independent={'w':w,'context_basis':b,'task_contexts':np.stack([observed[p] for p in PAIRS]),'factor_pairs':np.array(PAIRS,dtype=np.int8)}
                    ind_path=ROOT/'source'/f'{world}_{seed}_independent.npz'; np.savez_compressed(ind_path,**independent)
                    ind_bytes=ind_path.stat().st_size
                    ind_state=independent['task_contexts'].nbytes+independent['factor_pairs'].nbytes
                    for p in HELD:
                        x=audit[p]; y=x@w+(x@b.T)@targets[p]; yh=x@w+(x@b.T)@observed[p];
                        # one record per private rank compares against this same independent control
                        ind_err=nrmse(y,yh); break
                out.append({'world':world,'seed':seed,'rank':rank,'method':method,'serialized_bytes':fn.stat().st_size,
                            'task_state_bytes':int(task_state),'independent_payload_bytes':int(ind_bytes),'independent_state_bytes':int(ind_state),
                            'heldout_nrmse':float(np.mean(errs)),'independent_nrmse':float(ind_err),
                            'support_examples':16,'factor_fit_evaluations':int(fit.nfev),'optimizer_updates':0,
                            'active_compute_proxy':int(4*128*16*8),'wall_time_s':0.0})
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',nargs='+',type=int,required=True); a=ap.parse_args(); t=time.perf_counter(); rows=[]
    for seed in a.seeds: rows.extend(run(seed))
    elapsed=time.perf_counter()-t
    for x in rows: x['wall_time_s']=elapsed
    (ROOT/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
        for x in rows: w.writerow([x['world'],x['seed'],f"{x['method']}_r{x['rank']}",x['serialized_bytes'],x['support_examples'],0,x['active_compute_proxy'],x['wall_time_s'],'heldout_nrmse',x['heldout_nrmse'],'task_state_bytes',x['task_state_bytes'],'development; fresh sealed'])
    print(json.dumps(rows,indent=2))
