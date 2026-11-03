import argparse,csv,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
R=Path(__file__).resolve().parents[1]; PAIRS=[(i,j) for i in range(4) for j in range(4)]; HOLD={(i,i) for i in range(4)}; TRAIN=[p for p in PAIRS if p not in HOLD]
def rot(a):
 q=np.eye(8)
 for k,x in enumerate(a):
  i,j=2*k,2*k+1;c,s=np.cos(x),np.sin(x);g=np.eye(8);g[i,i]=g[j,j]=c;g[i,j]=-s;g[j,i]=s;q=q@g
 return q
def gen(seed,k):
 r=np.random.default_rng(seed*31+k); a=r.uniform(-.7,.7,4); return rot(a),a
def fit_ctx(x,y,w,b): return np.linalg.lstsq(x@b.T,y-x@w,rcond=None)[0]
def nrmse(y,z): return float(np.sqrt(np.mean((y-z)**2))/max(np.sqrt(np.mean(y*y)),1e-12))
def run(seed):
 rng=np.random.default_rng(seed); w=rng.normal(0,.3,16); b=rng.normal(0,.25,(8,16)); c0=rng.normal(0,.5,8)
 A=[gen(seed,i) for i in range(4)]; B=[gen(seed+53,i) for i in range(4)]; target={p:A[p[0]][0]@B[p[1]][0]@c0 for p in PAIRS}
 sup={}; aud={}; obs={}
 for p in PAIRS:
  x=rng.normal(size=(16,16)); z=rng.normal(size=(128,16)); sup[p]=x; aud[p]=z; obs[p]=fit_ctx(x,x@w+(x@b.T)@target[p],w,b)
 def dec(z,i,j):return rot(z[8+4*i:12+4*i])@rot(z[24+4*j:28+4*j])@z[:8]
 z0=np.zeros(40);z0[:8]=np.mean([obs[p] for p in TRAIN],axis=0)
 fit=least_squares(lambda z:np.concatenate([dec(z,*p)-obs[p] for p in TRAIN]),z0,max_nfev=250)
 codebook=np.stack([dec(fit.x,*p) for p in PAIRS]); out=[]
 common={'w':w,'context_basis':b}
 payloads={
 'direct_cavia':dict(common,solver_schema=np.array([1],dtype=np.uint8)),
 'generic_pooled_context':dict(common,aggregation_schema=np.array([1],dtype=np.uint8)),
 'mirror_inferred_code':dict(common,factor_angles=np.r_[fit.x[8:24],fit.x[24:40]],shared_seed=fit.x[:8],factor_indices=np.array(PAIRS,dtype=np.int8)),
 'independent_codebook_upper':dict(common,task_contexts=codebook,factor_indices=np.array(PAIRS,dtype=np.int8))}
 for name,arr in payloads.items(): np.savez_compressed(R/'source'/f'{seed}_{name}.npz',**arr)
 candidates=codebook
 for p in HOLD:
  x=sup[p]; y=x@w+(x@b.T)@target[p]; z=aud[p]; ya=z@w+(z@b.T)@target[p]
  # generic pooled sufficient statistics equal the ordinary linear least-squares context estimator
  c_direct=fit_ctx(x,y,w,b)
  errs={}
  errs['direct_cavia']=nrmse(ya,z@w+(z@b.T)@c_direct)
  errs['generic_pooled_context']=errs['direct_cavia']
  losses=[]; evals=0
  for c in candidates:
   pred=x@w+(x@b.T)@c; losses.append(np.mean((y-pred)**2)); evals+=len(x)
  cm=candidates[int(np.argmin(losses))]
  errs['mirror_inferred_code']=nrmse(ya,z@w+(z@b.T)@cm)
  errs['independent_codebook_upper']=errs['mirror_inferred_code']
  for method,err in errs.items():
   f=R/'source'/f'{seed}_{method}.npz'; out.append({'seed':seed,'task':list(p),'method':method,'serialized_bytes':f.stat().st_size,'heldout_nrmse':err,'support_examples':16,'codebook_candidate_evaluations':evals if method=='mirror_inferred_code' else 0,'active_compute_proxy':evals*16*8 if method=='mirror_inferred_code' else 16*8*8,'optimizer_updates':0})
 return out
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);a=ap.parse_args();t=time.perf_counter();rows=[]
 for s in a.seeds:rows+=run(s)
 elapsed=time.perf_counter()-t
 for z in rows:z['wall_time_s']=elapsed
 (R/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (R/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
  for z in rows:w.writerow(['aligned',z['seed'],z['method'],z['serialized_bytes'],z['support_examples'],0,z['active_compute_proxy'],z['wall_time_s'],'heldout_nrmse',z['heldout_nrmse'],'candidate_evaluations',z['codebook_candidate_evaluations'],'development; fresh sealed'])
 print(json.dumps(rows,indent=2))
