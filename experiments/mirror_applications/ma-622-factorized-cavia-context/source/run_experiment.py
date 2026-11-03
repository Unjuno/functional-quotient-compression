import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

D=Path(__file__).resolve().parents[1]

def givens(seed, factor, dim=8):
    rng=np.random.default_rng(seed*100+factor)
    angles=rng.uniform(-.8,.8,4)
    q=np.eye(dim)
    for k,ang in enumerate(angles):
        a,b=2*k,2*k+1
        c,s=np.cos(ang),np.sin(ang); g=np.eye(dim); g[a,a]=g[b,b]=c; g[a,b]=-s; g[b,a]=s; q=q@g
    return q,angles

def rot_from_angles(a):
    q=np.eye(8)
    for k,ang in enumerate(a):
        i,j=2*k,2*k+1; c,s=np.cos(ang),np.sin(ang); g=np.eye(8); g[i,i]=g[j,j]=c; g[i,j]=-s; g[j,i]=s; q=q@g
    return q
def rmse(y,z): return float(np.sqrt(np.mean((y-z)**2))/max(np.sqrt(np.mean(y*y)),1e-12))
def save_payload(name, arrays, meta):
    p=D/'source'/f'{name}.npz'; np.savez_compressed(p,**arrays)
    return p.stat().st_size

def run(seed):
    rng=np.random.default_rng(seed); inp=16; cd=8
    w=rng.normal(0,.3,(inp,)); basis=rng.normal(0,.25,(cd,inp)); c0=rng.normal(0,.5,(cd,))
    rp=[givens(seed,i) for i in range(4)]; sp=[givens(seed+71,i) for i in range(4)]
    rs=[z[0] for z in rp]; ss=[z[0] for z in sp]; angle_codes=np.stack([z[1] for z in rp+sp])
    held={(i,i) for i in range(4)}; train=[(i,j) for i in range(4) for j in range(4) if (i,j) not in held]
    contexts={(i,j):rs[i]@ss[j]@c0 for i in range(4) for j in range(4)}
    records=[]; rows={}
    base={'w':w,'basis':basis}
    base_bytes=save_payload('base',base,{})
    for world in ['aligned','private_interaction']:
        residual={(i,j):rng.normal(0,.25*np.linalg.norm(c0)/np.sqrt(cd),(cd,)) for i in range(4) for j in range(4)}
        target={k:contexts[k]+(residual[k] if world=='private_interaction' else 0) for k in contexts}
        # fit fixed shared predictor on generated context target; audit uses disjoint inputs
        audits={}; supports={}
        for k in contexts:
            x=rng.normal(size=(128,inp)); xs=rng.normal(size=(16,inp)); audits[k]=x; supports[k]=xs
        def pred(x,c): return x@w + (x@basis.T)@c
        # Estimate visible task contexts only from support examples.
        observed={}
        for i,j in train:
            x=supports[i,j]; y=x@w+(x@basis.T)@target[i,j]
            observed[i,j]=np.linalg.lstsq(x@basis.T,y-x@w,rcond=None)[0]
        # Fit shared seed and factor angles on the 12 visible combinations.
        def decode(z,i,j): return rot_from_angles(z[8+4*i:12+4*i]) @ rot_from_angles(z[24+4*j:28+4*j]) @ z[:8]
        # compact 8 + 16 + 16 parameter vector
        def dec(z,i,j): return rot_from_angles(z[8+4*i:12+4*i]) @ rot_from_angles(z[24+4*j:28+4*j]) @ z[:8]
        z0=np.zeros(40); z0[:8]=np.mean(list(observed.values()),axis=0)
        fit=least_squares(lambda z: np.concatenate([dec(z,i,j)-observed[i,j] for i,j in train]),z0,max_nfev=250)
        fitted_codes=fit.x
        # zero context; held-out tasks
        for method in ['zero_context','independent_cavia','mirror_rotation','native_direct_angle','native_additive']:
            if method=='mirror_rotation' or method=='native_direct_angle':
                predctx={(i,j):dec(fitted_codes,i,j) for i,j in contexts}
                arrays={'w':w,'basis':basis,'c0':fitted_codes[:8],'factor_angles':np.concatenate([fitted_codes[8:24],fitted_codes[24:40]])}
                task_bytes=arrays['c0'].nbytes+arrays['factor_angles'].nbytes
            elif method=='native_additive':
                # additive least squares fit to visible generated contexts; represent u_i + v_j
                A=[]; Y=[]
                for i,j in train:
                    row=np.zeros((cd,8*2)); row[:,:cd]=np.eye(cd); row[:,cd:]=np.eye(cd)
                    A.append(row); Y.append(target[i,j])
                # explicit factor context approximation via per-factor means
                observed={}
                for i,j in train:
                    x=supports[i,j]; y=x@w+(x@basis.T)@target[i,j]
                    observed[i,j]=np.linalg.lstsq(x@basis.T,y-x@w,rcond=None)[0]
                u=np.stack([np.mean([observed[i,j] for i,j in train if i==ii],axis=0) for ii in range(4)])
                v=np.stack([np.mean([observed[i,j] for i,j in train if j==jj],axis=0) for jj in range(4)])
                predctx={(i,j):u[i]+v[j]-np.mean(u,axis=0) for i,j in contexts}
                arrays={'w':w,'basis':basis,'u':u,'v':v}; task_bytes=u.nbytes+v.nbytes
            elif method=='independent_cavia':
                predctx={}
                for k in contexts:
                    x=supports[k]; y=x@w+(x@basis.T)@target[k]
                    predctx[k]=np.linalg.lstsq(x@basis.T,y-x@w,rcond=None)[0]
                arrays={'w':w,'basis':basis,'contexts':np.stack([predctx[k] for k in contexts])}; task_bytes=arrays['contexts'].nbytes
            else:
                predctx={k:np.zeros(cd) for k in contexts}; arrays={'w':w,'basis':basis}; task_bytes=0
            arrays['factor_pairs']=np.array([(i,j) for i in range(4) for j in range(4)],dtype=np.int8)
            path=D/'source'/f'{world}_{seed}_{method}.npz'; np.savez_compressed(path,**arrays); nbytes=path.stat().st_size
            vals=[]
            for k in held:
                x=audits[k]; vals.append(rmse(pred(x,target[k]),pred(x,predctx[k])))
            records.append({'world':world,'seed':seed,'method':method,'serialized_bytes':nbytes,'task_state_bytes':int(task_bytes),'heldout_nrmse':float(np.mean(vals)),'support_examples':16,'context_solve_evaluations':int(fit.nfev),'active_compute_proxy':int(4*128*inp*cd),'wall_time_s':0.0})
    return records

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',nargs='+',type=int,required=True); args=ap.parse_args()
    t=time.perf_counter(); out=[]
    for s in args.seeds: out.extend(run(s))
    for r in out: r['wall_time_s']=time.perf_counter()-t
    (D/'source'/'development.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
