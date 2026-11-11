"""Stable recurrent reservoir and task gain-state screen for MA-874."""
from __future__ import annotations
import json,struct,time
from pathlib import Path
from typing import Any
import numpy as np
from scipy.optimize import least_squares

K=8; R=2; NTRAIN=16; NTEST=16; SHOTS=(1,2,4,8); T=64; NTESTSEQ=8; NOISE=.02
HEADER=struct.Struct('<5sBBB')

def rng(seed:int,tag:int)->np.random.Generator: return np.random.default_rng(np.random.SeedSequence([seed,tag]))

def states(x:np.ndarray,lambdas:np.ndarray,inputs:np.ndarray)->np.ndarray:
    # x has [sequence,time], return [sequence,time,channel], zero state at every sequence boundary.
    out=np.zeros((x.shape[0],x.shape[1],K),dtype=np.float64)
    h=np.zeros((x.shape[0],K),dtype=np.float64)
    for t in range(x.shape[1]):
        h=h*lambdas[None,:]+x[:,t,None]*inputs[None,:]
        out[:,t,:]=h
    return out

def make_world(seed:int)->dict[str,Any]:
    g=rng(seed,1); lambdas=np.linspace(.15,.9,K)+g.uniform(-.008,.008,K); inputs=np.ones(K)
    basis=g.normal(size=(R,K)); q,_=np.linalg.qr(basis.T); basis=q[:,:R].T
    ztrain=g.normal(size=(NTRAIN,R)); ztest=g.normal(size=(NTEST,R)); atrain=ztrain@basis; atest=ztest@basis
    train_phi=[]; train_y=[]
    for i in range(NTRAIN):
        rr=rng(seed,100+i); x=rr.normal(size=(8,T)); ph=states(x,lambdas,inputs); y=np.einsum('ntk,k->nt',ph,atrain[i])+rr.normal(0,NOISE,size=(8,T)); train_phi.append(ph.reshape(-1,K)); train_y.append(y.ravel())
    support_phi=[]; support_y=[]; test_phi=[]; test_y=[]
    for i in range(NTEST):
        rr=rng(seed,10_000+i); xs=rr.normal(size=(max(SHOTS),T)); ps=states(xs,lambdas,inputs); ys=np.einsum('ntk,k->nt',ps,atest[i])+rr.normal(0,NOISE,size=(max(SHOTS),T))
        xt=rr.normal(size=(NTESTSEQ,T)); pt=states(xt,lambdas,inputs); yt=np.einsum('ntk,k->nt',pt,atest[i])
        support_phi.append(ps); support_y.append(ys); test_phi.append(pt); test_y.append(yt)
    return {'seed':seed,'lambdas':lambdas,'inputs':inputs,'true_basis':basis,'atrain':atrain,'atest':atest,'train_phi':train_phi,'train_y':train_y,'support_phi':support_phi,'support_y':support_y,'test_phi':test_phi,'test_y':test_y}

def fit_shared(world:dict[str,Any])->dict[str,np.ndarray]:
    coeff=np.stack([np.linalg.lstsq(p,y,rcond=None)[0] for p,y in zip(world['train_phi'],world['train_y'])])
    center=coeff.mean(axis=0); _,_,vt=np.linalg.svd(coeff-center,full_matrices=False)
    pooled=np.linalg.lstsq(np.concatenate(world['train_phi']),np.concatenate(world['train_y']),rcond=None)[0]
    return {'train_coeff':coeff,'basis':vt[:R],'pooled':pooled}

def lowrank_fit(phi:np.ndarray,y:np.ndarray,basis:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    design=phi@basis.T; code=np.linalg.lstsq(design,y,rcond=None)[0]; return code@basis,code

def payload(world:dict[str,Any],model:dict[str,np.ndarray],method:str,codes:np.ndarray|None=None)->bytes:
    ids={'mirror':1,'native_gain':2,'independent':3,'fast_weights':4,'pooled':5,'teacher':6}
    b=bytearray(HEADER.pack(b'MA874',ids[method],K,R)); b.extend(np.asarray(world['lambdas'],dtype='<f4').tobytes()); b.extend(np.asarray(world['inputs'],dtype='<f4').tobytes())
    if method in ('mirror','native_gain'):
        b.extend(np.asarray(model['basis'],dtype='<f4').tobytes()); b.extend(np.asarray(codes,dtype='<f4').tobytes())
    elif method in ('independent','teacher'): b.extend(np.asarray(codes,dtype='<f4').tobytes())
    elif method=='fast_weights': b.extend(np.asarray(codes,dtype='<f4').tobytes())
    elif method=='pooled': b.extend(np.asarray(model['pooled'],dtype='<f4').tobytes())
    return bytes(b)

def optimize_fast(world:dict[str,Any],task:int,nshots:int)->tuple[np.ndarray,float,int,float]:
    # Writable recurrent update proxy: fit a diagonal decay vector and a scalar output gain against
    # support trajectories reconstructed from h[t]=lambda*h[t-1]+x[t]. Store all nine values.
    # The task input sequences are recovered from the first-channel recurrence (input weight is 1).
    ph=world['support_phi'][task][:nshots]
    y=world['support_y'][task][:nshots]
    lam0=world['lambdas']
    # Since channels share x, x[t] = h_0[t] - lambda_0*h_0[t-1].
    x=np.zeros((nshots,T)); x[:,0]=ph[:,:1,0].ravel()
    x[:,1:]=ph[:,1:,0]-lam0[0]*ph[:,:-1,0]
    target=y
    def residual(theta:np.ndarray)->np.ndarray:
        ls=theta[:K]; gain=theta[K]
        pred=np.einsum('ntk,k->nt',states(x,ls,world['inputs']),np.ones(K))*gain
        return (pred-target).ravel()
    init=np.concatenate([lam0,[1.0]])
    t0=time.perf_counter(); fit=least_squares(residual,init,bounds=(np.r_[np.full(K,.05),-4.],np.r_[np.full(K,.99),4.]),max_nfev=30,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    elapsed=time.perf_counter()-t0
    return fit.x,elapsed,int(fit.nfev),float(np.mean(residual(fit.x)**2))

def evaluate(seed:int)->dict[str,Any]:
    started=time.perf_counter(); world=make_world(seed); t0=time.perf_counter(); model=fit_shared(world); train_s=time.perf_counter()-t0; methods=('mirror','native_gain','independent','fast_weights','pooled','teacher'); by={}
    for n in SHOTS:
        mse={m:[] for m in methods}; states_by={m:[] for m in methods}; adapt={m:0. for m in methods}; infer={m:0. for m in methods}; opt_evals=0; reset_err=0.
        for task in range(NTEST):
            phi=world['support_phi'][task][:n].reshape(-1,K); y=world['support_y'][task][:n].ravel(); test=world['test_phi'][task].reshape(-1,K); target=world['test_y'][task].ravel(); coeff_true=world['atest'][task]
            t1=time.perf_counter(); cm,code=lowrank_fit(phi,y,model['basis']); adapt['mirror']+=time.perf_counter()-t1; states_by['mirror'].append(code)
            t1=time.perf_counter(); cn,code_n=lowrank_fit(phi,y,model['basis']); adapt['native_gain']+=time.perf_counter()-t1; states_by['native_gain'].append(code_n)
            for m,c in [('mirror',cm),('native_gain',cn)]:
                t1=time.perf_counter(); pred=test@c; infer[m]+=time.perf_counter()-t1; mse[m].append(float(np.mean((pred-target)**2)))
            t1=time.perf_counter(); ci=np.linalg.lstsq(phi,y,rcond=None)[0]; adapt['independent']+=time.perf_counter()-t1; states_by['independent'].append(ci)
            t1=time.perf_counter(); pred=test@ci; infer['independent']+=time.perf_counter()-t1; mse['independent'].append(float(np.mean((pred-target)**2)))
            fast,dt,nev,_=optimize_fast(world,task,n); adapt['fast_weights']+=dt; opt_evals+=nev; states_by['fast_weights'].append(fast)
            # Evaluate optimized fast decay state on the saved test input sequences.
            x_test=np.zeros((NTESTSEQ,T)); ph_test=world['test_phi'][task]; x_test[:,0]=ph_test[:,0,0]; x_test[:,1:]=ph_test[:,1:,0]-world['lambdas'][0]*ph_test[:,:-1,0]
            t1=time.perf_counter(); pred=(np.einsum('ntk,k->nt',states(x_test,fast[:K],world['inputs']),np.ones(K))*fast[K]).ravel(); infer['fast_weights']+=time.perf_counter()-t1; mse['fast_weights'].append(float(np.mean((pred-target)**2)))
            mse['pooled'].append(float(np.mean((test@model['pooled']-target)**2))); states_by['pooled'].append(model['pooled'])
            mse['teacher'].append(0.); states_by['teacher'].append(coeff_true)
            # Explicit zero-state reset replay for identical first test sequence.
            again=states(x_test[:1],world['lambdas'],world['inputs']); reset_err=max(reset_err,float(np.max(np.abs(again[0]-ph_test[0]))))
        codes={m:np.stack(states_by[m]) for m in methods}
        sizes={m:len(payload(world,model,m,codes.get(m))) for m in methods}
        by[str(n)]={'mean_mse':{m:float(np.mean(mse[m])) for m in methods},'worst_task_mse':{m:float(max(mse[m])) for m in methods},'per_task_mse':mse,'payload_bytes':sizes,'adaptation_seconds':adapt,'inference_seconds':infer,'optimizer_evaluations':opt_evals,'reset_max_abs_error':reset_err}
    return {'seed':seed,'train_basis_seconds':train_s,'by_support':by,'reservoir_update_ops_per_timestep':3*K,'decoder_ops_proxy':{'mirror':2*R*K+2*K,'native_gain':2*R*K+2*K,'independent':2*K,'fast_weights':3*K},'max_wall_seconds':time.perf_counter()-started}

def run(seeds:list[int])->dict[str,Any]:
    rows=[evaluate(s) for s in seeds]; methods=('mirror','native_gain','independent','fast_weights','pooled','teacher'); out={'experiment_id':'MA-874','seeds':seeds,'support_sequences':list(SHOTS),'rows':rows}
    out['mean_mse_by_support']={n:{m:float(np.mean([r['by_support'][n]['mean_mse'][m] for r in rows])) for m in methods} for n in map(str,SHOTS)}
    out['max_world_mean_mse_by_support']={n:{m:max(r['by_support'][n]['mean_mse'][m] for r in rows) for m in methods} for n in map(str,SHOTS)}
    out['max_task_mse_by_support']={n:{m:max(r['by_support'][n]['worst_task_mse'][m] for r in rows) for m in methods} for n in map(str,SHOTS)}
    out['payload_bytes']={m:int(np.mean([r['by_support']['2']['payload_bytes'][m] for r in rows])) for m in methods}
    out['mean_adaptation_seconds_at_2']={m:float(np.mean([r['by_support']['2']['adaptation_seconds'][m] for r in rows])) for m in methods}
    out['max_reset_error']=max(r['by_support']['2']['reset_max_abs_error'] for r in rows); out['max_wall_seconds']=max(r['max_wall_seconds'] for r in rows); out['mean_train_basis_seconds']=float(np.mean([r['train_basis_seconds'] for r in rows]) )
    return out

def write_json(path:str|Path,data:dict[str,Any])->None: Path(path).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
