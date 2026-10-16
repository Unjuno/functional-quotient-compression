"""GIC-001: controlled shared-spline gradient-interference experiment.
No network I/O. All scientific settings are in PROTOCOL.json.
The rank-2 'Mirror' is exactly an ordinary low-rank spline dictionary.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse, hashlib, itertools, json, math, struct, time
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'PROTOCOL.json').read_text())
HEADER=struct.Struct('<8s6I2d')
MAGIC=b'GIC001\x00\x00'

def basis(x: np.ndarray, k: int=12) -> np.ndarray:
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    if k<4 or not np.isfinite(x).all() or np.any(np.abs(x)>1):
        raise ValueError('finite x in [-1,1] and k>=4 required')
    knots=np.r_[np.repeat(-1.,4),np.linspace(-1,1,k-2)[1:-1],np.repeat(1.,4)]
    return BSpline.design_matrix(x,knots,3,extrapolate=False).toarray()

def population_gram(k: int=12) -> np.ndarray:
    # Eight nodes integrate products of cubic polynomials exactly on each interval.
    v,w=np.polynomial.legendre.leggauss(8)
    edges=np.linspace(-1,1,k-2); h=np.zeros((k,k))
    for a,b in zip(edges[:-1],edges[1:]):
        phi=basis((a+b)/2+(b-a)*v/2,k)
        h+=phi.T@((w*(b-a)/4)[:,None]*phi)
    return h

def stats(b: np.ndarray,y: np.ndarray) -> dict:
    return {'H':b.T@b/len(b),'F':y.T@b/len(b),'y2':(y*y).mean(0)}

def make_world(world: int,regime: str,noise: float|None=None) -> dict:
    t,k=CFG['tasks'],CFG['basis_count']; h=population_gram(k)
    rng=np.random.default_rng(np.random.SeedSequence([world,100]))
    q=np.linalg.qr(rng.normal(size=(k,k)))[0]
    directions=np.linalg.solve(np.linalg.cholesky(h).T,q)
    common=.7*directions[:,0]
    codes=rng.normal(size=(t,2)); codes-=codes.mean(0); codes/=codes.std(0)
    c=np.tile(common,(t,1)); exceptions=[]
    if regime in ('related','partial'): c+=.6*codes@directions[:,1:3].T
    if regime=='partial':
        exceptions=sorted(rng.choice(t,2,replace=False).tolist())
        c[exceptions]+=1.3*directions[:,3:5].T
    if regime=='unrelated': c+=.35*rng.normal(size=(t,8))@directions[:,1:9].T
    if regime not in CFG['regimes']: raise ValueError(regime)
    ans={'truth':c,'population_H':h,'exceptions':exceptions}
    sd=CFG['label_noise_sd'] if noise is None else noise
    for split,n,tag in [('train',256,201),('val',128,202),('test',512,203)]:
        gen=np.random.default_rng(np.random.SeedSequence([world,tag]))
        x=gen.uniform(-1,1,n); b=basis(x,k); y=b@c.T
        if split=='train': y=y+sd*gen.normal(size=y.shape)
        ans[split]={'x':x,'B':b,'Y':y,'stats':stats(b,y)}
    return ans

class Model:
    def __init__(self,t: int,k: int,r: int,seed: int,kind: str='factor'):
        self.t,self.k,self.r,self.kind=t,k,r,kind; self.ids=[]
        rng=np.random.default_rng(seed)
        if kind=='independent': self.p={'C':np.zeros((t,k))}
        else:
            self.p={'c':np.zeros(k),'D':rng.normal(0,.1,(k,r)),
                    'M':rng.normal(0,.1,(t,r)),'P':np.zeros((0,k))}
    def coeff(self) -> np.ndarray:
        if self.kind=='independent': return self.p['C'].copy()
        c=self.p['c'][None,:]+self.p['M']@self.p['D'].T
        c=c.copy()
        if self.ids: c[self.ids]+=self.p['P']
        return c
    def allocate(self,ids: list[int]):
        if self.kind!='factor' or self.ids: raise ValueError('allocate once on factor model')
        if len(set(ids))!=len(ids) or any(i<0 or i>=self.t for i in ids): raise ValueError(ids)
        self.ids=sorted(ids); self.p['P']=np.zeros((len(ids),self.k))
    def predict(self,x: np.ndarray) -> np.ndarray:
        return basis(x,self.k)@self.coeff().T

def gradients(model: Model,st: dict) -> tuple[dict,np.ndarray,np.ndarray]:
    c=model.coeff(); gc=2*(c@st['H']-st['F'])
    losses=np.maximum(0,(c*(c@st['H']-2*st['F'])).sum(1)+st['y2'])
    if model.kind=='independent': return {'C':gc/model.t},np.zeros((model.t,0)),losses
    m,d=model.p['M'],model.p['D']
    per=np.concatenate([gc,(gc[:,:,None]*m[:,None,:]).reshape(model.t,-1)],axis=1)
    g={'c':gc.mean(0),'D':gc.T@m/model.t,'M':gc@d/model.t,'P':gc[model.ids]/model.t}
    return g,per,losses

def pcgrad(g: np.ndarray,rng: np.random.Generator) -> np.ndarray:
    p=g.copy(); n=len(g); orders=np.argsort(rng.random((n,n)),axis=1)
    for col in range(n):
        j=orders[:,col]; target=g[j]; dot=(p*target).sum(1)
        scale=np.minimum(dot,0)/np.maximum((target*target).sum(1),1e-30)
        scale[j==np.arange(n)]=0
        p-=scale[:,None]*target
    return p.mean(0)

def diagnostic(g: np.ndarray) -> dict:
    norms=np.linalg.norm(g,axis=1)
    dot=g@g.T; valid=np.outer(norms>1e-14,norms>1e-14)&np.triu(np.ones(dot.shape,dtype=bool),1)
    if not valid.any(): fraction=None
    else:
        cosine=dot/np.maximum(np.outer(norms,norms),1e-30)
        fraction=float(np.mean(cosine[valid]<-1e-8))
    cancellation=float(np.linalg.norm(g.mean(0))/norms.mean()) if norms.mean()>1e-14 else None
    return {'conflict_fraction':fraction,'cancellation':cancellation,'mean_task_grad_norm':float(norms.mean()),
            'mean_gradient_norm':float(np.linalg.norm(g.mean(0)))}

def optimal_coeff(c: np.ndarray,h: np.ndarray,rank: int,ids: list[int]) -> np.ndarray:
    """Global mean-MSE optimum for shared affine rank-r plus free selected rows."""
    keep=np.array([i for i in range(len(c)) if i not in ids]); out=c.copy()
    if not len(keep): return out
    l=np.linalg.cholesky(h); z=c[keep]@l; avg=z.mean(0,keepdims=True)
    u,s,v=np.linalg.svd(z-avg,full_matrices=False)
    zhat=avg+(u[:,:rank]*s[:rank])@v[:rank]
    out[keep]=np.linalg.solve(l.T,zhat.T).T
    return out

def population_mse(c: np.ndarray,truth: np.ndarray,h: np.ndarray) -> np.ndarray:
    e=c-truth; return np.maximum(0,np.einsum('ti,ij,tj->t',e,h,e))

def encode(model: Model) -> bytes:
    data=HEADER.pack(MAGIC,model.t,model.k,model.r,len(model.ids),int(model.kind=='independent'),3,-1.,1.)
    data+=np.asarray(model.ids,dtype='<u4').tobytes()
    for a in model.p.values(): data+=np.asarray(a,dtype='<f8').tobytes()
    return data

def save_model(model: Model,path: Path): path.write_bytes(encode(model))

def load_model(path: Path) -> Model:
    data=path.read_bytes()
    magic,t,k,r,s,kind,degree,lo,hi=HEADER.unpack(data[:HEADER.size])
    if magic!=MAGIC or degree!=3 or (lo,hi)!=(-1.,1.) or kind not in (0,1): raise ValueError('bad codec')
    model=Model(t,k,r,0,'independent' if kind else 'factor')
    off=HEADER.size; ids=np.frombuffer(data,dtype='<u4',count=s,offset=off).tolist(); off+=s*4
    if s: model.allocate(ids)
    for key,a in model.p.items():
        n=a.size; model.p[key]=np.frombuffer(data,dtype='<f8',count=n,offset=off).copy().reshape(a.shape); off+=n*8
    if off!=len(data): raise ValueError('payload length mismatch')
    return model

def train(method: str,w: dict,world: int,init: int,steps: int|None=None) -> tuple[Model,dict]:
    steps=CFG['updates'] if steps is None else steps
    independent=method=='independent'; rank=0 if method.startswith('shared') or independent else 3 if method=='rank3' else 2
    model=Model(8,12,rank,1000+init,'independent' if independent else 'factor')
    st=w['train']['stats']; train_ls=np.linalg.solve(st['H'],st['F'].T).T
    rng=np.random.default_rng(np.random.SeedSequence([world,init,500]))
    random_rng=np.random.default_rng(np.random.SeedSequence([world,init,600]))
    moments={}; trace=[]; conflict_hist=[]; loss_hist=[]; parent_hash=None
    last_diag=[]; elapsed=time.perf_counter()
    for step in range(steps):
        if method.startswith('private') and step==CFG['allocation_step']:
            parent_hash=hashlib.sha256(encode(model)).hexdigest()
            if method=='private_random': ids=random_rng.choice(8,2,replace=False).tolist()
            elif method=='private_oracle':
                options=list(itertools.combinations(range(8),2)); scores=[]
                for choice in options:
                    co=optimal_coeff(train_ls,st['H'],2,list(choice))
                    scores.append(population_mse(co,train_ls,st['H']).mean())
                ids=list(options[int(np.argmin(scores))])
            else:
                score=np.mean(loss_hist if method=='private_loss' else conflict_hist,axis=0)
                ids=np.argsort(-score,kind='stable')[:2].tolist()
            old=model.coeff(); model.allocate(ids)
            if not np.array_equal(old,model.coeff()): raise RuntimeError('split changed function')
        g,per,losses=gradients(model,st)
        if CFG['allocation_step']-CFG['allocation_window']<=step<CFG['allocation_step']:
            conflict_hist.append(np.maximum(0,-per@per.T).sum(1)); loss_hist.append(losses)
        if step in (0,100,249,250,500,1000,1499) or step>=steps-50:
            dd=diagnostic(per); dd.update(step=step,train_mean=float(losses.mean()),train_worst=float(losses.max()))
            if step>=steps-50: last_diag.append(dd)
            if step in (0,100,249,250,500,1000,1499): trace.append(dd)
        use_pc=method in ('shared_pcgrad','rank2_pcgrad') or (method=='private_conflict_pcgrad' and step>=250)
        if use_pc:
            merged=pcgrad(per,rng); g['c']=merged[:model.k]; g['D']=merged[model.k:].reshape(model.k,model.r)
        rate=CFG['learning_rate']
        if step>=1000: rate*=.01+.99*(1+math.cos(math.pi*(step-1000)/499))/2
        for key,param in model.p.items():
            if not param.size: continue
            if key not in moments: moments[key]=[np.zeros_like(param),np.zeros_like(param),0]
            a,b,n=moments[key]; n+=1; moments[key][2]=n
            a*=.9; a+=.1*g[key]; b*=.999; b+=.001*g[key]**2
            param-=rate*(a/(1-.9**n))/(np.sqrt(b/(1-.999**n))+1e-8)
        if step%250==0 and not all(np.isfinite(p).all() for p in model.p.values()): raise FloatingPointError(method)
    fit_seconds=time.perf_counter()-elapsed
    coef=model.coeff(); train_losses=((w['train']['B']@coef.T-w['train']['Y'])**2).mean(0)
    test_losses=((w['test']['B']@coef.T-w['test']['Y'])**2).mean(0)
    pop=population_mse(coef,w['truth'],w['population_H'])
    empirical_oracle=train_ls if independent else optimal_coeff(train_ls,st['H'],rank,model.ids)
    oracle_train=((w['train']['B']@empirical_oracle.T-w['train']['Y'])**2).mean()
    pop_oracle=w['truth'] if independent else optimal_coeff(w['truth'],w['population_H'],rank,model.ids)
    floor=population_mse(pop_oracle,w['truth'],w['population_H'])
    def tail(field):
        values=[x[field] for x in last_diag if x[field] is not None]
        return float(np.mean(values)) if values else None
    row={'method':method,'world':world,'init':init,'rank':rank,'selected_ids':model.ids,
         'train_mean':float(train_losses.mean()),'test_mean':float(test_losses.mean()),'test_worst':float(test_losses.max()),
         'population_mean':float(pop.mean()),'population_worst':float(pop.max()),'population_per_task':pop.tolist(),
         'oracle_train_mean':float(oracle_train),'train_optimum_gap':float(train_losses.mean()-oracle_train),
         'population_floor':float(floor.mean()),'population_floor_worst_at_mean_optimum':float(floor.max()),
         'gradient_conflict_tail':tail('conflict_fraction'),'cancellation_tail':tail('cancellation'),
         'individual_gradient_norm_tail':tail('mean_task_grad_norm'),'aggregate_gradient_norm_tail':tail('mean_gradient_norm'),
         'payload_bytes':len(encode(model)),'learned_scalars':sum(p.size for p in model.p.values()),
         'train_seconds':fit_seconds,'trace':trace,'preallocation_parent_sha256':parent_hash,
         'allocation_conflict_scores':np.mean(conflict_hist,axis=0).tolist() if conflict_hist else [],
         'allocation_loss_scores':np.mean(loss_hist,axis=0).tolist() if loss_hist else []}
    return model,row

def quadratic_checks() -> dict:
    hs=np.array([[[1.,2.],[2.,5.]],[[1.,-2.],[-2.,5.]]]); results={}
    for method in ('mean','pcgrad','pcgrad_norm_matched'):
        theta=np.array([1.,0.]); rng=np.random.default_rng(10)
        for _ in range(100):
            g=np.einsum('tij,j->ti',hs,theta); raw=g.mean(0)
            d=raw if method=='mean' else pcgrad(g,rng)
            if method=='pcgrad_norm_matched': d*=np.linalg.norm(raw)/max(np.linalg.norm(d),1e-30)
            theta-=.1*d
        results[method]={'theta':theta.tolist(),'mean_loss':float(.5*np.einsum('i,tij,j->t',theta,hs,theta).mean())}
    results['initial_conflict_cosine']=-.6
    results['incompatible_shared_at_zero']={'mean_loss':.5,'mean_gradient':0.,'pcgrad':float(pcgrad(np.array([[1.],[-1.]]),np.random.default_rng(1))[0])}
    # One ordinary signed task coefficient is sufficient; the 'Mirror' is an alias here.
    a=0.
    for _ in range(100): a-=.1*(a-1.)
    results['incompatible_with_signed_task_code']={'coefficient':a,'mean_loss':float(.5*(a-1)**2)}
    return results

def run_case(world: int,init: int,out: Path):
    out.mkdir(parents=True,exist_ok=True)
    for regime in CFG['regimes']:
        w=make_world(world,regime)
        for method in CFG['methods']:
            stem=f'{regime}_{world}_{init}_{method}'; metadata=out/(stem+'.json')
            if metadata.exists(): raise FileExistsError(metadata)
            model,row=train(method,w,world,init); row['regime']=regime
            payload=out/(stem+'.bin'); save_model(model,payload)
            row['payload_sha256']=hashlib.sha256(payload.read_bytes()).hexdigest()
            row['payload_name']=payload.name
            metadata.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'world':world,'init':init,'regime':regime,'finished_methods':len(CFG['methods'])}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--world',type=int); p.add_argument('--init',type=int,default=0)
    p.add_argument('--out',type=Path,default=ROOT/'evidence'/'development'); p.add_argument('--quadratic',action='store_true')
    args=p.parse_args()
    if args.quadratic: print(json.dumps(quadratic_checks(),indent=2))
    else:
        if args.world is None: p.error('--world required')
        run_case(args.world,args.init,args.out)
