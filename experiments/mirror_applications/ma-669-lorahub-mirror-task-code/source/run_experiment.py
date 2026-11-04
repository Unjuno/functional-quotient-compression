import argparse,csv,json,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
D=8;J=8;R=2;META=128;QUERIES=128
TASKS=(16,64);SHOTS=(2,4,8);RHOS=(0.,.1,.25)
SCHEMA=np.frombuffer(b"LORAHUB-MIRROR-TASK-CODE-V1",dtype=np.uint8)

def make_world(seed,n,shots,rho):
    rng=np.random.default_rng(seed*100000+n*100+shots)
    left=rng.normal(size=(J,D)).astype(np.float32)/np.sqrt(D)
    right=rng.normal(size=(J,D)).astype(np.float32)/np.sqrt(D)
    atoms=np.einsum('jo,ji->joi',left,right)
    qtrue,_=np.linalg.qr(rng.normal(size=(J,R)))
    meta_z=rng.normal(size=(META,R));meta=qtrue@meta_z.T
    eps=rng.normal(size=(J,META));eps-=qtrue@(qtrue.T@eps)
    eps*=rho/max(float(np.sqrt(np.mean(eps**2))),1e-12)
    qhat=np.linalg.svd(meta+eps,full_matrices=False)[0][:,:R].astype(np.float32)
    z=rng.normal(size=(n,R)).astype(np.float32)*.7
    coeff=z@qtrue.T
    er=rng.normal(size=(n,J)).astype(np.float32);er-= (er@qtrue)@qtrue.T
    er*=rho*np.sqrt(np.mean(coeff**2,axis=1,keepdims=True))/np.maximum(np.sqrt(np.mean(er**2,axis=1,keepdims=True)),1e-12)
    coeff=(coeff+er).astype(np.float32)
    base=(rng.normal(size=(D,D))/np.sqrt(D)).astype(np.float32)*.1
    weights=base[None,:,:]+np.einsum('nj,joi->noi',coeff,atoms)
    xs=rng.normal(size=(n,shots,D)).astype(np.float32)
    xq=rng.normal(size=(n,QUERIES,D)).astype(np.float32)
    ys=np.einsum('nsi,noi->nso',xs,weights);yq=np.einsum('nsi,noi->nso',xq,weights)
    return base,left,right,atoms,qhat,coeff,weights,xs,xq,ys,yq

def features(x,atoms):return np.einsum('nsi,joi->nsoj',x,atoms)

def search_coeff(phi,y,steps=6):
    j=phi.shape[-1];c=np.zeros(j,dtype=np.float64);target=y.reshape(-1).astype(np.float64);mat=phi.reshape(-1,j).astype(np.float64)
    def loss(v):return float(np.mean((mat@v-target)**2))
    best=loss(c);evals=1;step=.5
    for _ in range(steps):
        changed=False
        for i in range(j):
            for sign in (-1.,1.):
                trial=c.copy();trial[i]+=sign*step;value=loss(trial);evals+=1
                if value+1e-15<best:c=trial;best=value;changed=True
        if not changed:step*=.5
    return c.astype(np.float32),evals

def fit_methods(xs,ys,xq,base,atoms,qhat,weights,coeff):
    ps=features(xs,atoms);pq=features(xq,atoms);res=ys-np.einsum('nsi,oi->nso',xs,base)
    results={};
    search=[];full=[];mirror=[];oracle=[]
    for n in range(len(xs)):
        phi=ps[n].reshape(-1,J);target=res[n].reshape(-1)
        c_search,ne=search_coeff(ps[n],res[n]);search.append((c_search,ne))
        c_full=np.linalg.lstsq(phi,target,rcond=None)[0].astype(np.float32);full.append(c_full)
        qm=qhat;latent=np.linalg.lstsq(phi@qm,target,rcond=None)[0].astype(np.float32);mirror.append(latent)
        oracle.append(coeff[n])
    csearch=np.stack([v[0] for v in search]);evals=np.asarray([v[1] for v in search])
    cfull=np.stack(full);mcode=np.stack(mirror);cmirror=mcode@qhat.T
    coracle=np.stack(oracle)
    for method,c in [('lorahub_search',csearch),('native_full_coeff_ls',cfull),('mirror_code',cmirror),('native_lowrank_code',cmirror.copy()),('oracle_coeff',coracle)]:
        pred=np.einsum('nqoj,nj->nqo',pq,c)+np.einsum('nqi,oi->nqo',xq,base)
        err=float(np.sqrt(np.mean((pred-np.einsum('nqi,noi->nqo',xq,weights))**2)/np.mean(np.einsum('nqi,noi->nqo',xq,weights)**2)))
        results[method]=(c,err)
    results['mirror_code']=(mcode,results['mirror_code'][1])
    results['native_lowrank_code']=(mcode.copy(),results['native_lowrank_code'][1])
    results['search_evals']=evals
    return results

def save(name,payload):
    p=ROOT/'source'/name;started=time.perf_counter();np.savez_compressed(p,**payload)
    return int(p.stat().st_size),time.perf_counter()-started

def run_condition(seed,n,shots,rho):
    base,left,right,atoms,qhat,coeff,weights,xs,xq,ys,yq=make_world(seed,n,shots,rho)
    fits=fit_methods(xs,ys,xq,base,atoms,qhat,weights,coeff)
    payloads={
      'independent_full':{'schema':SCHEMA,'task_ids':np.arange(n,dtype=np.uint16),'task_matrices':weights},
      'lorahub_search':{'schema':SCHEMA.copy(),'task_ids':np.arange(n,dtype=np.uint16),'shared_base':base,'atom_left':left,'atom_right':right,'task_coefficients':fits['lorahub_search'][0],'search_steps':np.asarray([6],dtype=np.uint8)},
      'native_full_coeff_ls':{'schema':SCHEMA.copy(),'task_ids':np.arange(n,dtype=np.uint16),'shared_base':base.copy(),'atom_left':left.copy(),'atom_right':right.copy(),'task_coefficients':fits['native_full_coeff_ls'][0]},
      'mirror_code':{'schema':SCHEMA.copy(),'task_ids':np.arange(n,dtype=np.uint16),'shared_base':base.copy(),'atom_left':left.copy(),'atom_right':right.copy(),'coefficient_basis':qhat,'mirror_codes':fits['mirror_code'][0]},
      'native_lowrank_code':{'schema':SCHEMA.copy(),'task_ids':np.arange(n,dtype=np.uint16),'shared_base':base.copy(),'atom_left':left.copy(),'atom_right':right.copy(),'coefficient_basis':qhat.copy(),'task_codes':fits['native_lowrank_code'][0].copy()},
      'oracle_coeff':{'schema':SCHEMA.copy(),'task_ids':np.arange(n,dtype=np.uint16),'shared_base':base.copy(),'atom_left':left.copy(),'atom_right':right.copy(),'task_coefficients':fits['oracle_coeff'][0]},
    }
    evals=fits['search_evals'];rows=[];query_proxy=n*QUERIES*D*J
    for method,payload in payloads.items():
        size,wall=save(f'{seed}_n{n}_shots{shots}_rho{rho:g}_{method}.npz',payload)
        if method=='lorahub_search':adapt=int(np.sum(evals)*shots*D*J)
        elif method=='mirror_code':adapt=n*shots*D*R+META*J*R*R
        elif method=='native_lowrank_code':adapt=n*shots*D*R+META*J*R*R
        elif method=='native_full_coeff_ls':adapt=n*shots*D*J
        else:adapt=0
        val=fits[method][1]
        rows.append({'seed':seed,'task_count':n,'support_examples':shots,'residual_rms':rho,'method':method,'serialized_bytes':size,'heldout_nrmse':val,'task_queries':n*QUERIES,'optimizer_updates':0,'function_evaluations':int(np.sum(evals)) if method=='lorahub_search' else int(n if method in ('mirror_code','native_lowrank_code','native_full_coeff_ls') else 0),'adaptation_compute_proxy':adapt,'query_compute_proxy':int(query_proxy),'meta_tasks':META,'wall_time_s':wall,'search_evaluations_mean':float(np.mean(evals))})
    for a,b in (('mirror_code','native_lowrank_code'),):
        pa,pb=payloads[a],payloads[b]
        # Payload field names differ for clarity; all common physical state and code arrays are equal.
        for key1,key2 in [('shared_base','shared_base'),('atom_left','atom_left'),('atom_right','atom_right'),('coefficient_basis','coefficient_basis')]:assert np.array_equal(pa[key1],pb[key2])
        assert np.array_equal(fits[a][0],fits[b][0]) and fits[a][1]==fits[b][1]
    return rows

def run(seeds):
    rows=[r for seed in seeds for n in TASKS for shots in SHOTS for rho in RHOS for r in run_condition(seed,n,shots,rho)]
    (ROOT/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
        for r in rows:w.writerow([f"n{r['task_count']}_shots{r['support_examples']}_rho{r['residual_rms']:g}",r['seed'],r['method'],r['serialized_bytes'],r['task_queries']+r['support_examples']*r['task_count'],0,r['adaptation_compute_proxy']+r['query_compute_proxy'],r['wall_time_s'],'heldout_nrmse',r['heldout_nrmse'],'function_evaluations',r['function_evaluations'],'development; fresh sealed'])
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);a=ap.parse_args();print(json.dumps(run(a.seeds),indent=2))
