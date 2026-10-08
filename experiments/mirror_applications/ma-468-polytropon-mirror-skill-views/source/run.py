#!/usr/bin/env python3
"""Frozen MA-468 shared/private skill-bank representation screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
from torch import nn
D,PHYS,ALIGNED,PRIVATE,NSKILL,TASKS=8,3,6,4,10,12
UPDATES,BATCH,SUPPORT,QUERY=4000,32,128,256
ALLOC=[[0,1],[1,3],[2,4],[4,5],[0,5],[1,2],[3,4],[2,5],[6],[7],[8],[9]]
METHODS=('polytropon','physical_only','mirror','native_lowrank','independent')
COEFF=np.array([-.75,.75,-1.1,1.1,-.5,.5],np.float32)

def world(seed):
    rng=np.random.default_rng(seed+468);base=rng.normal(scale=.22,size=(PHYS,D,D)).astype(np.float32)
    u=rng.normal(size=D).astype(np.float32);u/=np.linalg.norm(u);v=rng.normal(size=D).astype(np.float32);v/=np.linalg.norm(v);basis=.8*np.outer(u,v)
    skills=[base[k//2]+COEFF[k]*basis for k in range(ALIGNED)]
    skills += [rng.normal(scale=.38,size=(D,D)).astype(np.float32) for _ in range(PRIVATE)]
    skills=torch.tensor(np.stack(skills));alloc=np.zeros((TASKS,NSKILL),np.uint8)
    for t,ids in enumerate(ALLOC):alloc[t,ids]=1
    mats=[]
    for ids in ALLOC:mats.append(torch.stack([skills[i] for i in ids]).sum(0)/(len(ids)**.5))
    mats=torch.stack(mats);xtr=np.empty((TASKS,SUPPORT,D),np.float32);ytr=np.empty_like(xtr);xq=np.empty((TASKS,QUERY,D),np.float32);yq=np.empty_like(xq)
    for t in range(TASKS):
        for split,xbank,ybank,n,off in [('train',xtr,ytr,SUPPORT,1000),('query',xq,yq,QUERY,90000)]:
            rr=np.random.default_rng(seed+off+t);x=rr.normal(size=(n,D)).astype(np.float32);xt=torch.tensor(x);y=xt@mats[t].T+torch.tensor(rr.normal(scale=.015,size=(n,D)).astype(np.float32));xbank[t]=x;ybank[t]=y.numpy()
    data={'xtr':torch.tensor(xtr),'ytr':torch.tensor(ytr),'xq':torch.tensor(xq),'yq':torch.tensor(yq)}
    teacher={'base':torch.tensor(base),'basis_u':torch.tensor(u),'basis_v':torch.tensor(v),'coeff':torch.tensor(COEFF),'skills':skills,'allocation':alloc,'task_matrices':mats}
    return teacher,data

def init(method,seed):
    torch.manual_seed(seed+46801)
    if method in ('mirror','native_lowrank'):return {'base':nn.Parameter(torch.randn(PHYS,D,D)*.08),'u':nn.Parameter(torch.randn(D)*.08),'v':nn.Parameter(torch.randn(D)*.08),'codes':nn.Parameter(torch.zeros(ALIGNED)),'private':nn.Parameter(torch.randn(PRIVATE,D,D)*.08)}
    if method=='physical_only':return {'modules':nn.Parameter(torch.randn(PHYS+PRIVATE,D,D)*.08)}
    if method=='polytropon':return {'skills':nn.Parameter(torch.randn(NSKILL,D,D)*.08)}
    return {'tasks':nn.Parameter(torch.randn(TASKS,D,D)*.08)}

def skill_matrix(method,s,k):
    if method in ('mirror','native_lowrank'):
        if k<ALIGNED:return s['base'][k//2]+s['codes'][k]*torch.outer(s['u'],s['v'])
        return s['private'][k-ALIGNED]
    if method=='physical_only':return s['modules'][k//2 if k<ALIGNED else k-ALIGNED+PHYS]
    if method=='polytropon':return s['skills'][k]
    raise ValueError(method)

def task_matrix(method,s,t,alloc):
    if method=='independent':return s['tasks'][t]
    ids=np.flatnonzero(alloc[t]);return torch.stack([skill_matrix(method,s,int(k)) for k in ids]).sum(0)/(len(ids)**.5)

def predict(method,s,t,x,alloc):return x@task_matrix(method,s,t,alloc).T

def train(method,seed,teacher,data):
    s=init(method,seed);opt=torch.optim.Adam(list(s.values()),lr=.01);rng=np.random.default_rng(seed+46811);start=time.perf_counter()
    for _ in range(UPDATES):
        t=int(rng.integers(TASKS));ix=rng.integers(SUPPORT,size=BATCH);x=data['xtr'][t,ix];y=data['ytr'][t,ix];p=predict(method,s,t,x,teacher['allocation']);loss=((p-y)**2).mean();opt.zero_grad();loss.backward();opt.step()
    return {k:v.detach() for k,v in s.items()},time.perf_counter()-start

def packed_arrays(method,s,teacher):
    alloc=teacher['allocation']
    if method in ('mirror','native_lowrank'):
        a={'shared_physical_skills':s['base'].numpy(),'view_basis_left':s['u'].numpy(),'view_basis_right':s['v'].numpy(),'logical_skill_view_codes':s['codes'].numpy(),'private_skills':s['private'].numpy(),'task_skill_allocation':alloc};fam='shared-rank1-view-skill-bank-v1'
    elif method=='physical_only':a={'physical_modules':s['modules'].numpy(),'skill_to_module':np.array([k//2 if k<ALIGNED else k-ALIGNED+PHYS for k in range(NSKILL)],np.int16),'task_skill_allocation':alloc};fam='physical-only-private-skill-bank-v1'
    elif method=='polytropon':a={'skill_bank':s['skills'].numpy(),'task_skill_allocation':alloc};fam='polytropon-explicit-skill-bank-v1'
    else:a={'task_matrices':s['tasks'].numpy(),'task_ids':np.arange(TASKS,dtype=np.int16)};fam='independent-task-functions-v1'
    return a,fam

def serialize(method,s,teacher):
    a,fam=packed_arrays(method,s,teacher);buf=io.BytesIO();d={k:np.asarray(v) for k,v in a.items()};d['__schema_json__']=np.frombuffer(json.dumps({'format':'MA468-skill-bank-v1','method_family':fam,'dimension':D,'tasks':TASKS,'logical_skills':NSKILL,'dtype':'float32'},sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);np.savez(buf,**d);return buf.getvalue()

def evaluate(method,s,teacher,data,wall):
    t0=time.perf_counter();scores=[];parts={'composition':[],'private':[]}
    with torch.no_grad():
        for t in range(TASKS):
            pred=predict(method,s,t,data['xq'][t],teacher['allocation']);y=data['yq'][t];e=float(torch.sqrt(((pred-y)**2).mean()));scores.append(e);parts['composition' if t<8 else 'private'].append(e)
    query_wall=time.perf_counter()-t0;raw=serialize(method,s,teacher)
    if method in ('mirror','native_lowrank'):mods=PHYS+PRIVATE;cold=ALIGNED*D*D*2+TASKS*D*D;per=D*D
    elif method=='physical_only':mods=PHYS+PRIVATE;cold=TASKS*D*D;per=D*D
    elif method=='polytropon':mods=NSKILL;cold=TASKS*D*D;per=D*D
    else:mods=TASKS;cold=0;per=D*D
    m={'mean_query_rmse':float(np.mean(scores)),'per_task_rmse':scores,'composition_rmse':float(np.mean(parts['composition'])),'private_rmse':float(np.mean(parts['private'])),'payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'physical_skill_modules':mods,'private_modules':PRIVATE if method in ('mirror','native_lowrank','physical_only') else 0,'logical_skills':NSKILL,'mean_active_skills_per_task':float(np.mean([len(x) for x in ALLOC])),'optimizer_updates':UPDATES,'examples_seen':UPDATES*BATCH,'training_wall_s':wall,'query_wall_s':query_wall,'cold_reconstruction_ops_proxy':cold,'inference_ops_proxy_per_example':per,'training_ops_proxy':UPDATES*BATCH*per*3,'evaluation_ops_proxy':TASKS*QUERY*per,'active_ops_proxy':UPDATES*BATCH*per*3+TASKS*QUERY*per,'query_examples':TASKS*QUERY}
    return m,raw

def run(seed,out):
    out.mkdir(parents=True,exist_ok=True);teacher,data=world(seed);results={}
    for method in METHODS:
        s,wall=train(method,seed,teacher,data);m,raw=evaluate(method,s,teacher,data,wall);results[method]=m;(out/f'{method}_payload.npz').write_bytes(raw)
    assert results['mirror']['payload_sha256']==results['native_lowrank']['payload_sha256']
    assert results['mirror']['per_task_rmse']==results['native_lowrank']['per_task_rmse']
    d={'experiment_id':'MA-468','seed':seed,'split':'dev','task':{'dimension':D,'tasks':TASKS,'logical_skills':NSKILL,'allocation':ALLOC,'support_per_task':SUPPORT,'query_per_task':QUERY},'methods':results};(out/'metrics.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
