#!/usr/bin/env python3
"""Frozen MA-457 continual path reuse before private module birth."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np
import torch
D=4;NTASK=8;META_UPDATES=1600;BATCH=32;CODE_UPDATES=300;SUPPORT=256;QUERY=256;THRESH=.05
COEFF=np.array([0.,.7,-.7,1.1,-1.1,1.6],dtype=np.float32)
METHODS=('pathnet','shared','mirror','native_rank1','independent')

def task_matrices(seed):
    rng=np.random.default_rng(seed+457);base=rng.normal(scale=.32,size=(D,D)).astype(np.float32)
    left=rng.normal(size=D).astype(np.float32);left/=np.linalg.norm(left)
    right=rng.normal(size=D).astype(np.float32);right/=np.linalg.norm(right)
    b=np.outer(left,right).astype(np.float32)*.9
    mats=[base+c*b for c in COEFF]
    mats += [base+rng.normal(scale=.65,size=(D,D)).astype(np.float32) for _ in range(2)]
    return torch.tensor(base),torch.tensor(left),torch.tensor(right),torch.tensor(np.stack(mats))

def task_data(seed,task,split):
    _,_,_,mats=task_matrices(seed);rng=np.random.default_rng(seed+9000+task*(1 if split=='support' else 13)+(0 if split=='support' else 100000))
    x=torch.tensor(rng.normal(size=(SUPPORT if split=='support' else QUERY,D)).astype(np.float32));m=mats[task]
    y=x@m.T+torch.tensor(rng.normal(scale=.01,size=x.shape).astype(np.float32))
    return x,y

def rmse(x,y,m):return float(torch.sqrt(((x@m.T-y)**2).mean()))
def fit_matrix(x,y):return torch.linalg.lstsq(x,y).solution.T.detach()
def rank1_output(x,w,u,v,c):return x@w.T + c[:,None]*(x@v)[:,None]*u[None,:]
def code_output(x,w,u,v,c):return x@w.T + c*(x@v)[:,None]*u[None,:]

def train_rank1(seed):
    torch.manual_seed(seed+45701);w=torch.randn(D,D)*.12;u=torch.randn(D)*.08;v=torch.randn(D)*.08;codes=torch.zeros(4,requires_grad=True)
    params=[torch.nn.Parameter(w),torch.nn.Parameter(u),torch.nn.Parameter(v),torch.nn.Parameter(codes)]
    opt=torch.optim.Adam(params,lr=.02);rng=np.random.default_rng(seed+45702);start=time.perf_counter()
    for _ in range(META_UPDATES):
        task=int(rng.integers(4));idx=rng.integers(SUPPORT,size=BATCH);x,y=task_data(seed,task,'support');xb=x[idx];yb=y[idx]
        pred=code_output(xb,params[0],params[1],params[2],params[3][task]);loss=((pred-yb)**2).mean();opt.zero_grad();loss.backward();opt.step()
    updates=META_UPDATES;seen=META_UPDATES*BATCH
    # The first four tasks are accepted together at the end of the frozen meta stage.
    for task in range(4):
        xq,yq=task_data(seed,task,'query');accepted[task]=float(torch.sqrt(((code_output(xq,params[0].detach(),params[1].detach(),params[2].detach(),params[3][task].detach())-yq)**2).mean()))
    # Sequential task arrival: preserve meta-task codes, fit each new scalar from support.
    learned={i:float(params[3][i].detach()) for i in range(4)}; births={};per_task=[]
    for task in range(4,NTASK):
        x,y=task_data(seed,task,'support');c=torch.nn.Parameter(torch.zeros(()));o=torch.optim.Adam([c],lr=.02)
        for _ in range(CODE_UPDATES):
            idx=rng.integers(SUPPORT,size=BATCH);pred=code_output(x[idx],params[0].detach(),params[1].detach(),params[2].detach(),c);loss=((pred-y[idx])**2).mean();o.zero_grad();loss.backward();o.step()
        learned[task]=float(c.detach());updates+=CODE_UPDATES;seen+=CODE_UPDATES*BATCH
        xq,yq=task_data(seed,task,'query');accepted[task]=float(torch.sqrt(((code_output(xq,params[0].detach(),params[1].detach(),params[2].detach(),c.detach())-yq)**2).mean()))
    # Birth only when the support set says the task-specific View is insufficient.
    for task in range(NTASK):
        x,y=task_data(seed,task,'support');c=learned[task]
        sr=rmse(x,y,params[0].detach()+c*torch.outer(params[1].detach(),params[2].detach()))
        if sr>THRESH:births[task]=fit_matrix(x,y)
        if task in births:
            xq,yq=task_data(seed,task,'query');accepted[task]=rmse(xq,yq,births[task])
    wall=time.perf_counter()-start
    state={'w':params[0].detach(),'u':params[1].detach(),'v':params[2].detach(),'codes':learned,'births':births,'accepted_query_rmse':accepted}
    return state,{'optimizer_updates':updates,'sampled_examples':seen,'training_wall_s':wall,'module_births':len(births)}

def pathnet(seed):
    modules=[];paths=[];support_examples=0;accepted={};start=time.perf_counter()
    for task in range(NTASK):
        x,y=task_data(seed,task,'support');support_examples+=len(x)
        scores=[rmse(x,y,m) for m in modules]
        if scores and min(scores)<=THRESH:path=scores.index(min(scores))
        else:modules.append(fit_matrix(x,y));path=len(modules)-1
        paths.append(path)
        xq,yq=task_data(seed,task,'query');accepted[task]=rmse(xq,yq,modules[path])
    return {'modules':modules,'paths':paths,'accepted_query_rmse':accepted},{'optimizer_updates':0,'sampled_examples':support_examples,'training_wall_s':time.perf_counter()-start,'module_births':len(modules)}

def pooled_shared(seed):
    xs=[];ys=[]
    for t in range(NTASK):x,y=task_data(seed,t,'support');xs.append(x);ys.append(y)
    return fit_matrix(torch.cat(xs),torch.cat(ys))

def pack(arr,meta):
    b=io.BytesIO();d={k:np.asarray(v) for k,v in arr.items()};d['__schema_json__']=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);np.savez(b,**d);return b.getvalue()

def payload(method,state):
    if method in ('mirror','native_rank1'):
        births=state['births'];arr={'shared_module':state['w'].numpy(),'basis_left':state['u'].numpy(),'basis_right':state['v'].numpy(),'task_codes':np.array([state['codes'][i] for i in range(NTASK)],np.float32),'task_module_index':np.array([-1 if i not in births else list(sorted(births)).index(i) for i in range(NTASK)],np.int16),'private_modules':np.stack([births[i].numpy() for i in sorted(births)]) if births else np.zeros((0,D,D),np.float32)}
        fam='shared-rank1-task-code-v1'
    elif method=='pathnet':arr={'module_bank':np.stack([m.numpy() for m in state['modules']]),'task_path':np.array(state['paths'],np.int16)};fam='pathnet-frozen-module-bank-v1'
    elif method=='shared':arr={'shared_module':state.numpy()};fam='ordinary-single-shared-module-v1'
    else:arr={'task_modules':state.numpy()};fam='independent-task-module-bank-v1'
    return pack(arr,{'format':'MA457-continual-path-v1','method_family':fam,'dimension':D,'tasks':NTASK,'dtype':'float32'})

def predict_task(method,state,task,x):
    if method=='pathnet':return x@state['modules'][state['paths'][task]].T
    if method in ('mirror','native_rank1'):
        if task in state['births']:return x@state['births'][task].T
        w,u,v,c=state['w'],state['u'],state['v'],state['codes'][task];return x@w.T+c*(x@v)[:,None]*u
    if method=='shared':return x@state.T
    return x@state[task].T

def evaluate(method,state,seed,train_meta):
    scores=[];outs=[];start=time.perf_counter()
    for task in range(NTASK):
        x,y=task_data(seed,task,'query');pred=predict_task(method,state,task,x);scores.append(float(torch.sqrt(((pred-y)**2).mean())));outs.append(float(pred.std()))
    query_wall=time.perf_counter()-start;raw=payload(method,state)
    mac={'pathnet':D*D,'shared':D*D,'mirror':D*D+2*D,'native_rank1':D*D+2*D,'independent':D*D}[method]
    evalops=NTASK*QUERY*mac
    births=(len(state['births'])+1 if method in ('mirror','native_rank1') else train_meta['module_births']) if method!='shared' else 1
    m={'per_task_query_rmse':scores,'mean_query_rmse':float(np.mean(scores)),'max_early_task_retention_increase':float(max((scores[i]-state['accepted_query_rmse'][i] for i in range(min(4,NTASK))),default=0.0)) if 'accepted_query_rmse' in state else 0.0,'activation_output_std_ratio':max(outs)/min(outs),'payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'physical_modules_total':births,'module_births_after_initial':(len(state['births']) if method in ('mirror','native_rank1') else max(0,train_meta['module_births']-1) if method=='pathnet' else 0),'query_examples':NTASK*QUERY,'query_wall_s':query_wall,'inference_ops_per_example_proxy':mac,'evaluation_ops_proxy':evalops}
    m.update(train_meta);m['active_ops_proxy']=int(evalops+train_meta['sampled_examples']*mac*3)
    return m,raw

def run(seed,out):
    out.mkdir(parents=True,exist_ok=True);result={}
    p,t=pathnet(seed);result['pathnet'],raw=evaluate('pathnet',p,seed,t);(out/'pathnet_payload.npz').write_bytes(raw)
    s=pooled_shared(seed);meta={'optimizer_updates':0,'sampled_examples':8*SUPPORT,'training_wall_s':0.0,'module_births':1};result['shared'],raw=evaluate('shared',s,seed,meta);(out/'shared_payload.npz').write_bytes(raw)
    m,mt=train_rank1(seed)
    for name in ('mirror','native_rank1'):
        result[name],raw=evaluate(name,m,seed,mt);(out/f'{name}_payload.npz').write_bytes(raw)
    mats=[]
    for task in range(NTASK):x,y=task_data(seed,task,'support');mats.append(fit_matrix(x,y))
    ind=torch.stack(mats);it={'optimizer_updates':0,'sampled_examples':NTASK*SUPPORT,'training_wall_s':0.0,'module_births':NTASK};result['independent'],raw=evaluate('independent',ind,seed,it);(out/'independent_payload.npz').write_bytes(raw)
    assert result['mirror']['payload_sha256']==result['native_rank1']['payload_sha256']
    assert result['mirror']['per_task_query_rmse']==result['native_rank1']['per_task_query_rmse']
    d={'experiment_id':'MA-457','seed':seed,'split':'dev','task':{'dimension':D,'count':NTASK,'aligned_tasks':6,'outliers':2,'support_per_task':SUPPORT,'query_per_task':QUERY},'methods':result};(out/'metrics.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');return d

def main():
    a=argparse.ArgumentParser();a.add_argument('--seed',type=int,required=True);a.add_argument('--out',type=Path,required=True);x=a.parse_args();print(json.dumps(run(x.seed,x.out),sort_keys=True))
if __name__=='__main__':main()
