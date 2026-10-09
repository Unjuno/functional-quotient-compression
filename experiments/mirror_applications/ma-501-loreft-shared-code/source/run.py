#!/usr/bin/env python3
"""MA-501 synthetic LoReFT-style shared-subspace intervention screen."""
from __future__ import annotations
import argparse, hashlib, io, json, time
from pathlib import Path
import numpy as np
import torch
D,T,K,NTRAIN,NTEST=24,16,4,128,256
RHOS=(0.0,0.25,0.5,1.0)
METHODS=('identity','activation_add','task_loreft_r1','task_loreft_r4','shared_mirror','native_shared_code','full_matrix')
BASIS_TASKS=tuple(range(12)); HELDOUT_TASKS=tuple(range(12,16))

def orthogonal(g, d, k, avoid=None):
    x=torch.randn(d,k,generator=g)
    if avoid is not None: x=x-avoid@(avoid.T@x)
    q,_=torch.linalg.qr(x,mode='reduced')
    return q

def world(seed,rho):
    g=torch.Generator().manual_seed(seed+501)
    u=orthogonal(g,D,K);v=orthogonal(g,D,K)
    coeff=torch.randn(T,K,generator=g)*0.10
    deltas=[]
    for t in range(T):
        a=orthogonal(g,D,K,u);b=orthogonal(g,D,K,v);priv=torch.randn(K,generator=g)*0.10
        delta=(u*coeff[t])@v.T+rho*(a*priv)@b.T
        deltas.append(delta)
    deltas=torch.stack(deltas)
    xtr=torch.randn(T,NTRAIN,D,generator=g);xte=torch.randn(T,NTEST,D,generator=g)
    ytr=xtr+torch.einsum('tni,tji->tnj',xtr,deltas)
    yte=xte+torch.einsum('tni,tji->tnj',xte,deltas)
    estimates=[]
    for t in range(T):
        # Exact least-squares estimate whenever the 128 examples span R^24.
        w=torch.linalg.lstsq(xtr[t],ytr[t]-xtr[t]).solution
        estimates.append(w.T)
    return {'delta':deltas,'xtr':xtr,'ytr':ytr,'xte':xte,'yte':yte,'est':torch.stack(estimates)}

def fit_shared(est):
    start=time.perf_counter()
    left=sum((est[t]@est[t].T for t in BASIS_TASKS),torch.zeros(D,D))
    right=sum((est[t].T@est[t] for t in BASIS_TASKS),torch.zeros(D,D))
    _,u=torch.linalg.eigh(left);_,v=torch.linalg.eigh(right)
    u=u[:,-K:];v=v[:,-K:]
    codes=torch.stack([torch.diag(u.T@est[t]@v) for t in range(T)])
    return u,v,codes,time.perf_counter()-start

def fit_independent(est,rank):
    start=time.perf_counter();left=[];sing=[];right=[]
    for mat in est:
        u,s,vh=torch.linalg.svd(mat,full_matrices=False);left.append(u[:,:rank]);sing.append(s[:rank]);right.append(vh[:rank,:].T)
    return torch.stack(left),torch.stack(sing),torch.stack(right),time.perf_counter()-start

def estimate_add(ytr,xtr):
    start=time.perf_counter();b=(ytr-xtr).mean(1);return b,time.perf_counter()-start

def pack(method, objects):
    schema={'format':'MA501-hidden-intervention-v1','method_class':method,'hidden_dimension':D,'tasks':T,'rank':K if method in ('shared_mirror','native_shared_code') else None}
    arrays={k:v.detach().cpu().numpy().astype(np.float32) for k,v in objects.items()}
    arrays['schema_json']=np.frombuffer(json.dumps(schema,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
    bio=io.BytesIO();np.savez(bio,**arrays);return bio.getvalue()

def apply(method,payload,x,task):
    if method=='identity':return x
    if method=='activation_add':return x+payload['bias'][task]
    if method in ('task_loreft_r1','task_loreft_r4'):
        l=payload['left'][task];s=payload['singular'][task];r=payload['right'][task]
        return x+(x@r*s)@l.T
    if method in ('shared_mirror','native_shared_code'):
        u=payload['left'];v=payload['right'];m=payload['codes'][task]
        return x+(x@v*m)@u.T
    if method=='full_matrix':return x+x@payload['delta'][task].T
    raise ValueError(method)

def make_objects(method,w):
    if method=='identity':return {}
    if method=='activation_add':return {'bias':estimate_add(w['ytr'],w['xtr'])[0]}
    if method=='task_loreft_r1':
        l,s,r,_=fit_independent(w['est'],1);return {'left':l,'singular':s,'right':r}
    if method=='task_loreft_r4':
        l,s,r,_=fit_independent(w['est'],K);return {'left':l,'singular':s,'right':r}
    if method in ('shared_mirror','native_shared_code'):
        u,v,c,_=fit_shared(w['est']);return {'left':u,'right':v,'codes':c}
    if method=='full_matrix':return {'delta':w['est']}
    raise ValueError(method)

def score(method,objects,w):
    pred=[];truth=[];start=time.perf_counter()
    for t in range(T):pred.append(apply(method,objects,w['xte'][t],t));truth.append(w['yte'][t])
    infer=time.perf_counter()-start;pred=torch.stack(pred);truth=torch.stack(truth);res=pred-truth
    base=(truth-w['xte']).square().mean().sqrt().clamp_min(1e-12)
    rel=float(res.square().mean().sqrt()/base)
    held=(pred[list(HELDOUT_TASKS)]-truth[list(HELDOUT_TASKS)])
    held_base=(truth[list(HELDOUT_TASKS)]-w['xte'][list(HELDOUT_TASKS)]).square().mean().sqrt().clamp_min(1e-12)
    hrel=float(held.square().mean().sqrt()/held_base)
    per={str(t):float((pred[t]-truth[t]).square().mean().sqrt()/((truth[t]-w['xte'][t]).square().mean().sqrt().clamp_min(1e-12))) for t in HELDOUT_TASKS}
    if method=='identity':ops=0
    elif method=='activation_add':ops=D
    elif method=='task_loreft_r1':ops=2*D+1
    elif method=='task_loreft_r4':ops=2*D*K+K
    elif method in ('shared_mirror','native_shared_code'):ops=2*D*K+K
    else:ops=D*D
    causal=0.
    if method in ('shared_mirror','native_shared_code'):
        zero=objects['codes'].clone();zero[12:]=0
        perm=objects['codes'].clone();perm[12:]=perm[12:].flip(0)
        cp=[];pp=[]
        for t in HELDOUT_TASKS:
            cp.append(w['xte'][t]+(w['xte'][t]@objects['right']*zero[t])@objects['left'].T)
            pp.append(w['xte'][t]+(w['xte'][t]@objects['right']*perm[t])@objects['left'].T)
        causal=float(torch.maximum((torch.stack(cp)-pred[list(HELDOUT_TASKS)]).abs().max(),(torch.stack(pp)-pred[list(HELDOUT_TASKS)]).abs().max()))
    return {'all_task_relative_rmse':rel,'heldout_task_relative_rmse':hrel,'per_heldout_task_relative_rmse':per,'heldout_task_count':len(HELDOUT_TASKS),'active_compute_proxy_per_example':ops,'active_compute_proxy_all_eval_examples':ops*T*NTEST,'inference_wall_s':infer,'m_causal_max_output_change':causal}

def run(seed,out):
    out.mkdir(parents=True,exist_ok=True);doc={'experiment_id':'MA-501','seed':seed,'split':'dev','dimensions':{'tasks':T,'hidden':D,'rank':K,'train_examples_per_task':NTRAIN,'heldout_examples_per_task':NTEST},'rhos':{}}
    for rho in RHOS:
        w=world(seed,rho);rd=out/f'rho_{rho:g}';rd.mkdir(exist_ok=True);items={}
        cache={}
        for method in METHODS:
            key='shared_mirror' if method=='native_shared_code' else method
            if key not in cache:
                start=time.perf_counter();obj=make_objects(key,w);fitwall=time.perf_counter()-start;raw=pack(key,obj);cache[key]=(obj,raw,fitwall)
            obj,raw,fitwall=cache[key];(rd/f'{method}_payload.npz').write_bytes(raw)
            met=score(key,obj,w);met.update({'inference_payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'fit_wall_s':fitwall,'optimizer_updates':0,'calibration_examples':T*NTRAIN,'task_basis_training_examples':len(BASIS_TASKS)*NTRAIN})
            items[method]=met
        assert (rd/'shared_mirror_payload.npz').read_bytes()==(rd/'native_shared_code_payload.npz').read_bytes()
        doc['rhos'][str(rho)]=items
    (out/'metrics.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');return doc

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
