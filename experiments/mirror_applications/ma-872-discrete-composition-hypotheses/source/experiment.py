"""Discrete task-hypothesis screen for MA-872."""
from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

K=8
D=8
R=2
N_CODES=8
N_TRAIN_PER_CODE=3
N_TRAIN=N_CODES*N_TRAIN_PER_CODE
N_HELDOUT=16
SUPPORT=(1,2,4,8,16,32)
NOISE=.03
HEADER=struct.Struct('<5sBBBB')


def rng(seed:int, tag:int)->np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed,tag]))


def make_world(seed:int, ntest:int=2048)->dict[str,Any]:
    g=rng(seed,1)
    w=g.normal(size=(K,D))/np.sqrt(D)
    b=g.uniform(-.7,.7,size=K)
    basis=g.normal(size=(R,K)); basis,_=np.linalg.qr(basis.T); basis=basis[:,:R].T
    angles=2*np.pi*np.arange(N_CODES)/N_CODES
    codes=np.stack([np.cos(angles),np.sin(angles)],axis=1)
    prototypes=codes@basis
    train_phi=[]; train_y=[]; train_labels=[]
    for code in range(N_CODES):
        for rep in range(N_TRAIN_PER_CODE):
            tag=100+code*N_TRAIN_PER_CODE+rep; rr=rng(seed,tag)
            x=rr.uniform(-1,1,size=(128,D)); phi=np.tanh(x@w.T+b)
            y=phi@prototypes[code]+rr.normal(0,NOISE,size=128)
            train_phi.append(phi); train_y.append(y); train_labels.append(code)
    support_phi=[]; support_y=[]; test_phi=[]; test_labels=[]
    rr=rng(seed,9000)
    labels=rr.integers(0,N_CODES,size=N_HELDOUT)
    for task,label in enumerate(labels):
        tr=rng(seed,10000+task)
        x=tr.uniform(-1,1,size=(max(SUPPORT),D)); ph=np.tanh(x@w.T+b)
        y=ph@prototypes[label]+tr.normal(0,NOISE,size=max(SUPPORT))
        xt=tr.uniform(-1,1,size=(ntest,D)); pht=np.tanh(xt@w.T+b)
        support_phi.append(ph); support_y.append(y); test_phi.append(pht); test_labels.append(int(label))
    return {'seed':seed,'w':w,'b':b,'true_basis':basis,'true_prototypes':prototypes,
            'train_phi':train_phi,'train_y':train_y,'train_labels':train_labels,
            'support_phi':support_phi,'support_y':support_y,'test_phi':test_phi,'test_labels':test_labels}


def fit_models(world:dict[str,Any])->dict[str,np.ndarray]:
    train_coeffs=[]
    for phi,y in zip(world['train_phi'],world['train_y']):
        train_coeffs.append(np.linalg.lstsq(phi,y,rcond=None)[0])
    train_coeffs=np.stack(train_coeffs)
    means=np.stack([train_coeffs[np.array(world['train_labels'])==i].mean(axis=0) for i in range(N_CODES)])
    center=train_coeffs.mean(axis=0)
    _,_,vt=np.linalg.svd(train_coeffs-center,full_matrices=False)
    pca_basis=vt[:R]
    rg=rng(world['seed'],44); random_codes=rg.normal(size=(N_CODES,K)); random_codes/=np.linalg.norm(random_codes,axis=1,keepdims=True)
    pooled=np.mean(train_coeffs,axis=0)
    return {'train_coeffs':train_coeffs,'prototypes':means,'pca_basis':pca_basis,'random_prototypes':random_codes,'pooled':pooled}


def serialize(world:dict[str,Any],model:dict[str,np.ndarray],method:str,codes:np.ndarray|None=None,indices:list[int]|None=None)->bytes:
    ids={'mirror':1,'native_codebook':2,'continuous':3,'random_codebook':4,'independent':5,'pooled':6,'teacher':7}
    out=bytearray(HEADER.pack(b'MA872',ids[method],K,R,N_CODES))
    out.extend(np.asarray(world['w'],dtype='<f4').tobytes(order='C'))
    out.extend(np.asarray(world['b'],dtype='<f4').tobytes(order='C'))
    if method in ('mirror','native_codebook'):
        out.extend(np.asarray(model['prototypes'],dtype='<f4').tobytes(order='C'))
        out.extend(bytes(indices or []))
    elif method=='continuous':
        out.extend(np.asarray(model['pca_basis'],dtype='<f4').tobytes(order='C'))
        out.extend(np.asarray(codes,dtype='<f4').tobytes(order='C'))
    elif method=='random_codebook':
        out.extend(np.asarray(model['random_prototypes'],dtype='<f4').tobytes(order='C'))
        out.extend(bytes(indices or []))
    elif method in ('independent','teacher'):
        out.extend(np.asarray(codes,dtype='<f4').tobytes(order='C'))
    elif method=='pooled':
        out.extend(np.asarray(model['pooled'],dtype='<f4').tobytes(order='C'))
    return bytes(out)


def evaluate(seed:int)->dict[str,Any]:
    wall0=time.perf_counter(); world=make_world(seed); t0=time.perf_counter(); model=fit_models(world); fit_s=time.perf_counter()-t0
    methods=('mirror','native_codebook','continuous','random_codebook','independent','pooled','teacher')
    by_support={}; N=len(world['test_labels'])
    for shots in SUPPORT:
        per={m:[] for m in methods}; task_indices={m:[] for m in methods}; adapt_s={m:0. for m in methods}; infer_s={m:0. for m in methods}; total_search_macs=0
        for task,label in enumerate(world['test_labels']):
            ph=world['support_phi'][task][:shots]; y=world['support_y'][task][:shots]; test=world['test_phi'][task]; target=world['true_prototypes'][label]
            # Mirror address and ordinary native nearest-codebook are deliberately the same rule.
            for method,table in (('mirror',model['prototypes']),('native_codebook',model['prototypes']),('random_codebook',model['random_prototypes'])):
                t1=time.perf_counter(); errs=[np.mean((ph@c-y)**2) for c in table]; idx=int(np.argmin(errs)); adapt_s[method]+=time.perf_counter()-t1; total_search_macs+=len(table)*shots*K*2
                coeff=table[idx]; task_indices[method].append(idx)
                t1=time.perf_counter(); pred=test@coeff; infer_s[method]+=time.perf_counter()-t1
                per[method].append(float(np.mean((pred-test@target)**2)))
            # continuous two-coordinate fit in training-task PCA subspace
            t1=time.perf_counter(); a=ph@model['pca_basis'].T; z=np.linalg.lstsq(a,y,rcond=None)[0]; coeff=z@model['pca_basis']; adapt_s['continuous']+=time.perf_counter()-t1
            t1=time.perf_counter(); pred=test@coeff; infer_s['continuous']+=time.perf_counter()-t1
            per['continuous'].append(float(np.mean((pred-test@target)**2))); task_indices['continuous'].append(z)
            t1=time.perf_counter(); coeff=np.linalg.lstsq(ph,y,rcond=None)[0]; adapt_s['independent']+=time.perf_counter()-t1
            t1=time.perf_counter(); pred=test@coeff; infer_s['independent']+=time.perf_counter()-t1
            per['independent'].append(float(np.mean((pred-test@target)**2))); task_indices['independent'].append(coeff)
            t1=time.perf_counter(); pred=test@model['pooled']; infer_s['pooled']+=time.perf_counter()-t1
            per['pooled'].append(float(np.mean((pred-test@target)**2))); task_indices['pooled'].append(model['pooled'])
            per['teacher'].append(0.0); task_indices['teacher'].append(target)
        codes={m:np.stack(task_indices[m]) if isinstance(task_indices[m][0],np.ndarray) else None for m in methods}
        inds={m:[int(x) for x in task_indices[m]] for m in ('mirror','native_codebook','random_codebook')}
        payload={
            'mirror':len(serialize(world,model,'mirror',indices=inds['mirror'])),
            'native_codebook':len(serialize(world,model,'native_codebook',indices=inds['native_codebook'])),
            'continuous':len(serialize(world,model,'continuous',codes=codes['continuous'])),
            'random_codebook':len(serialize(world,model,'random_codebook',indices=inds['random_codebook'])),
            'independent':len(serialize(world,model,'independent',codes=codes['independent'])),
            'pooled':len(serialize(world,model,'pooled')),
            'teacher':len(serialize(world,model,'teacher',codes=codes['teacher']))}
        by_support[str(shots)]={'mean_mse':{m:float(np.mean(per[m])) for m in methods},'worst_task_mse':{m:float(max(per[m])) for m in methods},'per_task_mse':per,'payload_bytes':payload,'adaptation_seconds':adapt_s,'inference_seconds':infer_s,'indices':inds,'search_macs_per_task':{m:N_CODES*shots*K*2 for m in ('mirror','native_codebook','random_codebook')}}
    return {'seed':seed,'prototype_fit_seconds':fit_s,'by_support':by_support,'module_evaluation_mac_proxy_per_query':K*D+K,'continuous_decoder_mac_proxy':2*R*K+2*K,'direct_decoder_mac_proxy':2*K,'max_wall_seconds':time.perf_counter()-wall0}


def run(seeds:list[int])->dict[str,Any]:
    rows=[evaluate(s) for s in seeds]; methods=('mirror','native_codebook','continuous','random_codebook','independent','pooled','teacher')
    out={'experiment_id':'MA-872','seeds':seeds,'support_sizes':list(SUPPORT),'rows':rows}
    out['mean_mse_by_support']={n:{m:float(np.mean([r['by_support'][n]['mean_mse'][m] for r in rows])) for m in methods} for n in map(str,SUPPORT)}
    out['max_world_mean_mse_by_support']={n:{m:max(r['by_support'][n]['mean_mse'][m] for r in rows) for m in methods} for n in map(str,SUPPORT)}
    out['max_task_mse_by_support']={n:{m:max(r['by_support'][n]['worst_task_mse'][m] for r in rows) for m in methods} for n in map(str,SUPPORT)}
    out['payload_bytes']={m:int(np.mean([r['by_support']['4']['payload_bytes'][m] for r in rows])) for m in methods}
    out['mean_adaptation_seconds_at_4']={m:float(np.mean([r['by_support']['4']['adaptation_seconds'][m] for r in rows])) for m in methods}
    out['mean_inference_seconds_at_4']={m:float(np.mean([r['by_support']['4']['inference_seconds'][m] for r in rows])) for m in methods}
    out['mean_prototype_fit_seconds']=float(np.mean([r['prototype_fit_seconds'] for r in rows])); out['max_wall_seconds']=max(r['max_wall_seconds'] for r in rows)
    return out


def write_json(path:str|Path,data:dict[str,Any])->None: Path(path).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
