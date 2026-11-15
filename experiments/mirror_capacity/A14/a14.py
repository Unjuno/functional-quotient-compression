# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A14 privileged IID+OOD code oracle; *not* valid held-out adaptation.

Uses EXACT A13 frozen parents and support-only ES codes; only new task codes change.
The IID and OOD labels intentionally enter oracle code selection, so the resulting
numbers measure observed test-set fitting, NOT predictive generalization.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
def select_a13_root(here:Path)->Path:
    repo=here.parent/'A13'
    return repo if (repo/'a13.py').is_file() else here.parent/'mirror_es_a13_20261115'

A13=select_a13_root(HERE)
if not (A13/'a13.py').is_file():raise RuntimeError('Pinned A13 source is not mounted')
sys.path.insert(0,str(A13))
from a13 import (es_optimize,candidate_predict,score_support,make_world,
                 make_combined_model,serialize,deserialize,Student,K_NEW,K_TRAIN,A11_DIR)
sys.path.insert(0,str(A11_DIR))
from a11 import evaluate

FAMILIES=('tent',)
WORLDS=(1200,1201,1202,1300,1301,1302)
SEEDS=(0,1)
METHODS=('mirror','shift','film')
POP=32
GEN=30
CAP=1024

def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()

def oracle_examples(world:dict,t:int)->tuple[torch.Tensor,torch.Tensor]:
    if not isinstance(t,int) or not 0<=t<K_NEW:raise ValueError('task out of range')
    return (torch.cat((world['new_iid_x'],world['new_ood_x']),dim=0),
            torch.cat((world['new_iid_y'][:,t],world['new_ood_y'][:,t]),dim=0))

def check_frozen_parent(parent:Student,child:Student,expected_sha:str):
    if sha(serialize(parent))!=expected_sha:raise RuntimeError('parent SHA mismatch')
    if parent.method!=child.method or parent.width!=child.width:raise RuntimeError('architecture altered')
    for name,param in parent.named_parameters():
        if name!='new_codes' and not torch.equal(param,dict(child.named_parameters())[name]):
            raise RuntimeError('shared/old parameters were modified: '+name)

def oracle_code(parent:Student,x,y,start,seed:int,pop=POP,generations=GEN)->dict:
    c=np.asarray(start,dtype=np.float64).reshape(2)
    before=score_support(parent,x,y,c)
    result=es_optimize(parent,x,y,c,seed=seed,pop=pop,generations=generations)
    return {'code':np.asarray(result['code'],dtype=np.float64),
            'incumbent_mse':before,'oracle_mse':result['best_support_mse'],
            'evaluations':result['evaluations'],
            'bound_hit_rate':result['bound_hit_rate'],
            'best_trace':result['trace'][-1]}

def verify_kernel(parent:Student,x,code,atol:float=3e-5):
    c=np.asarray(code,dtype=np.float64).reshape(1,2)
    numpy_pred=candidate_predict(parent,x,c)[0]
    model=make_combined_model(parent,np.broadcast_to(c.astype(np.float32),(K_NEW,2)).copy())
    ids=torch.full((len(x),),K_TRAIN,dtype=torch.long)
    with torch.no_grad(): torch_pred=model(x,ids).numpy()
    error=float(np.max(np.abs(numpy_pred-torch_pred)))
    if error>atol:raise RuntimeError(f'NP/PyTorch mismatch {error}')
    return error

def load_parent_and_es(base:Path,family:str,world:int,seed:int,method:str):
    lane='main' if world<1300 else 'replication'
    folder=base/lane
    ledger=json.loads((folder/'train.lock.json').read_text())
    es=[r for r in ledger['models'] if r['family']==family and r['world']==world and r['seed']==seed and r['method']==method and r['algorithm']=='es']
    if len(es)!=1:raise RuntimeError('ES source selection is not unique')
    rec=es[0]
    parents=[r for r in ledger['parents'] if r['sha256']==rec['parent_sha256']]
    if len(parents)!=1:raise RuntimeError('parent selection is not unique')
    parent_row=parents[0]
    parent_raw=(folder/parent_row['path']).read_bytes()
    es_raw=(folder/rec['path']).read_bytes()
    if sha(parent_raw)!=parent_row['sha256'] or len(parent_raw)!=parent_row['stored_bytes']:
        raise RuntimeError('archive parent SHA or size mismatch')
    if sha(es_raw)!=rec['sha256'] or len(es_raw)!=rec['stored_bytes']:
        raise RuntimeError('archive ES SHA or size mismatch')
    parent=deserialize(parent_raw)
    es_model=deserialize(es_raw)
    if serialize(parent)!=parent_raw or serialize(es_model)!=es_raw:
        raise RuntimeError('FP32 serializer roundtrip mismatch')
    check_frozen_parent(parent,es_model,parent_row['sha256'])
    w=make_world(world,family)
    if w['fingerprint']!=rec['world_fingerprint']:raise RuntimeError('world fingerprint changed')
    return parent,es_model,w,parent_row,rec

def one_run(args):
    out,base,fam,world,seed,method=args
    torch.set_num_threads(1)
    parent,es_model,w,pr,er=load_parent_and_es(Path(base),fam,world,seed,method)
    # Objective is intentionally built from labels belonging to A13's AUDIT splits.
    codes=[]
    task_records=[]
    t0=time.perf_counter()
    for t in range(K_NEW):
        x,y=oracle_examples(w,t)
        start=es_model.new_codes[t].detach().cpu().numpy()
        rs=oracle_code(parent,x,y,start,seed=7711049+world*127+seed*23+METHODS.index(method)*491+t*31)
        codes.append(rs['code'])
        task_records.append({'task_id':K_TRAIN+t,'incumbent_joint_mse':rs['incumbent_mse'],
                             'oracle_joint_mse':rs['oracle_mse'],
                             'evaluations':rs['evaluations'],'bound_hit_rate':rs['bound_hit_rate'],
                             'code':rs['code'].tolist()})
    code=np.stack(codes)
    model=make_combined_model(parent,code)
    check_frozen_parent(parent,model,pr['sha256'])
    raw=serialize(model)
    if serialize(deserialize(raw))!=raw:raise RuntimeError('serialization roundtrip mismatch')
    path=f'models/{fam}_w{world}_s{seed}_{method}_privileged.a11w'
    dest=Path(out)/path;dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:f.write(raw)
    return dict(path=path,sha256=sha(raw),stored_bytes=len(raw),family=fam,world=world,seed=seed,method=method,
                parent_sha256=pr['sha256'],support_es_sha256=er['sha256'],
                world_fingerprint=w['fingerprint'],task_records=task_records,
                total_new_candidate_evaluations=int(K_NEW*POP*GEN),
                elapsed_seconds=time.perf_counter()-t0)

def run(out:Path,base:Path=A13,workers:int=4):
    if out.exists() and any(out.iterdir()):raise FileExistsError('destination is not empty')
    out.mkdir(parents=True,exist_ok=True)
    from itertools import product
    specs=[(str(out),str(base),f,w,s,m) for f,w,s,m in product(FAMILIES,WORLDS,SEEDS,METHODS)]
    cfg=dict(experiment='A14_PRIVILEGED_ORACLE',family=FAMILIES,worlds=WORLDS,seeds=SEEDS,methods=METHODS,
             objective='privileged leak: A13 IID+OOD labels used to optimize new codes; NON-DEPLOYABLE',
             pop=POP,generations=GEN,code_box=[-4,4],source_sha256={f:sha((HERE/f).read_bytes()) for f in ('a14.py','PROTOCOL.md')},
             dependency_source_sha256={f:sha((base/f).read_bytes()) for f in ('a13.py',)},
             fixed_parent='A13 main+replication meta2400, originally trained on independent A11 meta data')
    raw=json.dumps(cfg,sort_keys=True,indent=2).encode()
    (out/'config.json').write_bytes(raw)
    start=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures=[ex.submit(one_run,s) for s in specs]
        for f in as_completed(futures):
            r=f.result();rows.append(r)
            print('A14_FIT',len(rows),'/',len(specs),r['world'],r['seed'],r['method'],'seconds',round(r['elapsed_seconds'],2),flush=True)
    rows.sort(key=lambda r:(r['family'],r['world'],r['seed'],r['method']))
    if len(rows)!=len(specs):raise RuntimeError('missing frozen outputs')
    lock={'config_sha256':sha(raw),'models':rows,'elapsed_seconds':time.perf_counter()-start}
    (out/'train.lock.json').write_text(json.dumps(lock,sort_keys=True,indent=2))
    print('A14_LOCKED',len(rows),'models',flush=True)

def audit(out:Path,base:Path=A13):
    cfgraw=(out/'config.json').read_bytes();cfg=json.loads(cfgraw)
    lock=json.loads((out/'train.lock.json').read_text())
    if lock['config_sha256']!=sha(cfgraw):raise RuntimeError('frozen config altered')
    for f,d in cfg['source_sha256'].items():
        if sha((HERE/f).read_bytes())!=d:raise RuntimeError('source altered')
    source_rows=[]
    for row in lock['models']:
        parent,es_model,w,pr,er=load_parent_and_es(base,row['family'],row['world'],row['seed'],row['method'])
        raw=(out/row['path']).read_bytes()
        if sha(raw)!=row['sha256'] or len(raw)!=row['stored_bytes']:
            raise RuntimeError('saved model SHA or size corrupted')
        model=deserialize(raw)
        if serialize(model)!=raw:raise RuntimeError('saved model roundtrip mismatch')
        check_frozen_parent(parent,model,pr['sha256'])
        if pr['sha256']!=row['parent_sha256'] or er['sha256']!=row['support_es_sha256']:
            raise RuntimeError('parent provenance mismatch')
        if row['world_fingerprint']!=w['fingerprint']:raise RuntimeError('world/source mismatch')
        sample=oracle_examples(w,0)[0]
        verify_kernel(model,sample,model.new_codes[0].detach().numpy())
        for label,m in (('support_es',es_model),('privileged_oracle',model)):
            old=evaluate(m,w['meta_dev_x'],w['meta_dev_y'],offset=0)
            iid=evaluate(m,w['new_iid_x'],w['new_iid_y'],offset=16)
            ood=evaluate(m,w['new_ood_x'],w['new_ood_y'],offset=16)
            supp=np.mean([score_support(m,w['support_x'][t],w['support_y'][t],m.new_codes[t].detach().numpy()) for t in range(K_NEW)])
            # Explicit source label: same test examples were used to fit the privileged oracle.
            source_rows.append(dict(family=row['family'],world=row['world'],seed=row['seed'],method=row['method'],algorithm=label,
                                    objective_leaks_audit_labels=int(label=='privileged_oracle'),
                                    nrmse_old=old['nrmse'],nrmse_iid=iid['nrmse'],nrmse_ood=ood['nrmse'],
                                    mse_support=float(supp),joint_passing_in_sample=int(max(old['nrmse'],iid['nrmse'],ood['nrmse'])<=0.20),
                                    stored_bytes=row['stored_bytes'],parent_sha256=row['parent_sha256']))
    import pandas as pd
    frame=pd.DataFrame(source_rows)
    frame.to_csv(out/'audit.csv',index=False,float_format='%.12g')
    worlds=frame.groupby(['family','world','method','algorithm'],as_index=False).agg(
        nrmse_old=('nrmse_old','mean'),nrmse_iid=('nrmse_iid','mean'),nrmse_ood=('nrmse_ood','mean'),
        mse_support=('mse_support','mean'),stored_bytes=('stored_bytes','first'))
    worlds['joint_passing_in_sample']=(worlds[['nrmse_old','nrmse_iid','nrmse_ood']].max(axis=1)<=.20).astype(int)
    worlds.to_csv(out/'worlds.csv',index=False,float_format='%.12g')
    summary={}
    for method in METHODS:
        for alg in ('support_es','privileged_oracle'):
            s=worlds[(worlds.method==method)&(worlds.algorithm==alg)]
            summary[method+'/'+alg]=dict(worlds=len(s),quality_pass_worlds=int(s.joint_passing_in_sample.sum()),
                mean_iid=float(s.nrmse_iid.mean()),mean_ood=float(s.nrmse_ood.mean()),
                mean_support_mse=float(s.mse_support.mean()),stored_bytes=int(s.stored_bytes.iloc[0]))
    (out/'summary.json').write_text(json.dumps(dict(summary=summary,models=len(lock['models']),audit_rows=len(frame),
        error_category='privileged oracle fitted to audit labels; in-sample descriptive, not held-out evidence'),indent=2,sort_keys=True))
    print('A14_AUDITED',len(lock['models']),'models',len(frame),'score rows',flush=True)
    for k,v in summary.items():print(k,v,flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['run','audit'])
    p.add_argument('--out',type=Path,default=HERE/'results')
    p.add_argument('--base',type=Path,default=A13)
    p.add_argument('--workers',type=int,default=4)
    o=p.parse_args()
    if o.action=='run':run(o.out,o.base,o.workers)
    else:audit(o.out,o.base)
