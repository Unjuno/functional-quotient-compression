# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A12b post-A12 diagnostic: can new-task joint fine-tuning rescue a frozen scaffold?"""
from __future__ import annotations
import argparse, copy, concurrent.futures, csv, hashlib, json, multiprocessing as mp, sys, time
from collections import defaultdict
from pathlib import Path
import torch
from a12 import (K_TRAIN,K_NEW,FAMILIES,METHODS,PROTOCOL,make_world,make_support,
                 adapt_branch,serialize,deserialize,evaluate)

A12B_PROTOCOL=Path(__file__).with_name('A12B_PROTOCOL.md')


def train_joint_model(parent,world:dict,world_id:int,family:str,seed:int,steps:int=2400,batch:int=128):
    """Train all shared params on 16 existing + 8 new task labels. Diagnostic only."""
    torch.set_num_threads(1)
    if steps<1 or batch<1:raise ValueError('invalid steps')
    model=copy.deepcopy(parent)
    model.train()
    for p in model.parameters():p.requires_grad_(True)
    x_old=world['meta_x'];y_old=world['train_y']
    x_new,y_new=make_support(world_id,family,1024)
    gen=torch.Generator().manual_seed(850001+world_id*107+seed*347)
    opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0)
    for _ in range(steps):
        task=torch.randint(K_TRAIN+K_NEW,(batch,),generator=gen)
        i=torch.randint(1024,(batch,),generator=gen)
        old_mask=task<K_TRAIN
        old_task=torch.clamp(task,max=K_TRAIN-1)
        new_task=torch.clamp(task-K_TRAIN,min=0)
        inputs=torch.where(old_mask[:,None],x_old[i],x_new[new_task,i])
        labels=torch.where(old_mask,y_old[i,old_task],y_new[new_task,i])
        opt.zero_grad(set_to_none=True)
        loss=(model(inputs,task)-labels).square().mean()
        if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite joint loss')
        loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not bool(torch.isfinite(gn)):raise RuntimeError('nonfinite joint gradient')
        opt.step()
    return model.eval()


def _one(job):
    parent_root,out,fam,wid,seed,method,parent_record=job
    raw=(Path(parent_root)/parent_record['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=parent_record['sha256']:
        raise RuntimeError('A12 parent checkpoint hash mismatch')
    parent=deserialize(raw)
    world=make_world(wid,fam)
    start=time.perf_counter()
    _,snap=adapt_branch(parent,world,wid,fam,seed,method,support_n=1024,steps=(2400,),batch=128)
    code=snap[2400]
    joint=train_joint_model(parent,world,wid,fam,seed,steps=2400,batch=128)
    receipts=[]
    for variant,model in [('code',code),('joint',joint)]:
        b=serialize(model)
        filename=f'{fam}_w{wid}_s{seed}_{method}_{variant}.a11w'
        with (Path(out)/'models'/filename).open('xb') as f:f.write(b)
        receipts.append(dict(family=fam,world=wid,seed=seed,method=method,variant=variant,
            path='models/'+filename,stored_bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),
            parent_meta_sha256=parent_record['sha256'],world_fingerprint=world['fingerprint'],width=model.width))
    return receipts,time.perf_counter()-start


def train_b(parent_root:Path,out:Path,workers:int=4)->dict:
    root=Path(parent_root);out=Path(out)
    if out.exists():raise FileExistsError(out)
    parent=json.loads((root/'train.lock.json').read_text())
    if parent['protocol_sha256']!=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest():
        raise RuntimeError('A12 parent protocol SHA mismatch')
    if parent['config_sha256']!=hashlib.sha256((root/'config.json').read_bytes()).hexdigest():
        raise RuntimeError('A12 parent config SHA mismatch')
    cfg=json.loads((root/'config.json').read_text())
    if cfg['experiment']!='A12_MAIN':raise RuntimeError('not A12 main')
    parents=[r for r in parent['meta_models'] if r['meta_steps']==2400]
    if len(parents)!=36:raise RuntimeError('unexpected meta parent count')
    out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir()
    jobs=[(str(root),str(out),r['family'],r['world'],r['seed'],r['method'],r) for r in parents]
    all_recs=[];start=time.perf_counter()
    if workers==1:
        for i,x in enumerate(jobs,1):
            recs,elapsed=_one(x);all_recs.extend(recs)
            if i%6==0:print('A12B TRAINED',i,'/',len(jobs),flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as ex:
            for i,(recs,elapsed) in enumerate(ex.map(_one,jobs,chunksize=1),1):
                all_recs.extend(recs)
                if i%6==0:print('A12B TRAINED',i,'/',len(jobs),flush=True)
    lock=dict(experiment='A12B_POST_DISCOVERY',parent_sha256=hashlib.sha256((root/'train.lock.json').read_bytes()).hexdigest(),
              protocol_sha256=hashlib.sha256(A12B_PROTOCOL.read_bytes()).hexdigest(),
              A12_protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              models=sorted(all_recs,key=lambda x:(x['family'],x['world'],x['method'],x['seed'],x['variant'])),
              workers=workers,wall_seconds=time.perf_counter()-start,meta_parent_count=len(parents))
    (out/'train.lock.json').write_text(json.dumps(lock,indent=2,allow_nan=False)+'\n')
    print('A12B FROZEN',len(all_recs),flush=True)
    return lock


def audit_b(parent_root:Path,out:Path)->dict:
    root=Path(parent_root);out=Path(out)
    lock=json.loads((out/'train.lock.json').read_text())
    if lock['protocol_sha256']!=hashlib.sha256(A12B_PROTOCOL.read_bytes()).hexdigest():
        raise RuntimeError('A12b protocol SHA mismatch')
    if lock['parent_sha256']!=hashlib.sha256((root/'train.lock.json').read_bytes()).hexdigest():
        raise RuntimeError('A12 parent lock mismatch')
    if (out/'audit.csv').exists():raise FileExistsError('audit exists')
    parents=json.loads((root/'train.lock.json').read_text())
    if parents['protocol_sha256']!=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest():raise RuntimeError('A12 protocol mismatch')
    metas={(r['family'],r['world'],r['method'],r['seed']):r for r in parents['meta_models'] if r['meta_steps']==2400}
    loaded=[]
    if len(lock['models'])!=72:raise RuntimeError('unexpected A12b model count')
    for r in lock['models']:
        key=(r['family'],r['world'],r['method'],r['seed'])
        if r['parent_meta_sha256']!=metas[key]['sha256']:
            raise RuntimeError('parent metadata mismatch')
        b=(out/r['path']).read_bytes()
        if hashlib.sha256(b).hexdigest()!=r['sha256'] or len(b)!=r['stored_bytes']:
            raise RuntimeError('model hash error')
        model=deserialize(b)
        if serialize(model)!=b or model.method!=r['method']:
            raise RuntimeError('serialization mismatch')
        loaded.append((r,model))
    worlds={(r['family'],r['world']):make_world(r['world'],r['family']) for r,_ in loaded}
    scored=[]
    for r,m in loaded:
        w=worlds[(r['family'],r['world'])]
        if r['world_fingerprint']!=w['fingerprint']:raise RuntimeError('world fingerprint mismatch')
        old=evaluate(m,w['meta_dev_x'],w['meta_dev_y'],0)
        iid=evaluate(m,w['new_iid_x'],w['new_iid_y'],K_TRAIN)
        ood=evaluate(m,w['new_ood_x'],w['new_ood_y'],K_TRAIN)
        scored.append({**r,'old_dev_nrmse':old['nrmse'],'new_iid_nrmse':iid['nrmse'],'new_ood_nrmse':ood['nrmse']})
    with (out/'audit.csv').open('x',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=scored[0].keys());wr.writeheader();wr.writerows(scored)
    grouped=defaultdict(list)
    for r in scored:grouped[(r['family'],r['world'],r['method'],r['variant'])].append(r)
    rows=[]
    for (family,world,method,variant),group in sorted(grouped.items()):
        rr=dict(family=family,world=world,method=method,variant=variant,seeds=len(group))
        for k in ('old_dev_nrmse','new_iid_nrmse','new_ood_nrmse'):
            rr[k]=sum(x[k] for x in group)/len(group)
        rr['quality_pass']=int(max(rr[k] for k in ('old_dev_nrmse','new_iid_nrmse','new_ood_nrmse'))<=.20)
        rows.append(rr)
    with (out/'world_means.csv').open('x',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    summary=dict(models=len(scored),split_scores=len(scored)*3,world_means=rows,
       boundary='Post-A12 discovery oracle: unfreezing and 1024-support code fitting are not matched deployment conditions.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--audit',action='store_true')
    a=p.parse_args()
    if a.audit:print(audit_b(a.parent,a.out)['models'])
    else:train_b(a.parent,a.out,a.workers)
