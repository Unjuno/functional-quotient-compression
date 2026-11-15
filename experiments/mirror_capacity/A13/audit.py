# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A13 post-lock audit. No optimization/search is performed in this module."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from a13 import make_world,deserialize,serialize,score_support,K_TRAIN,K_NEW
from a11 import evaluate,count_bytes

ROOT=Path(__file__).resolve().parent

def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()


def verify_frozen(out:Path,records:list[dict])->list[tuple[dict,object]]:
    models=[]
    for r in records:
        p=Path(out)/r['path']
        if not p.is_file():raise RuntimeError(f'missing locked model: {r["path"]}')
        raw=p.read_bytes()
        if sha(raw)!=r['sha256'] or len(raw)!=r['stored_bytes']:
            raise RuntimeError('locked hash/size mismatch: '+r['path'])
        m=deserialize(raw)
        if serialize(m)!=raw or count_bytes(m)!=r['stored_bytes']:
            raise RuntimeError('locked reserialization/size mismatch: '+r['path'])
        models.append((r,m))
    return models


def audit(out:Path):
    torch.set_num_threads(1)
    out=Path(out).resolve()
    if (out/'audit.csv').exists():raise FileExistsError('audit already exists')
    cfgraw=(out/'config.json').read_bytes()
    cfg=json.loads(cfgraw)
    lock=json.loads((out/'train.lock.json').read_bytes())
    if lock['config_sha256']!=sha(cfgraw):raise RuntimeError('config file fingerprint mismatch')
    for k,v in cfg['source_sha256'].items():
        if sha((ROOT/k).read_bytes())!=v:raise RuntimeError(f'source file changed: {k}')
    import a13
    if sha((a13.A11_DIR/'a11.py').read_bytes())!=cfg['inherited_a11_sha256']:
        raise RuntimeError('A11 source file mismatch')
    expected=(len(cfg['family'])*len(cfg['worlds'])*len(cfg['seeds'])*len(cfg['methods'])*3)
    if len(lock['models'])!=expected:raise RuntimeError('missing models in lock')
    # Crucial: read, hash, and round-trip the complete model set before any evaluation.
    models=verify_frozen(out,lock['models'])
    if len(lock['parents'])!=expected//3:raise RuntimeError('parent count mismatch')
    parents=verify_frozen(out,lock['parents'])
    if len(lock['traces'])!=expected//3:raise RuntimeError('trace count mismatch')
    for t in lock['traces']:
        if sha((out/t['path']).read_bytes())!=t['sha256']:
            raise RuntimeError('trace digest mismatch')
    old_sha={r['sha256']:m for r,m in parents}
    for row,m in models:
        if m.method!=row['method'] or row['parent_sha256'] not in old_sha or m.width!=old_sha[row['parent_sha256']].width:
            raise RuntimeError('method/parent mismatch')
        parent=old_sha[row['parent_sha256']]
        for (name,p),(_,q) in zip(m.named_parameters(),parent.named_parameters()):
            if name!='new_codes' and not torch.equal(p,q):
                raise RuntimeError('shared/old weights altered')
    print('VERIFY_ALL_FROZEN',len(models),'models',len(parents),'parents',len(lock['traces']),'traces',flush=True)
    worlds={(f,w):make_world(w,f) for f in cfg['family'] for w in cfg['worlds']}
    rows=[];t0=time.perf_counter()
    for r,m in models:
        w=worlds[(r['family'],r['world'])]
        if w['fingerprint']!=r['world_fingerprint']:
            raise RuntimeError('data world fingerprint mismatch')
        old=evaluate(m,w['meta_dev_x'],w['meta_dev_y'],0)
        iid=evaluate(m,w['new_iid_x'],w['new_iid_y'],K_TRAIN)
        ood=evaluate(m,w['new_ood_x'],w['new_ood_y'],K_TRAIN)
        sup_p=m(w['support_x'].reshape(-1,1),torch.arange(K_TRAIN,K_TRAIN+K_NEW,dtype=torch.long).repeat_interleave(w['support_x'].shape[1])).reshape(K_NEW,-1)
        sup_err=float((sup_p-w['support_y']).square().mean().item())
        rows.append({**r,'old_dev_nrmse':old['nrmse'],'iid_nrmse':iid['nrmse'],
                     'ood_nrmse':ood['nrmse'],'old_dev_mse':old['mse'],'iid_mse':iid['mse'],'ood_mse':ood['mse'],
                     'support_mse_reloaded':sup_err,
                     'joint_quality_pass':int(max(old['nrmse'],iid['nrmse'],ood['nrmse'])<=.20),
                     'support_consistency_abs':abs(sup_err-r['support_mse_mean'])})
    with (out/'audit.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    group=defaultdict(list)
    for r in rows:
        group[(r['family'],r['world'],r['method'],r['algorithm'])].append(r)
    world_rows=[]
    for key,a in sorted(group.items()):
        family,world,method,alg=key
        if len(a)!=len(cfg['seeds']):raise RuntimeError('not all paired seeds included')
        outrow={'family':family,'world':world,'method':method,'algorithm':alg,'seeds':len(a),
                'stored_bytes':float(np.mean([v['stored_bytes'] for v in a]))}
        for key in ('old_dev_nrmse','iid_nrmse','ood_nrmse','support_mse_reloaded','code_l2_mean'):
            outrow[key]=float(np.mean([v[key] for v in a]))
        outrow['joint_quality_pass']=int(max(outrow['old_dev_nrmse'],outrow['iid_nrmse'],outrow['ood_nrmse'])<=.20)
        world_rows.append(outrow)
    with (out/'world_means.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=world_rows[0].keys());writer.writeheader();writer.writerows(world_rows)
    if any(r['support_consistency_abs']>2e-5 for r in rows):
        raise RuntimeError('support objective implementation mismatch against reloaded float32 models')
    per_world={(r['family'],r['world'],r['method'],r['algorithm']):r for r in world_rows}
    contrasts=[]
    for fam in cfg['family']:
        for world in cfg['worlds']:
            for method in cfg['methods']:
                es=per_world[(fam,world,method,'es')]
                for alg in ('random','adam'):
                    ctrl=per_world[(fam,world,method,alg)]
                    contrasts.append(dict(family=fam,world=world,method=method,baseline=alg,
                        support_ratio=es['support_mse_reloaded']/max(ctrl['support_mse_reloaded'],1e-20),
                        iid_delta=ctrl['iid_nrmse']-es['iid_nrmse'],
                        ood_delta=ctrl['ood_nrmse']-es['ood_nrmse'],
                        support_gain_frac=1-es['support_mse_reloaded']/max(ctrl['support_mse_reloaded'],1e-20),
                        es_ood=es['ood_nrmse'],other_ood=ctrl['ood_nrmse']))
    with (out/'contrasts.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=contrasts[0].keys());w.writeheader();w.writerows(contrasts)
    summary={'experiment':lock['experiment'],'models':len(rows),'parents':len(parents),
            'task_code_optimization_problems':len(rows)//3*8,
            'scores':len(rows)*3,
            'elapsed_audit_seconds':time.perf_counter()-t0,
            'population_es':cfg['pop'],'generations_es':cfg['generations'],
            'support_examples_per_task':32,
            'results':{},'paired_es_minus_baseline':{},'environment':cfg['environment']}
    for fam in cfg['family']:
      summary['results'][fam]={}
      for method in cfg['methods']:
       summary['results'][fam][method]={}
       for alg in ('es','random','adam'):
        a=[r for r in world_rows if r['family']==fam and r['method']==method and r['algorithm']==alg]
        summary['results'][fam][method][alg]={
            'worlds':len(a),'support_mse_world_mean':float(np.mean([z['support_mse_reloaded'] for z in a])),
            'iid_nrmse_world_mean':float(np.mean([z['iid_nrmse'] for z in a])),
            'ood_nrmse_world_mean':float(np.mean([z['ood_nrmse'] for z in a])),
            'joint_pass_worlds':sum(z['joint_quality_pass'] for z in a),
            'stored_bytes':a[0]['stored_bytes']}
       for baseline in ('random','adam'):
        q=[z for z in contrasts if z['family']==fam and z['method']==method and z['baseline']==baseline]
        summary['paired_es_minus_baseline'][f'{fam}:{method}:vs_{baseline}']={
            'worlds':len(q),
            'worlds_support_gain_at_least_20pct':sum(z['support_gain_frac']>=.2 for z in q),
            'worlds_iid_gain_at_least_0.03_and_ood_not_worse':sum(z['iid_delta']>=.03 and z['ood_delta']>=-.02 for z in q),
            'mean_support_ratio':float(np.mean([z['support_ratio'] for z in q])),
            'mean_iid_gain':float(np.mean([z['iid_delta'] for z in q])),
            'mean_ood_gain':float(np.mean([z['ood_delta'] for z in q]))}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2))
    print('AUDIT_DONE',len(rows),'models','scores',len(rows)*3,'time',summary['elapsed_audit_seconds'],flush=True)
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    audit(ap.parse_args().out)
