# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A10 held-out audit. Refuses changed model bytes and mismatched world fingerprints."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math
from pathlib import Path
import numpy as np
from a10 import make_world,deserialize,serialize,evaluate


def _mean(items,key):
    return float(np.mean([float(x[key]) for x in items]))


def audit_run(out:Path)->dict:
    out=Path(out)
    if (out/'audit.csv').exists():raise FileExistsError('refusing to overwrite an existing audit')
    receipt=json.loads((out/'train.lock.json').read_text(encoding='utf-8'))
    config=json.loads((out/'config.json').read_text(encoding='utf-8'))
    if receipt['config_sha256']!=hashlib.sha256((out/'config.json').read_bytes()).hexdigest():
        raise RuntimeError('config SHA mismatch')
    rows=receipt['models']
    if len(rows)!=len(config['worlds'])*len(config['seeds'])*len(config['methods'])*len(config['caps']):
        raise RuntimeError('unexpected model count')
    # ALL models are verified and loaded BEFORE any development/held-out scoring.
    loaded=[]
    fingerprints={world:make_world(world)['fingerprint'] for world in config['worlds']}
    for r in rows:
        path=out/r['path']
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=r['model_sha256'] or len(data)!=r['stored_bytes']:
            raise RuntimeError('model hash/size mismatch: '+r['path'])
        if fingerprints[r['world']]!=r['world_fingerprint']:
            raise RuntimeError('world fingerprint mismatch: '+r['path'])
        m=deserialize(data)
        if serialize(m)!=data:raise RuntimeError('model failed full roundtrip: '+r['path'])
        if m.method!=r['method']:raise RuntimeError('method metadata mismatch')
        loaded.append((r,m))
    worlds={world:make_world(world) for world in config['worlds']}
    scored=[]
    for r,m in loaded:
        w=worlds[r['world']]
        a=evaluate(m,w['meta_dev_x'],w['meta_dev_y'],offset=0)
        b=evaluate(m,w['new_iid_x'],w['new_iid_y'],offset=16)
        c=evaluate(m,w['new_ood_x'],w['new_ood_y'],offset=16)
        scored.append({**r,**{f'old_dev_{k}':v for k,v in a.items()},
                              **{f'new_iid_{k}':v for k,v in b.items()},
                              **{f'new_ood_{k}':v for k,v in c.items()}})
    with (out/'audit.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=scored[0].keys());writer.writeheader();writer.writerows(scored)
    grouped=collections.defaultdict(list)
    for row in scored:grouped[(row['world'],row['cap'],row['method'])].append(row)
    world_rows=[]
    for (world,cap,method),group in sorted(grouped.items()):
        rr=dict(world=world,cap=cap,method=method,seeds=len(group),
                stored_bytes=_mean(group,'stored_bytes'),
                meta_seconds=_mean(group,'meta_seconds'),adapt_seconds=_mean(group,'adapt_seconds'))
        for name in ['old_dev_nrmse','new_iid_nrmse','new_ood_nrmse','old_dev_mse','new_iid_mse','new_ood_mse']:
            rr[name]=_mean(group,name)
        rr['passes']=int(rr['old_dev_nrmse']<=.20 and rr['new_iid_nrmse']<=.20 and rr['new_ood_nrmse']<=.20)
        world_rows.append(rr)
    with (out/'worlds.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=world_rows[0].keys());writer.writeheader();writer.writerows(world_rows)
    groups=collections.defaultdict(list)
    for rr in world_rows:groups[(rr['cap'],rr['method'])].append(rr)
    summary=[]
    for (cap,method),gg in sorted(groups.items()):
        summary.append(dict(cap=cap,method=method,n_worlds=len(gg),passes=sum(r['passes'] for r in gg),
            mean_old_dev_nrmse=_mean(gg,'old_dev_nrmse'),mean_new_iid_nrmse=_mean(gg,'new_iid_nrmse'),
            mean_new_ood_nrmse=_mean(gg,'new_ood_nrmse'),median_new_iid_nrmse=float(np.median([r['new_iid_nrmse'] for r in gg])),
            median_new_ood_nrmse=float(np.median([r['new_ood_nrmse'] for r in gg])),
            stored_bytes=_mean(gg,'stored_bytes'),mean_train_seconds=_mean(gg,'meta_seconds'),
            mean_adapt_seconds=_mean(gg,'adapt_seconds')))
    with (out/'summary.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=summary[0].keys());writer.writeheader();writer.writerows(summary)
    # Validation-selected strongest non-Mirror reference using only OLD-TASK development metric.
    paired=[]
    for w in config['worlds']:
        for cap in config['caps']:
            cand=[r for r in world_rows if r['world']==w and r['cap']==cap and r['method'] in ('shift','film','basis','multihead')]
            mir=next((r for r in world_rows if r['world']==w and r['cap']==cap and r['method']=='mirror'),None)
            if not cand or mir is None:continue
            ref=min(cand,key=lambda x:(x['old_dev_nrmse'],x['method']))
            paired.append(dict(world=w,cap=cap,reference=ref['method'],
                mirror_iid=mir['new_iid_nrmse'],ref_iid=ref['new_iid_nrmse'],
                iid_ratio=mir['new_iid_nrmse']/ref['new_iid_nrmse'] if ref['new_iid_nrmse'] else None,
                mirror_ood=mir['new_ood_nrmse'],ref_ood=ref['new_ood_nrmse'],
                mirror_pass=mir['passes'],ref_pass=ref['passes'],
                mirror_bytes=mir['stored_bytes'],ref_bytes=ref['stored_bytes']))
    if paired:
        with (out/'reference_pairs.csv').open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=paired[0].keys());writer.writeheader();writer.writerows(paired)
    ret=dict(n_models=len(loaded),n_scores=len(scored)*3,worlds=world_rows,summary=summary,reference_pairs=paired,
             interpretation='Exploratory. Synthetic ramp-only, same family meta and unseen tasks; no general Mirror-specific capacity claim.')
    (out/'summary.json').write_text(json.dumps(ret,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print('AUDITED',len(loaded),'models',len(scored)*3,'split scores',flush=True)
    return ret

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path)
    a=p.parse_args();audit_run(a.run)
