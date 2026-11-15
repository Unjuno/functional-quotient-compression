# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Fresh-world A13c evidence audit. No optimizer or checkpoint selection."""
from __future__ import annotations
from pathlib import Path
import argparse,csv,json,hashlib
from collections import defaultdict
import numpy as np
import torch
from audit import verify_frozen
from a13 import make_world,deserialize,K_TRAIN,A11_DIR
from a11 import evaluate

ROOT=Path(__file__).resolve().parent

def sha(b):return hashlib.sha256(b).hexdigest()

def audit(out:Path):
    torch.set_num_threads(1)
    out=out.resolve()
    cfgbytes=(out/'config.json').read_bytes();cfg=json.loads(cfgbytes)
    lock=json.loads((out/'train.lock.json').read_bytes());meta=json.loads((out/'meta.lock.json').read_bytes())
    if lock['config_sha256']!=sha(cfgbytes) or meta['config_sha256']!=sha(cfgbytes) or lock['meta_lock_sha256']!=sha((out/'meta.lock.json').read_bytes()):
        raise RuntimeError('meta/final/config lock mismatch')
    if cfg['a11_source_sha256']!=sha((A11_DIR/'a11.py').read_bytes()):raise RuntimeError('A11 changed')
    if cfg['a12_source_sha256']!=sha(((ROOT.parent/'A12'/'a12.py') if (ROOT.parent/'A12'/'a12.py').is_file() else (ROOT/'source'/'repo'/'experiments'/'mirror_capacity'/'A12'/'a12.py')).read_bytes()):raise RuntimeError('A12 changed')
    for name,value in cfg['source_sha256'].items():
        if value!=sha((ROOT/name).read_bytes()):raise RuntimeError('source altered: '+name)
    if (out/'audit.csv').exists():raise FileExistsError('audit exists')
    if len(meta['parents'])!=36 or len(lock['parents'])!=36 or len(lock['models'])!=108:raise RuntimeError('incomplete evidence set')
    parents=verify_frozen(out,meta['parents']);final=verify_frozen(out,lock['models'])
    orig={r['sha256']:m for r,m in parents}
    for row,m in final:
        if row['parent_sha256'] not in orig or row['method']!=m.method:raise RuntimeError('parent/method mismatch')
        for (name,p),(_,q) in zip(m.named_parameters(),orig[row['parent_sha256']].named_parameters()):
            if name!='new_codes' and not torch.equal(p,q):raise RuntimeError('frozen shared changed')
    print('A13C_ALL_FROZEN',len(parents),'parents',len(final),'new adapted models',flush=True)
    worlds={(f,w):make_world(w,f) for f in cfg['families'] for w in cfg['worlds']}
    scores=[]
    for row,m in final:
        w=worlds[(row['family'],row['world'])]
        if w['fingerprint']!=row['world_fingerprint']:raise RuntimeError('world changed')
        old=evaluate(m,w['meta_dev_x'],w['meta_dev_y'],0)['nrmse']
        iid=evaluate(m,w['new_iid_x'],w['new_iid_y'],K_TRAIN)['nrmse']
        ood=evaluate(m,w['new_ood_x'],w['new_ood_y'],K_TRAIN)['nrmse']
        scores.append({**row,'old_dev_nrmse':old,'iid_nrmse':iid,'ood_nrmse':ood,
                       'joint_quality_pass':int(max(old,iid,ood)<=.20)})
    with (out/'audit.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=scores[0].keys());writer.writeheader();writer.writerows(scores)
    grp=defaultdict(list)
    for row in scores:grp[(row['family'],row['world'],row['method'],row['algorithm'])].append(row)
    world_rows=[]
    for key,v in sorted(grp.items()):
        row=dict(zip(('family','world','method','algorithm'),key))
        if len(v)!=2:raise RuntimeError('missing seeds')
        for k in ('stored_bytes','support_mse_mean','old_dev_nrmse','iid_nrmse','ood_nrmse'):
            row[k]=float(np.mean([r[k] for r in v]))
        row['joint_quality_pass']=int(max(row['old_dev_nrmse'],row['iid_nrmse'],row['ood_nrmse'])<=.20)
        world_rows.append(row)
    with (out/'world_means.csv').open('x',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=world_rows[0].keys());wr.writeheader();wr.writerows(world_rows)
    data={}
    for fam in cfg['families']:
        data[fam]={}
        for meth in cfg['methods']:
            data[fam][meth]={}
            for alg in ('es','random','adam'):
                v=[r for r in world_rows if (r['family'],r['method'],r['algorithm'])==(fam,meth,alg)]
                data[fam][meth][alg]={
                    'worlds':len(v),'support_mse':float(np.mean([r['support_mse_mean'] for r in v])),
                    'iid_nrmse':float(np.mean([r['iid_nrmse'] for r in v])),
                    'ood_nrmse':float(np.mean([r['ood_nrmse'] for r in v])),
                    'joint_pass_worlds':sum(r['joint_quality_pass'] for r in v)}
    result=dict(experiment=cfg['experiment'],parents=len(parents),models=len(scores),split_scores=len(scores)*3,results=data)
    (out/'summary.json').write_text(json.dumps(result,sort_keys=True,indent=2))
    print('A13C_AUDIT_DONE',len(scores),'models',len(scores)*3,'scores',flush=True)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'replication');a=ap.parse_args();audit(a.out)
