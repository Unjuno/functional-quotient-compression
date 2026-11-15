# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A13d post-lock Mirror-favorable 10x search audit."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import torch
from a13 import make_world,K_TRAIN
from a11 import evaluate
from audit import verify_frozen

ROOT=Path(__file__).resolve().parent

def sha(x):return hashlib.sha256(x).hexdigest()

def audit(out:Path,main:Path,rep:Path):
    out=out.resolve()
    cfgbytes=(out/'config.json').read_bytes();cfg=json.loads(cfgbytes)
    lock=json.loads((out/'train.lock.json').read_bytes())
    if sha(cfgbytes)!=lock['config_sha256'] or len(lock['models'])!=12:raise RuntimeError('incomplete/corrupt config')
    if cfg['protocol_sha256']!=sha((ROOT/'PROTOCOL_A13D.md').read_bytes()) or cfg['script_sha256']!=sha((ROOT/'es10.py').read_bytes()):
        raise RuntimeError('source changed')
    for source in (main.resolve(),rep.resolve()):
        if cfg['source_ledger_hashes'][source.name]!=sha((source/'train.lock.json').read_bytes()):raise RuntimeError('parent ledger changed')
    # Verify all model hashes before any scoring.
    models=verify_frozen(out,lock['models'])
    source_parent={}
    for s in (main,rep):
        ledger=json.loads((s/'train.lock.json').read_bytes())
        for row,m in verify_frozen(s,ledger['parents']):source_parent[row['sha256']]=m
    for row,m in models:
        if row['parent_sha256'] not in source_parent:raise RuntimeError('orphan model')
        parent=source_parent[row['parent_sha256']]
        if parent.method!='mirror' or m.method!='mirror':raise RuntimeError('wrong method')
        for (name,p),(_,q) in zip(m.named_parameters(),parent.named_parameters()):
            if name!='new_codes' and not torch.equal(p,q):raise RuntimeError('frozen weights changed')
    print('A13D_VERIFY_ALL_12_LOCKED',flush=True)
    rows=[]
    for row,m in models:
        world=make_world(row['world'],row['family'])
        if world['fingerprint']!=row['world_fingerprint']:raise RuntimeError('world mismatch')
        old=evaluate(m,world['meta_dev_x'],world['meta_dev_y'],0)['nrmse']
        iid=evaluate(m,world['new_iid_x'],world['new_iid_y'],K_TRAIN)['nrmse']
        ood=evaluate(m,world['new_ood_x'],world['new_ood_y'],K_TRAIN)['nrmse']
        rows.append({**row,'old_dev_nrmse':old,'iid_nrmse':iid,'ood_nrmse':ood,'joint_quality_pass':int(max(old,iid,ood)<=.2)})
    with (out/'audit.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    worlds=[]
    for w in (1200,1201,1202,1300,1301,1302):
        ss=[r for r in rows if r['world']==w]
        if len(ss)!=2:raise RuntimeError('missing seeds')
        row={'family':'tent','world':w,'algorithm':'es10'}
        for field in ('old_dev_nrmse','iid_nrmse','ood_nrmse','support_mse_mean','stored_bytes'):
            row[field]=float(np.mean([r[field] for r in ss]))
        row['joint_quality_pass']=int(max(row['old_dev_nrmse'],row['iid_nrmse'],row['ood_nrmse'])<=.2)
        worlds.append(row)
    with (out/'world_means.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=worlds[0].keys());writer.writeheader();writer.writerows(worlds)
    result={'models':len(rows),'worlds':len(worlds),'worlds_joint_pass':int(sum(r['joint_quality_pass'] for r in worlds)),
            'iid_mean':float(np.mean([r['iid_nrmse'] for r in worlds])),
            'ood_mean':float(np.mean([r['ood_nrmse'] for r in worlds])),
            'support_mean':float(np.mean([r['support_mse_mean'] for r in worlds]))}
    (out/'summary.json').write_text(json.dumps(result,sort_keys=True,indent=2))
    print('A13D_AUDIT_DONE',result,flush=True)

if __name__=='__main__':
    audit(ROOT/'es10',ROOT/'main',ROOT/'replication')
