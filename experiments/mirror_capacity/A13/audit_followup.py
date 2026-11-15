# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Independent read-only A13b post-lock evaluation and mechanism comparison."""
from pathlib import Path
import argparse,csv,hashlib,json
from collections import defaultdict
import numpy as np
import torch
from a13 import make_world,K_TRAIN
from a11 import evaluate
from audit import verify_frozen

ROOT=Path(__file__).resolve().parent

def sha(b):return hashlib.sha256(b).hexdigest()

def audit_followup(main:Path,out:Path):
    torch.set_num_threads(1)
    main=main.resolve();out=out.resolve()
    if (out/'audit.csv').exists():raise FileExistsError('audit already exists')
    cfgraw=(out/'config.json').read_bytes();cfg=json.loads(cfgraw)
    lock=json.loads((out/'train.lock.json').read_bytes())
    if lock['config_sha256']!=sha(cfgraw):raise RuntimeError('A13b config hash mismatch')
    if cfg['parent_main_sha256']!=sha((main/'train.lock.json').read_bytes()):
        raise RuntimeError('main experiment parent ledger changed')
    if cfg['protocol_sha256']!=sha((ROOT/'PROTOCOL_A13B.md').read_bytes()) or cfg['source_sha256']!=sha((ROOT/'followup.py').read_bytes()):
        raise RuntimeError('source/protocol changed')
    if len(lock['models'])!=72:raise RuntimeError('A13b incomplete model set')
    # Check every model digest and roundtrip before *any* split scoring.
    all_models=verify_frozen(out,lock['models'])
    ml=json.loads((main/'train.lock.json').read_bytes())
    parent_meta={r['sha256']:r for r in ml['parents']}
    parents={r['sha256']:m for r,m in verify_frozen(main,ml['parents'])}
    for r,m in all_models:
        if m.method!=r['method'] or r['parent_sha256'] not in parents:raise RuntimeError('orphan/mismatched model')
        parent=parents[r['parent_sha256']]
        for (name,p),(_,q) in zip(m.named_parameters(),parent.named_parameters()):
            if name!='new_codes' and not torch.equal(p,q):raise RuntimeError('frozen shared weights changed')
    print('A13B_ALL_HASHES_PASS',len(all_models),'parents',len(parents),flush=True)
    rows=[]
    for r,m in all_models:
        w=make_world(r['world'],r['family'])
        if w['fingerprint']!=r['world_fingerprint']:raise RuntimeError('world changed')
        d=evaluate(m,w['meta_dev_x'],w['meta_dev_y'],0)
        i=evaluate(m,w['new_iid_x'],w['new_iid_y'],K_TRAIN)
        o=evaluate(m,w['new_ood_x'],w['new_ood_y'],K_TRAIN)
        rows.append({**r,'old_dev_nrmse':d['nrmse'],'iid_nrmse':i['nrmse'],'ood_nrmse':o['nrmse'],
                     'joint_quality_pass':int(max(d['nrmse'],i['nrmse'],o['nrmse'])<=.2)})
    with (out/'audit.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    dd=defaultdict(list)
    for r in rows:dd[(r['family'],r['world'],r['method'],r['algorithm'])].append(r)
    world_rows=[]
    for key,recs in sorted(dd.items()):
        rr=dict(zip(('family','world','method','algorithm'),key))
        if len(recs)!=2:raise RuntimeError('seeds are missing')
        for field in ('stored_bytes','support_mse','old_dev_nrmse','iid_nrmse','ood_nrmse','code_l2_mean'):
            rr[field]=float(np.mean([x[field] for x in recs]))
        rr['joint_quality_pass']=int(max(rr['old_dev_nrmse'],rr['iid_nrmse'],rr['ood_nrmse'])<=.2)
        world_rows.append(rr)
    with (out/'world_means.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=world_rows[0].keys());writer.writeheader();writer.writerows(world_rows)
    summary={'models':len(rows),'world_rows':len(world_rows),'results':{}}
    for fam in ('tent','sine'):
        summary['results'][fam]={}
        for meth in ('mirror','shift','film'):
            summary['results'][fam][meth]={}
            for alg in ('adam4','sobol'):
                a=[x for x in world_rows if (x['family'],x['method'],x['algorithm'])==(fam,meth,alg)]
                summary['results'][fam][meth][alg]={
                    'worlds':len(a),'support_mse':float(np.mean([x['support_mse'] for x in a])),
                    'iid_nrmse':float(np.mean([x['iid_nrmse'] for x in a])),
                    'ood_nrmse':float(np.mean([x['ood_nrmse'] for x in a])),
                    'pass_worlds':sum(x['joint_quality_pass'] for x in a)}
    (out/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2))
    print('A13B_AUDIT_DONE',len(rows),'records',flush=True)
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--main',type=Path,default=ROOT/'main');ap.add_argument('--out',type=Path,default=ROOT/'followup');a=ap.parse_args();audit_followup(a.main,a.out)
