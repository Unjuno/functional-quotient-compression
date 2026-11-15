# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A14b post-result challenge data, never used by A14 oracle selection.

Reconstructs the published synthetic tent generator from its pinned seed and
scores on fresh independent input draws. This is a posthoc sensitivity check,
not an independently preregistered experiment.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from a14 import load_parent_and_es,deserialize,evaluate,check_frozen_parent,sha,make_world,K_NEW


def independent_challenge(world_id:int,n=512):
    tag=1  # A11 FAMILIES=('ramp','tent','sine')
    rng=np.random.default_rng(417211+world_id*19937+tag*170981)
    center=rng.uniform(-.65,.65,24).astype('float32')
    width=rng.uniform(.20,.46,24).astype('float32')
    def fy(x):
        c=center[None,:];w=width[None,:]
        return (np.maximum(x-c+w,0)-2*np.maximum(x-c,0)+np.maximum(x-c-w,0))/w
    # Independent of A11's stream IDs 11/23/37/59 and A13 support streams.
    a=np.random.default_rng(88011004+world_id*103).uniform(-1.5,1.5,(n,1)).astype('float32')
    b=np.random.default_rng(88011017+world_id*103).uniform(-1.9,1.9,(n,1)).astype('float32')
    return torch.from_numpy(a),torch.from_numpy(fy(a)[:,16:].copy()),torch.from_numpy(b),torch.from_numpy(fy(b)[:,16:].copy()),center[16:],width[16:],fy


def audit_challenge(main:Path,base:Path):
    torch.set_num_threads(1)
    config=json.loads((main/'config.json').read_text())
    lock=json.loads((main/'train.lock.json').read_text())
    rows=[]
    for rec in lock['models']:
        parent,es_model,world,parentrec,esrec=load_parent_and_es(base,rec['family'],rec['world'],rec['seed'],rec['method'])
        raw=(main/rec['path']).read_bytes()
        if sha(raw)!=rec['sha256']:raise RuntimeError('oracle checkpoint corrupted')
        oracle=deserialize(raw)
        check_frozen_parent(parent,oracle,parentrec['sha256'])
        x,yy,xx,zz,center,width,fy=independent_challenge(rec['world'])
        # Check synthetic target compatibility on original inputs before interpreting results.
        a=fy(world['new_iid_x'].numpy())[:,16:]
        if not np.allclose(a,world['new_iid_y'].numpy(),atol=2e-6,rtol=1e-6):
            raise RuntimeError('teacher generator mismatch')
        for name,model in (('support_es',es_model),('privileged_oracle',oracle)):
            i=evaluate(model,x,yy,16)
            o=evaluate(model,xx,zz,16)
            old=evaluate(model,world['meta_dev_x'],world['meta_dev_y'],0)['nrmse']
            rows.append(dict(world=rec['world'],seed=rec['seed'],method=rec['method'],algorithm=name,
                challenge_iid_nrmse=i['nrmse'],challenge_ood_nrmse=o['nrmse'],
                old_dev_nrmse=old,
                challenge_joint_pass=int(max(i['nrmse'],o['nrmse'],old)<=.2)))
    df=pd.DataFrame(rows)
    df.to_csv(main/'posthoc_challenge.csv',index=False)
    grp=df.groupby(['world','method','algorithm'],as_index=False).agg(iid=('challenge_iid_nrmse','mean'),ood=('challenge_ood_nrmse','mean'),old=('old_dev_nrmse','mean'))
    grp['pass']=((grp[['iid','ood','old']].max(axis=1))<=.2).astype(int)
    grp.to_csv(main/'posthoc_challenge_worlds.csv',index=False)
    summary=[]
    for (method,algo),z in grp.groupby(['method','algorithm']):
        summary.append(dict(method=method,algorithm=algo,worlds=len(z),pass_worlds=int(z['pass'].sum()),iid_mean=float(z.iid.mean()),ood_mean=float(z.ood.mean())))
    (main/'posthoc_challenge_summary.json').write_text(json.dumps({'warning':'POSTHOC: oracle was selected on DIFFERENT A13 IID/OOD test labels; challenge labels never used in code selection; still privileged evaluation, not a deployable comparison','summary':summary},indent=2))
    print('A14_CHALLENGE_DONE',len(rows),'model-split rows',flush=True)
    for x in summary: print(x,flush=True)

if __name__=='__main__':
    audit_challenge(HERE/'results',HERE.parent/'mirror_es_a13_20261115')
