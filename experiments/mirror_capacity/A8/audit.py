# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Read-only A8 model audit after every FP32 weight file has been locked."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import math
import os
import pandas as pd
import numpy as np
from scipy.stats import t as student_t
import torch
import a8


def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()


def summarize(df:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    required={'world','family','seed','cap','kind','split','mse','nrmse','bytes'}
    if not required.issubset(df.columns): raise ValueError('missing audit columns')
    if df.duplicated(['world','family','seed','cap','kind','split']).any():raise ValueError('duplicate audit grain')
    table=df.pivot_table(index=['world','family','cap','seed','kind','bytes'],columns='split',values=['mse','nrmse'],aggfunc='mean')
    table.columns=[f'{metric}_{split}' for metric,split in table.columns]
    table=table.reset_index()
    for col in ('mse_dev','mse_iid','mse_ood','nrmse_iid','nrmse_ood'):
        if col not in table or table[col].isna().any():raise ValueError(f'missing {col}')
    grouped=table.groupby(['world','family','cap','kind'],as_index=False).agg(
        bytes=('bytes','first'),seed_n=('seed','nunique'),
        val_mse=('mse_dev','mean'),iid_mse=('mse_iid','mean'),ood_mse=('mse_ood','mean'),
        iid_nrmse=('nrmse_iid','mean'),ood_nrmse=('nrmse_ood','mean'))
    grouped['quality_pass']=(grouped.iid_nrmse<=.08)&(grouped.ood_nrmse<=.08)
    rows=[]
    for (w,f,c),g in grouped.groupby(['world','family','cap'],sort=True):
        mirror=g[g.kind=='mirror']
        controls=g[g.kind!='mirror'].sort_values(['val_mse','kind'],kind='mergesort')
        if len(mirror)!=1 or controls.empty:raise ValueError('missing mirror or controls')
        m=mirror.iloc[0];ref=controls.iloc[0]
        rows.append(dict(world=w,family=f,cap=c,selected_control=ref.kind,
                         iid_mirror_mse=float(m.iid_mse),iid_control_mse=float(ref.iid_mse),
                         ood_mirror_mse=float(m.ood_mse),ood_control_mse=float(ref.ood_mse),
                         iid_mse_ratio=float(m.iid_mse/ref.iid_mse),
                         ood_mse_ratio=float(m.ood_mse/ref.ood_mse),
                         iid_mirror_nrmse=float(m.iid_nrmse),iid_control_nrmse=float(ref.iid_nrmse),
                         mirror_bytes=int(m.bytes),control_bytes=int(ref.bytes),
                         mirror_pass=bool(m.quality_pass),control_pass=bool(ref.quality_pass),
                         seed_n=int(m.seed_n)))
    return pd.DataFrame(rows),grouped


def summary_stats(contrast:pd.DataFrame)->list[dict]:
    rows=[]
    for (f,c),g in contrast.groupby(['family','cap']):
        row={'family':f,'cap':int(c),'n_worlds':len(g),
             'mirror_beat_iid':int(sum(g.iid_mse_ratio<1.)),
             'mirror_beat_ood':int(sum(g.ood_mse_ratio<1.)),
             'mirror_joint_win':int(sum((g.iid_mse_ratio<1.)&(g.ood_mse_ratio<1.))),
             'mirror_pass_worlds':int(sum(g.mirror_pass)),
             'control_pass_worlds':int(sum(g.control_pass))}
        for suffix in ('iid','ood'):
            values=np.log(g[f'{suffix}_mse_ratio'].to_numpy(dtype=float))
            n=len(values)
            mean=float(values.mean())
            half=float(student_t.ppf(.975,n-1)*values.std(ddof=1)/np.sqrt(n)) if n>1 else float('nan')
            row[f'{suffix}_geomean_ratio']=float(np.exp(mean))
            row[f'{suffix}_95ci_lo']=float(np.exp(mean-half))
            row[f'{suffix}_95ci_hi']=float(np.exp(mean+half))
        rows.append(row)
    return rows


def audit(out:Path)->dict:
    torch.set_num_threads(1)
    lockpath=out/'train.lock.json'
    if not lockpath.is_file(): raise FileNotFoundError('training lock is required before audit')
    if (out/'audit.csv').exists():raise FileExistsError('audit is single use; refuses overwrite')
    lock=json.loads(lockpath.read_text())
    if lock['format']!='a8-lock-v1' or sha((out/'train.csv').read_bytes())!=lock['train_csv_sha256']:
        raise ValueError('train manifest invalid')
    rows=list(csv.DictReader((out/'train.csv').open()))
    if len(rows)!=lock['n'] or len(lock['files'])!=len(rows):raise ValueError('lock count mismatch')
    names=[r['weight_file'] for r in rows]
    if len(names)!=len(set(names)): raise ValueError('duplicate weight file')
    if {p.name for p in (out/'models').iterdir()}!=set(names):raise ValueError('extra/missing weight files')
    records=[]; ranks=[]
    envcache={}
    for r in rows:
        name=r['weight_file'];raw=(out/'models'/name).read_bytes();meta=lock['files'][name]
        if len(raw)!=int(meta['bytes']) or sha(raw)!=meta['sha256'] or sha(raw)!=r['sha256']:
            raise ValueError('file digest/length mismatch')
        model=a8.deserialize(raw)
        if a8.serialize(model)!=raw:raise ValueError('roundtrip mismatch')
        if model.kind!=r['kind'] or model.width!=int(r['width']):raise ValueError('wrong model kind/width')
        group=(int(r['world']),r['family'])
        if group not in envcache:envcache[group]=a8.make_world(*group)
        env=envcache[group]
        if env['fingerprint']!=meta['dataset_sha256']:raise ValueError('dataset changed')
        for split in ('dev','iid','ood'):
            q=a8.quality(model,env[split])
            records.append({k:r[k] for k in ('world','family','seed','cap','kind','width','bytes','weight_file','sha256')}
                           |dict(split=split,**q))
    df=pd.DataFrame(records)
    for col in ['world','seed','cap','width','bytes']:
        df[col]=df[col].astype(int)
    df=df.sort_values(['family','world','cap','kind','seed','split']).reset_index(drop=True)
    contrast,frontier=summarize(df)
    for (w,f),env in envcache.items():
        for c in lock['config']['caps']:
            h=a8.create('multihead',c).width
            cert=a8.rank_certificate(env,h)
            ranks.append({'world':w,'family':f,'cap':c,'head_width':h,
                          'target_rank_centered':cert['target_rank_centered'],
                          'rank_h_mse_floor_probe':cert['mse_floor'],
                          'rank_h_relative_frobenius_floor_probe':cert['relative_frobenius_floor'],
                          'smallest_singular_value':float(cert['singular_values'][-1])})
    # Write score files ONLY after full validation and all scores computed.
    df.to_csv(out/'audit.csv',index=False)
    contrast.to_csv(out/'contrasts.csv',index=False)
    frontier.to_csv(out/'frontier.csv',index=False)
    pd.DataFrame(ranks).sort_values(['family','world','cap']).to_csv(out/'rank.csv',index=False)
    aggregate=summary_stats(contrast)
    (out/'summary.json').write_text(json.dumps({'status':'exploratory A8','models':len(rows),
        'audit_scores':len(df),'summaries':aggregate,'bounds_note':'probe-only, not held-out empirical MSE',
        'selection':'non-Mirror comparison selected by mean dev MSE across two seeds',
        'quality':'IID and OOD NRMSE <=.08; missing quality frontiers censored'},indent=2)+'\n')
    names2=('audit.csv','contrasts.csv','frontier.csv','rank.csv','summary.json')
    auditlock={'schema':'a8-audit-v1','train_lock_sha256':sha(lockpath.read_bytes()),
               'file_sha256':{name:sha((out/name).read_bytes()) for name in names2}}
    (out/'audit.lock.json').write_text(json.dumps(auditlock,indent=2)+'\n')
    return {'models':len(rows),'scores':len(df),'summaries':aggregate}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path)
    r=audit(p.parse_args().out)
    print('A8_AUDIT_LOCKED',r['models'],r['scores'])
    for s in r['summaries']:print(s)
