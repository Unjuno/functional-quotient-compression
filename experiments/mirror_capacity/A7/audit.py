# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Verify fully frozen A7 model files then evaluate val/IID/OOD; no audit fitting."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from functools import lru_cache
import statistics

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import torch
import a7
a6=a7.source_core()

HERE=Path(__file__).resolve().parent


def sha(b:bytes):return hashlib.sha256(b).hexdigest()


@lru_cache(maxsize=100)
def world_cache(world:int,family:str):return a7.make_world(world,family)


def verify_and_score(folder:Path)->pd.DataFrame:
    lock=json.loads((folder/'train.lock.json').read_text())
    cfg=lock['created_from_config']
    if (sha((HERE/'PROTOCOL.md').read_bytes())!=cfg['model_protocol_sha256'] or
        sha((HERE/'a7.py').read_bytes())!=cfg['implementation_sha256'] or
        sha((HERE.parent/'A6'/'a6.py').read_bytes())!=cfg['a6_core_sha256']):
        raise ValueError('code or protocol changed after training')
    if sha((folder/'train.csv').read_bytes())!=lock['train_csv_sha256']:
        raise ValueError('train.csv changed after locking')
    trained=list(csv.DictReader((folder/'train.csv').open()))
    if len(trained)!=lock['model_count'] or len(trained)!=len(lock['rows']):
        raise ValueError('missing training checkpoints')
    scores=[]
    for row,record in zip(trained,lock['rows'],strict=True):
        if any(str(row[k])!=str(record[k]) for k in ('weight_file','digest','bytes','dataset_fingerprint')):
            raise ValueError('lock row mismatch')
        raw=(folder/'models'/row['weight_file']).read_bytes()
        if len(raw)!=int(row['bytes']) or sha(raw)!=row['digest'] or len(raw)>int(row['cap']):
            raise ValueError('weight file tampered with or over cap')
        model=a6.deserialize(raw).eval()
        if a6.serialize(model)!=raw or model.width!=int(row['width']) or model.kind!=row['kind']:
            raise ValueError('weight replay mismatch')
        data=world_cache(int(row['world']),row['family'])
        if data['fingerprint']!=row['dataset_fingerprint']:
            raise ValueError('dataset fingerprint changed')
        for split in ('val','iid','ood'):
            score=a6.eval_model(model,data[split])
            scores.append({'world':int(row['world']),'family':row['family'],'seed':int(row['seed']),
                           'cap':int(row['cap']),'kind':row['kind'],'split':split,
                           'bytes':int(row['bytes']),'width':model.width,**score})
    return pd.DataFrame(scores)


def summarize(sc:pd.DataFrame,allow_single_world:bool=False)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,dict]:
    # Average seeds within each world BEFORE choosing the validation-only competitor.
    if not allow_single_world and sc.world.nunique()<2:
        raise ValueError('need >=2 independent mapping worlds for interval estimation')
    paired=sc.groupby(['world','family','cap','kind','split'],as_index=False)[['bce','balanced_accuracy','accuracy','worst_task_bce','bytes']].mean()
    nonmirror=[k for k in a6.KINDS if k not in ('mirror4','pre4')]
    selected=[]
    for (w,f,c),gg in paired[paired.split=='val'].groupby(['world','family','cap']):
        cand=gg[gg.kind.isin(nonmirror)].sort_values(['bce','kind'])
        selected.append({'world':w,'family':f,'cap':c,'best_control':cand.iloc[0]['kind'],
                         'selected_val_bce':float(cand.iloc[0]['bce'])})
    selection=pd.DataFrame(selected)
    contrasts=[]
    for choice in selected:
        w,f,c,k=choice['world'],choice['family'],choice['cap'],choice['best_control']
        sel=paired[(paired.world==w)&(paired.family==f)&(paired.cap==c)]
        rec={'world':w,'family':f,'cap':c,'selected_control':k}
        for split in ('iid','ood'):
            a=sel[(sel.kind=='mirror4')&(sel.split==split)].iloc[0]
            b=sel[(sel.kind==k)&(sel.split==split)].iloc[0]
            rec[f'{split}_mirror_bce']=float(a.bce)
            rec[f'{split}_control_bce']=float(b.bce)
            rec[f'{split}_bce_ratio']=float(a.bce/b.bce)
            rec[f'{split}_mirror_balacc']=float(a.balanced_accuracy)
            rec[f'{split}_control_balacc']=float(b.balanced_accuracy)
        contrasts.append(rec)
    contrasts=pd.DataFrame(contrasts)
    # Independent, predeclared secondary comparison: inverse-removed shear.
    for i,rec in contrasts.iterrows():
        sel=paired[(paired.world==rec['world'])&(paired.family==rec['family'])&(paired.cap==rec['cap'])]
        for split in ('iid','ood'):
            a=sel[(sel.kind=='pre4')&(sel.split==split)]
            if len(a)==1:
                contrasts.loc[i,f'{split}_pre4_bce']=float(a.iloc[0].bce)
                contrasts.loc[i,f'{split}_pre4_bce_ratio']=float(a.iloc[0].bce/rec[f'{split}_control_bce'])
    pass_rows=[]
    for (w,f,k),g in paired.groupby(['world','family','kind']):
        possible=[]
        for cap in sorted(g.cap.unique()):
            iid=g[(g.cap==cap)&(g.split=='iid')].iloc[0]
            ood=g[(g.cap==cap)&(g.split=='ood')].iloc[0]
            if iid.balanced_accuracy>=0.80 and ood.balanced_accuracy>=0.70:
                possible.append((float(iid.bytes),int(cap)))
        best=min(possible) if possible else (float('nan'),float('nan'))
        pass_rows.append({'world':w,'family':f,'kind':k,'min_passing_bytes':best[0],
                          'min_passing_cap':best[1],'censored':len(possible)==0})
    frontier=pd.DataFrame(pass_rows)
    aggregates=[]
    for (fam,cap),g in contrasts.groupby(['family','cap']):
        rows={'family':fam,'cap':int(cap),'worlds':len(g)}
        for split in ('iid','ood'):
            logr=np.log(g[f'{split}_bce_ratio'].to_numpy())
            sd=float(np.std(logr,ddof=1)) if len(logr)>1 else float('nan')
            sem=sd/math.sqrt(len(logr))
            # Four family/cap evaluations: Bonferroni per-contrast 98.75% interval.
            quant=(float(student_t.ppf(1-0.05/(2*4),len(logr)-1)) if len(logr)>1 else float('nan'))
            rows[f'{split}_geomean_ratio']=float(np.exp(logr.mean()))
            rows[f'{split}_ratio_ci_lo']=float(np.exp(logr.mean()-quant*sem))
            rows[f'{split}_ratio_ci_hi']=float(np.exp(logr.mean()+quant*sem))
            rows[f'{split}_mirror_win_worlds']=int((g[f'{split}_bce_ratio']<1).sum())
            rows[f'{split}_uc_logratio']=sem
            rows[f'{split}_k_t']=quant
        rows['discovery_signal']=rows['iid_mirror_win_worlds']>=5 and rows['ood_mirror_win_worlds']>=5
        wins=0; censored=0
        for w in g.world.unique():
            choice=frontier[(frontier.world==w)&(frontier.family==fam)]
            mir=choice[choice.kind=='mirror4'].iloc[0]
            ctrl=choice[choice.kind.isin(nonmirror)]
            costs=ctrl.min_passing_bytes.dropna()
            if len(costs)==0 or bool(mir.censored):
                censored+=1
            elif float(mir.min_passing_bytes)<=0.9*float(costs.min()):
                wins+=1
        rows['practical_wins_worlds']=wins
        rows['censored_worlds']=censored
        rows['practical_pass']=wins>=5
        aggregates.append(rows)
    means=paired.groupby(['family','cap','kind','split'],as_index=False)[['bce','balanced_accuracy','bytes']].agg('mean')
    summary={'experiment':'A7 spatial-composition-conditional-capacity','model_count':int(len(sc)/3),
             'score_rows':int(len(sc)),'winner_rule':'validation-only selecting non-Mirror competitor at the world/family/cap level; seed-mean',
             'aggregate':aggregates,'pass_definition':{'iid_macro_balanced_accuracy':0.80,'ood_macro_balanced_accuracy':0.70,
                 'rate_bytes_ratio_max':0.90,'worlds_required':5},
             'scope':'16 known task IDs, 1797 shared digit images, pairwise spatial AND/XOR, no unseen task, no matched FLOPs'}
    return contrasts,frontier,means,summary


def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    torch.set_num_threads(1)
    folder=args.out
    dest=folder/'audit'
    if dest.exists():raise FileExistsError('audit already exists; refusing overwrite')
    if not (folder/'train.lock.json').exists():raise ValueError('no completed train lock')
    sc=verify_and_score(folder)
    contrasts,frontier,means,summary=summarize(sc)
    dest.mkdir()
    for name,df in [('scores.csv',sc),('world_contrasts.csv',contrasts),('capacity_frontier.csv',frontier),('model_means.csv',means)]:
        df.to_csv(dest/name,index=False,float_format='%.10g')
    (dest/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    (dest/'audit.sha256.json').write_text(json.dumps({name:sha((dest/name).read_bytes()) for name in
        ('scores.csv','world_contrasts.csv','capacity_frontier.csv','model_means.csv','summary.json')},indent=2)+'\n')
    print('AUDIT_OK',summary['model_count'],'models',summary['score_rows'],'score rows')
    print(json.dumps(aggregates:=summary['aggregate'],indent=2))

if __name__=='__main__':main()
