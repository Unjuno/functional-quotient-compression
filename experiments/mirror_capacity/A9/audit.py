# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A9 post-lock audit; all weights verified before reading any dev/IID/OOD metrics."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
import torch
from experiments.mirror_capacity.A8 import a8
from experiments.mirror_capacity.A9 import a9

def sha(path:Path):return hashlib.sha256(path.read_bytes()).hexdigest()

def write_csv(path:Path,rows:list[dict]):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def audit(out:Path):
    lock=json.loads((out/'train.lock.json').read_text())
    cfg=lock['config']
    if sha(out/'train.csv')!=lock['train_csv_sha256']:raise ValueError('train table changed')
    rows=list(csv.DictReader((out/'train.csv').open()))
    if len(rows)!=lock['num_models'] or len(rows)!=len(lock['models']):raise ValueError('missing runs')
    for r in rows:
        path=out/'models'/r['model_file']
        raw=path.read_bytes()
        item=lock['models'][r['model_file']]
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('locked model corrupted: '+str(path))
        if a8.serialize(a8.deserialize(raw))!=raw:
            raise ValueError('bad round-trip: '+str(path))
    print('A9_HASH_AUDIT_PASS',len(rows),flush=True)
    scored=[]
    worlds={}
    for r in rows:
        wid=int(r['world'])
        data=worlds.setdefault(wid,a8.make_world(wid,'ramp'))
        if data['fingerprint']!=r['dataset_sha256']:raise ValueError('dataset mismatch')
        m=a8.deserialize((out/'models'/r['model_file']).read_bytes())
        for split in ('train','dev','iid','ood'):
            result=a8.quality(m,data[split])
            scored.append({'world':wid,'seed':int(r['seed']),'kind':r['kind'],
                           'regime':r['regime'],'width':int(r['width']),
                           'serialized_bytes':int(r['serialized_bytes']),
                           'split':split,**result})
    write_csv(out/'audit.csv',scored)
    grouped=defaultdict(lambda:defaultdict(list))
    for r in scored:
        grouped[(r['world'],r['kind'],r['regime'],r['split'])]['mse'].append(r['mse'])
        grouped[(r['world'],r['kind'],r['regime'],r['split'])]['nrmse'].append(r['nrmse'])
    means=[]
    for (wid,kind,regime,split),v in sorted(grouped.items()):
        means.append({'world':wid,'kind':kind,'regime':regime,'split':split,
                      'mean_mse':float(np.mean(v['mse'])),'mean_nrmse':float(np.mean(v['nrmse'])),
                      'n_seeds':len(v['mse'])})
    write_csv(out/'world_means.csv',means)
    lookup={(r['world'],r['kind'],r['regime'],r['split']):r for r in means}
    rows_h=[]
    for wid in cfg['worlds']:
        for kind in cfg['kinds']:
            try:
                s=lookup[(wid,kind,'scaffold_code4','iid')]
                b=lookup[(wid,kind,'random_full4','iid')]
                ood=lookup[(wid,kind,'scaffold_code4','ood')]
            except KeyError:continue
            rows_h.append({'world':wid,'kind':kind,
                           'scaffold_iid_nrmse':s['mean_nrmse'],
                           'scaffold_ood_nrmse':ood['mean_nrmse'],
                           'scaffold_joint_pass':int(max(s['mean_nrmse'],ood['mean_nrmse'])<=0.08),
                           'iid_mse_ratio_vs_random_full4':s['mean_mse']/b['mean_mse'],
                           'scaffold_iid_lower_mse':int(s['mean_mse']<b['mean_mse'])})
    write_csv(out/'hypothesis_worlds.csv',rows_h)
    methods=[]
    for kind in cfg['kinds']:
        for regime in cfg['regimes']:
            iid=[r['mean_nrmse'] for r in means if r['kind']==kind and r['regime']==regime and r['split']=='iid']
            ood=[r['mean_nrmse'] for r in means if r['kind']==kind and r['regime']==regime and r['split']=='ood']
            if not iid:continue
            p=sum(max(i,o)<=0.08 for i,o in zip(iid,ood))
            methods.append({'kind':kind,'regime':regime,'median_world_iid_nrmse':float(np.median(iid)),
                            'median_world_ood_nrmse':float(np.median(ood)),
                            'joint_pass_worlds':int(p),'n_worlds':len(iid)})
    write_csv(out/'method_summary.csv',methods)
    logratios=np.array([math.log(r['iid_mse_ratio_vs_random_full4']) for r in rows_h if r['kind']=='mirror'])
    if len(logratios)>1:
        mean=float(logratios.mean());uc=float(logratios.std(ddof=1)/np.sqrt(len(logratios)))
        tcrit=float(student_t.ppf(.975,df=len(logratios)-1))
        ci=[float(math.exp(mean-tcrit*uc)),float(math.exp(mean+tcrit*uc))]
    else:mean=uc=float('nan');ci=[None,None]
    h1=[r for r in rows_h if r['kind']=='mirror']
    summary={'num_models':len(rows),'num_scores':len(scored),'num_worlds':len(cfg['worlds']),
             'primary':{'mirror_scaffold_code4_joint_pass_worlds':sum(r['scaffold_joint_pass'] for r in h1),
                        'mirror_scaffold_code4_iid_better_worlds':sum(r['scaffold_iid_lower_mse'] for r in h1),
                        'H1_PASS':sum(r['scaffold_joint_pass'] for r in h1)>=5 and sum(r['scaffold_iid_lower_mse'] for r in h1)>=5,
                        'mirror_codeonly_to_random_full4_geomean_iid_MSE_ratio':float(math.exp(mean)),
                        'logratio_combined_standard_uncertainty':uc,'ratio_95pct_t_CI':ci},
             'method_summary':methods,
             'caveats':['family-aware scaffold with no teacher thresholds; not a generic model comparison',
                        'same random minibatch stream but not matched FLOPs',
                        'only six synthetic worlds, two seeds and one optimizer; 95% interval descriptive']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    (out/'audit.lock.json').write_text(json.dumps({
        'format':'a9-audit-lock-v1','train_lock_sha256':sha(out/'train.lock.json'),
        'generated':{p.name:sha(p) for p in (out/'audit.csv',out/'world_means.csv',
                         out/'hypothesis_worlds.csv',out/'method_summary.csv',out/'summary.json')}},indent=2)+'\n')
    print('A9_AUDIT_DONE',len(scored),'H1_PASS',summary['primary']['H1_PASS'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();audit(args.out)
