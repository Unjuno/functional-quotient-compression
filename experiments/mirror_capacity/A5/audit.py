#!/usr/bin/env python3
"""A5 held-out audit: locked files only; no optimization and no test-driven selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats as st
import torch

from a5 import KINDS, deserialize, serialize, make_world, eval_model

ROOT = Path(__file__).resolve().parent
CONTROLS = tuple(x for x in KINDS if x != 'mirror4')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock_check(out: Path):
    lock = json.loads((out / 'train.lock.json').read_text())
    protocol = json.loads((out / 'protocol.json').read_text())
    if sha(out / 'protocol.json') != lock['protocol_sha256'] or sha(out / 'train.csv') != lock['train_sha256']:
        raise RuntimeError('metadata changed since training')
    for p, digest in protocol['source_sha256'].items():
        if sha(ROOT / p) != digest:
            raise RuntimeError('source changed since training: ' + p)
    for p, digest in lock['models'].items():
        if sha(out / 'models' / p) != digest:
            raise RuntimeError('locked weight hash changed: ' + p)
    if len(lock['models']) != protocol['expected_models']:
        raise RuntimeError('missing model')
    return protocol, lock


def output_csv(path, rows):
    if not rows:
        raise RuntimeError('cannot write empty results')
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    protocol, lock = lock_check(args.out)
    with (args.out / 'train.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    all_scores = []
    for record in rows:
        family, world = record['family'], int(record['world'])
        data = make_world(world, family)
        raw = (args.out / 'models' / record['artifact']).read_bytes()
        model = deserialize(raw).eval()
        if serialize(model) != raw or len(raw) != int(record['bytes']):
            raise RuntimeError('weight byte roundtrip differs: ' + record['artifact'])
        iid = eval_model(model, data['iid'])
        ood = eval_model(model, data['ood'])
        all_scores.append(dict(family=family, world=world, seed=int(record['seed']),
                              cap=int(record['cap']), kind=record['kind'], width=int(record['width']),
                              bytes=len(raw), val_nrmse=float(record['val_nrmse']),
                              iid_nrmse=iid['nrmse'], iid_worst=iid['worst_task_nrmse'],
                              ood_nrmse=ood['nrmse'], ood_worst=ood['worst_task_nrmse'],
                              sha256=record['sha256']))
    dest = args.out / 'audit'
    dest.mkdir(exist_ok=False)
    output_csv(dest / 'scores.csv', all_scores)
    groups = defaultdict(list)
    for score in all_scores:
        groups[score['family'], score['world'], score['cap'], score['kind']].append(score)
    means = []
    for (family, world, cap, kind), subset in sorted(groups.items()):
        if len(subset) != len(protocol['seeds']):
            raise RuntimeError('incorrect per-group seed coverage')
        means.append(dict(family=family, world=world, cap=cap, kind=kind,
                          bytes=subset[0]['bytes'], width=subset[0]['width'],
                          val_nrmse=float(np.mean([r['val_nrmse'] for r in subset])),
                          iid_nrmse=float(np.mean([r['iid_nrmse'] for r in subset])),
                          ood_nrmse=float(np.mean([r['ood_nrmse'] for r in subset])),
                          iid_worst=float(np.mean([r['iid_worst'] for r in subset]))))
    output_csv(dest / 'world_means.csv', means)
    lookup = {(r['family'], r['world'], r['cap'], r['kind']):r for r in means}
    contrasts = []
    for family in protocol['families']:
        for cap in protocol['caps']:
            for world in protocol['worlds']:
                mir=lookup[family, world, cap, 'mirror4']
                available=[lookup[family,world,cap,c] for c in CONTROLS]
                ctrl=min(available, key=lambda r:r['val_nrmse'])  # only development used to select
                oracle=min(available, key=lambda r:r['iid_nrmse'])  # conservative reporting floor, NOT selected model
                contrasts.append(dict(family=family,cap=cap,world=world,
                                      control_selected=ctrl['kind'],mirror_iid=mir['iid_nrmse'],
                                      control_iid=ctrl['iid_nrmse'],mirror_ood=mir['ood_nrmse'],
                                      control_ood=ctrl['ood_nrmse'],
                                      iid_ratio=mir['iid_nrmse']/ctrl['iid_nrmse'],
                                      ood_ratio=mir['ood_nrmse']/ctrl['ood_nrmse'],
                                      oracle_iid_ratio=mir['iid_nrmse']/oracle['iid_nrmse']))
    output_csv(dest/'contrasts.csv', contrasts)
    summaries=[]
    for family in protocol['families']:
        for cap in protocol['caps']:
            rr=[r for r in contrasts if r['family']==family and r['cap']==cap]
            x=np.log(np.array([r['iid_ratio'] for r in rr],float))
            u=float(x.std(ddof=1)/math.sqrt(len(x))) if len(x)>1 else float('nan')
            k=float(st.t.ppf(.975,len(x)-1)) if len(x)>1 else float('nan')
            median={kind:float(np.median([lookup[family,w,cap,kind]['iid_nrmse'] for w in protocol['worlds']])) for kind in KINDS}
            summaries.append(dict(family=family,cap=cap,worlds=len(rr),
                                  geometric_mean_ratio=float(np.exp(x.mean())),
                                  ci95_low=float(np.exp(x.mean()-k*u)),
                                  ci95_high=float(np.exp(x.mean()+k*u)),
                                  log_ratio_u_c=u,k_t=k,
                                  iid_wins=sum(r['iid_ratio']<1 for r in rr),
                                  ood_wins=sum(r['ood_ratio']<=1 for r in rr),
                                  all_control_model_medians=median))
    # Tested-rate quality status: report censored worlds; never substitute 3000B as "minimum" if none passes.
    rate=[]
    for family in protocol['families']:
        for world in protocol['worlds']:
            def minimum_for(kind):
                fits=[lookup[family,world,cap,kind]['bytes'] for cap in protocol['caps']
                      if lookup[family,world,cap,kind]['iid_nrmse']<=0.40]
                return min(fits) if fits else None
            m=minimum_for('mirror4')
            controls={kind:minimum_for(kind) for kind in CONTROLS}
            valid=[size for size in controls.values() if size is not None]
            best=min(valid) if valid else None
            rate.append(dict(family=family,world=world,mirror_min_passing_bytes=m,
                             best_control_min_passing_bytes=best,
                             mirror_better_by_10pct=(m is not None and best is not None and m<=0.9*best),
                             censor=('none_pass' if m is None and best is None else 'mirror_censored' if m is None else 'control_censored' if best is None else 'none')))
    output_csv(dest/'quality_frontier.csv',rate)
    summary=dict(provenance=dict(models=len(all_scores),expected=protocol['expected_models'],
                                 train_lock_sha256=sha(args.out/'train.lock.json'),
                                 audited_models=len(lock['models'])),
                 summaries=summaries,quality_frontier=rate,
                 interpretation='Tested small synthetic families, frozen 800-update budget; not a proof of general model capacity.')
    (dest/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
    print('AUDIT_COMPLETE',len(all_scores),'models',flush=True)
    for row in summaries:
        print(row['family'],row['cap'],'ratio',round(row['geometric_mean_ratio'],4),
              'iid_wins',row['iid_wins'],'ood_wins',row['ood_wins'],flush=True)
    print('RATE_WINS',sum(r['mirror_better_by_10pct'] for r in rate),'/',len(rate),flush=True)


if __name__ == '__main__':
    main()
