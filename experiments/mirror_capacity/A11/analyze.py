# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Observed-grid, censor-aware rate/quality analysis (NOT a theoretical minimum)."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

BASELINES=frozenset(('shift','film','basis','multihead'))


def observed_frontier(world_rows:list[dict])->list[dict]:
    groups=defaultdict(list)
    for row in world_rows:
        groups[(str(row['family']),int(row['world']))].append(row)
    out=[]
    for (family,world),rows in sorted(groups.items()):
        mirror=[r for r in rows if r['method']=='mirror' and int(r['passes'])==1]
        ctrl=[r for r in rows if r['method'] in BASELINES and int(r['passes'])==1]
        m=min(mirror,key=lambda r:float(r['stored_bytes'])) if mirror else None
        c=min(ctrl,key=lambda r:float(r['stored_bytes'])) if ctrl else None
        mb=float(m['stored_bytes']) if m else None
        cb=float(c['stored_bytes']) if c else None
        ratio=mb/cb if mb is not None and cb is not None else None
        classif=('both_observed' if ratio is not None else
                 'both_censored' if mb is None and cb is None else
                 'mirror_censored' if mb is None else 'control_censored')
        out.append({'family':family,'world':world,'mirror_bytes':mb,'mirror_cap':int(m['cap']) if m else None,
                    'reference_bytes':cb,'reference_method':c['method'] if c else None,
                    'reference_cap':int(c['cap']) if c else None,'observed_ratio':ratio,
                    'mirror_10pct_win':bool(ratio<=0.9) if ratio is not None else None,
                    'classification':classif})
    return out


def analyze_run(root:Path)->dict:
    root=Path(root)
    rows=list(csv.DictReader((root/'worlds.csv').open(encoding='utf-8',newline='')))
    summary=list(csv.DictReader((root/'summary.csv').open(encoding='utf-8',newline='')))
    receipt=json.loads((root/'train.lock.json').read_text(encoding='utf-8'))
    frontier=observed_frontier(rows)
    counts=defaultdict(lambda:defaultdict(int))
    for row in rows:counts[row['family']][row['method']]+=int(row['passes'])
    result=dict(experiment='A11_MAIN',model_count=len(receipt['models']),split_score_count=len(receipt['models'])*3,
                independent_worlds_per_family=len(set(int(x['world']) for x in rows)),
                seeds_per_world=len(set(int(x['seed']) for x in receipt['models'])),
                caps=sorted(set(int(x['cap']) for x in rows)),
                frontiers=frontier,
                passing_group_counts={family:dict(method_counts) for family,method_counts in counts.items()},
                warning='Observed passing model size is a finite-grid upper bound only; with no pass, frontier censored. Comparisons do NOT match training FLOPs.' )
    (root/'frontier.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    with (root/'frontier.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=frontier[0].keys());wr.writeheader();wr.writerows(frontier)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    r=analyze_run(a.run)
    print('FRONTIER',json.dumps(r['frontiers'],ensure_ascii=False))
