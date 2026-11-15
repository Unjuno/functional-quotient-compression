# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Produce auditable H_S/H_M/H_C descriptive classification from frozen A12 data."""
from __future__ import annotations
import csv,json,statistics
from pathlib import Path
from collections import defaultdict

def diagnostic_gates(factors:list[dict])->list[dict]:
    grouped=defaultdict(list)
    for r in factors:grouped[(r['family'],r['method'],r['axis'])].append(r)
    result=[]
    for (family,method,axis),group in sorted(grouped.items()):
        if len(group)!=3:raise ValueError('requires three world comparisons')
        qualify=[]
        for r in group:
            good=r['delta_iid']>=.05 or r['relative_iid_gain']>=.25
            if axis=='meta':good &= (r['delta_old']>=.05)
            if axis=='support':good &= (r['delta_ood']>=-.02)
            qualify.append(int(good))
        result.append(dict(family=family,method=method,axis=axis,n_worlds=len(group),n_benefit=sum(qualify),
            prereg_benefit=int(sum(qualify)>=2),
            mean_iid_delta=statistics.mean(float(r['delta_iid']) for r in group),
            mean_iid_relative_gain=statistics.mean(float(r['relative_iid_gain']) for r in group),
            mean_old_delta=statistics.mean(float(r['delta_old']) for r in group),
            mean_ood_delta=statistics.mean(float(r['delta_ood']) for r in group)))
    return result


def main(root:Path)->None:
    root=Path(root)
    with (root/'main/factor_effects.csv').open() as f:
        factors=list(csv.DictReader(f))
    for r in factors:
        for k in ('delta_iid','relative_iid_gain','delta_old','delta_ood'):
            r[k]=float(r[k])
    rows=diagnostic_gates(factors)
    with (root/'main/attribution_gates.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print('ATTRIBUTION',len(rows),'gates')
    for fam in ('tent','sine'):
        for m in ('mirror','shift','film'):
            print(fam,m,[(r['axis'],r['n_benefit']) for r in rows if r['family']==fam and r['method']==m])

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();main(args.root)
