#!/usr/bin/env python3
"""Post-hoc MA-1062 frequency-slice report; writes aggregates only, no row labels or IDs."""
import importlib.util,json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma1062',ROOT/'source/train.py'); ma=importlib.util.module_from_spec(spec); spec.loader.exec_module(ma)

def bucket(n):
    if n==0:return '0'
    if n<=5:return '1-5'
    if n<=20:return '6-20'
    return '>20'

def score(logits,labels):
    m=ma.metric(logits,labels)
    return {k:(None if not np.isfinite(v) else v) for k,v in m.items()}

def main():
    base=ma.read_base(); audit=ma.read_audit_after_freeze()
    uc=Counter(r[0] for r in base); ic=Counter(r[1] for r in base)
    out={'experiment_id':'MA-1062','audit_rows':len(audit),'slices':{'user_frequency':{},'movie_frequency':{}}}
    for axis,counter,index in [('user_frequency',uc,0),('movie_frequency',ic,1)]:
        groups={b:[] for b in ('0','1-5','6-20','>20')}
        for i,row in enumerate(audit): groups[bucket(counter[row[index]])].append(i)
        for b,inds in groups.items():
            out['slices'][axis][b]={'n':len(inds),'methods':{}}
            if not inds: continue
            labels=np.asarray([audit[i][5] for i in inds],dtype=np.float32)
            for method in ma.METHODS:
                out['slices'][axis][b]['methods'][method]={}
                for seed in ma.FRESH_SEEDS:
                    logits=np.load(ROOT/'runs/audit'/f'{method}_seed{seed}.npz')['logits'][inds]
                    out['slices'][axis][b]['methods'][method][str(seed)]=score(logits,labels)
    (ROOT/'runs/audit/frequency_slices.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'audit_rows':out['audit_rows'],'user_frequency_n':{k:v['n'] for k,v in out['slices']['user_frequency'].items()},'movie_frequency_n':{k:v['n'] for k,v in out['slices']['movie_frequency'].items()}},indent=2))
if __name__=='__main__':main()
