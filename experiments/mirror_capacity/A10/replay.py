# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Independent-process selected replay. No audit inputs are read."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from a10 import make_world,fit_two_stage,serialize


def choose_cases(rows:list[dict],count:int)->list[dict]:
    if count<1 or not rows:raise ValueError('count / rows invalid')
    groups={m:[r for r in rows if r['method']==m] for m in sorted({r['method'] for r in rows})}
    names=list(groups)
    if count>len(rows):raise ValueError('sample exceeds population')
    seen=set(); chosen=[]
    for i in range(count):
        group=groups[names[i%len(names)]]
        j=(3*(i%len(names)) + 7*(i//len(names)))%len(group)
        while id(group[j]) in seen:j=(j+1)%len(group)
        chosen.append(group[j]);seen.add(id(group[j]))
    return chosen


def replay(run:Path,count:int)->dict:
    run=Path(run)
    lock=json.loads((run/'train.lock.json').read_text())
    conf=json.loads((run/'config.json').read_text())
    allrows=lock['models']
    # Stratified deterministic positions, not selected by score.
    selected=choose_cases(allrows,count)
    rows=[]
    for r in selected:
        ix=allrows.index(r)
        world=make_world(r['world'])
        ft=fit_two_stage(world,r['method'],r['cap'],r['seed'],meta_steps=conf['meta_steps'],adapt_steps=conf['adapt_steps'],batch=conf['batch'])
        data=serialize(ft['model'])
        sha=hashlib.sha256(data).hexdigest()
        equal=data==(run/r['path']).read_bytes()
        rows.append(dict(method=r['method'],world=r['world'],seed=r['seed'],cap=r['cap'],
                         sha_match=sha==r['model_sha256'],byte_identical=equal))
        print('REPLAY',ix,r['method'],r['world'],r['seed'],equal,flush=True)
        if not (equal and sha==r['model_sha256']):raise AssertionError('independent replay mismatch')
    out=dict(n=len(rows),all_byte_identical=all(r['byte_identical'] for r in rows),cases=rows)
    (run/'replay.json').write_text(json.dumps(out,indent=2)+'\n')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--count',type=int,default=8)
    a=p.parse_args();replay(a.run,a.count)
