# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Independently rerun a deterministic sample of A11 frozen model fits.

Never selects a model by its held-out score. Rebuilds entire meta/adapt path.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from a11 import make_world,fit_two_stage,serialize


def select_replay_cases(rows:list[dict],limit:int)->list[dict]:
    if not isinstance(limit,int) or limit<1:raise ValueError('positive limit required')
    if not rows:raise ValueError('empty frozen model table')
    pending=sorted(rows,key=lambda r:(hashlib.sha256(str(r['path']).encode()).hexdigest(),str(r['path'])))
    seen={k:set() for k in ('family','method','cap','world','seed')}
    out=[]
    for _ in range(min(limit,len(pending))):
        # Greedily maximize uncovered experimental axes; no test-score access.
        def novelty(r):return sum(r[k] not in seen[k] for k in seen)
        winner=max(pending,key=novelty)
        pending.remove(winner)
        out.append(winner)
        for k in seen:seen[k].add(winner[k])
    return out


def verify_replay(run:Path,limit:int=12)->dict:
    run=Path(run)
    receipt=json.loads((run/'train.lock.json').read_text())
    config=json.loads((run/'config.json').read_text())
    sha=lambda b:hashlib.sha256(b).hexdigest()
    if receipt['config_sha256']!=sha((run/'config.json').read_bytes()):raise RuntimeError('config SHA mismatch')
    if receipt['protocol_sha256']!=sha((Path(__file__).parent/'PROTOCOL.md').read_bytes()):
        raise RuntimeError('protocol SHA mismatch')
    cases=select_replay_cases(receipt['models'],limit)
    results=[]
    for row in cases:
        world=make_world(row['world'],row['family'])
        if world['fingerprint']!=row['world_fingerprint']:raise RuntimeError('world fingerprint changed')
        fit=fit_two_stage(world,row['method'],row['cap'],row['seed'],
                          meta_steps=config['meta_steps'],adapt_steps=config['adapt_steps'],batch=config['batch'])
        actual=serialize(fit['model'])
        frozen=(run/row['path']).read_bytes()
        ok=(actual==frozen and sha(actual)==row['model_sha256'])
        results.append({'path':row['path'],'bytes':len(actual),'sha256':sha(actual),'bitwise_equal':ok})
        if not ok:raise RuntimeError('independent replay mismatch: '+row['path'])
    report={'checked':len(results),'passed':sum(r['bitwise_equal'] for r in results),
            'failed':sum(not r['bitwise_equal'] for r in results),
            'cases':results,'training_only':True}
    (run/'replay.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--limit',type=int,default=12)
    a=p.parse_args()
    r=verify_replay(a.run,a.limit)
    print(f"REPLAY PASS {r['passed']}/{r['checked']}")
