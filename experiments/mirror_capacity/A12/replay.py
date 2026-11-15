# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Independent process/regeneration of 12 predeclared A12 checkpoint identities."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from a12 import make_world,fit_meta_checkpoints,adapt_branch,serialize


def replay_main(out:Path,receipt_path:Path)->dict:
    out=Path(out); lock=json.loads((out/'train.lock.json').read_text())
    records={(r['family'],r['world'],r['seed'],r['method'],r['meta_steps'],r['support_n'],r['adapt_steps']):r for r in lock['models']}
    cases=[]
    for family in ('tent','sine'):
      for method in ('mirror','shift','film'):
        cases.append((family,1200,0,method,600,32,200))
        cases.append((family,1202,1,method,2400,256,800))
    checks=[]
    for family,world_id,seed,method,meta,support,adapt in cases:
        w=make_world(world_id,family)
        parent=fit_meta_checkpoints(w,method,1024,seed,steps=(meta,),batch=128)[meta]
        _,models=adapt_branch(parent,w,world_id,family,seed,method,support,steps=(adapt,),batch=128)
        got=serialize(models[adapt])
        row=records[(family,world_id,seed,method,meta,support,adapt)]
        want=(out/row['path']).read_bytes()
        checks.append(dict(path=row['path'],byte_identical=got==want,sha256=hashlib.sha256(got).hexdigest()))
    result=dict(experiment='A12_MAIN',checked=len(checks),passed=sum(r['byte_identical'] for r in checks),checks=checks)
    receipt_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if result['passed']!=len(checks):raise RuntimeError('independent replay failure')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--receipt',required=True,type=Path);args=p.parse_args()
    r=replay_main(args.run,args.receipt)
    print('REPLAYED',r['passed'],'/',r['checked'])
