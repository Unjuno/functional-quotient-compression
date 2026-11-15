# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Post-hoc independent evidence guard for the immutable A14 optimizer source.

A14's source SHA is locked in the experiment config. This additional validator
checks the referenced A13 dependency hashes and each model against the ledger.
It does not alter the original A14 training or scoring procedure.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from a14 import A13,sha,deserialize,serialize,check_frozen_parent,load_parent_and_es

def verify_dependency(expected:dict,root:Path)->int:
    n=0
    for path,digest in expected.items():
        source=root/path
        if not source.is_file() or sha(source.read_bytes())!=digest:
            raise RuntimeError('dependency SHA mismatch: '+path)
        n+=1
    return n

def verify(results:Path=HERE/'results',base:Path=A13):
    raw=(results/'config.json').read_bytes()
    cfg=json.loads(raw)
    lock=json.loads((results/'train.lock.json').read_text())
    if sha(raw)!=lock['config_sha256']:raise RuntimeError('frozen config SHA mismatch')
    verify_dependency(cfg['source_sha256'],HERE)
    verify_dependency(cfg['dependency_source_sha256'],base)
    for r in lock['models']:
        path=results/r['path'];data=path.read_bytes()
        if sha(data)!=r['sha256'] or len(data)!=r['stored_bytes']:raise RuntimeError('model file SHA/byte mismatch')
        child=deserialize(data)
        if serialize(child)!=data:raise RuntimeError('replay mismatch')
        parent,_,world,parentrec,esrec=load_parent_and_es(base,r['family'],r['world'],r['seed'],r['method'])
        if r['world_fingerprint']!=world['fingerprint'] or r['support_es_sha256']!=esrec['sha256']:
            raise RuntimeError('provenance mismatch')
        check_frozen_parent(parent,child,parentrec['sha256'])
    result=dict(checkpoints=len(lock['models']),config_sha256=sha(raw),source_files_verified=len(cfg['source_sha256']),
                dependencies_verified=len(cfg['dependency_source_sha256']),all_parent_weights_frozen=True)
    (results/'validation.json').write_text(json.dumps(result,indent=2,sort_keys=True))
    print('VALIDATION_PASS',result)
    return result

if __name__=='__main__':verify()
