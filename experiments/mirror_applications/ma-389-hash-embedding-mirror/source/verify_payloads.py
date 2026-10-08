"""Replay saved token prediction metrics from the serialized inference archives."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import run_experiment as exp

ROOT=Path(__file__).resolve().parents[1]


def verify():
    checks=[]
    for seed in (38901,38902):
        data,hashes,token_labels=exp.make_world(seed)
        degrees=exp.collision_bins(hashes)
        summary=json.loads((ROOT/'results'/'development'/f'{seed}_result.json').read_text())
        for method in ('full','hash','unweighted','scalar','mirror'):
            path=ROOT/'results'/'development'/f'{seed}_{method}.zip'
            model=exp.TokenModel(method,hashes)
            arrays=exp.load_archive(path,model)
            got=exp.evaluate(model,*data['test'],degrees,token_labels)
            expected=summary['methods'][method]['test']
            assert got['accuracy']==expected['accuracy']
            assert got['nll']==expected['nll']
            assert got['per_token_collision_bins']==expected['per_token_collision_bins']
            blob=path.read_bytes()
            checks.append({'seed':seed,'method':method,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),
                           'arrays':sorted(arrays),'charged_hash_indices':method!='full'})
    return checks


if __name__=='__main__':
    rows=verify();print(json.dumps({'payloads_replayed':len(rows),'checks':rows},indent=2))
