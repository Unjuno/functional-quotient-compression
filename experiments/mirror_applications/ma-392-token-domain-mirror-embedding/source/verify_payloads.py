"""Replay seen and held-out metrics from each serialized FP16 model."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import torch
import run_experiment as exp

ROOT=Path(__file__).resolve().parents[1]


def verify():
    checks=[]
    for seed in (39201,39202):
        splits,_=exp.make_world(seed);summary=json.loads((ROOT/'results'/'development'/f'{seed}_result.json').read_text())
        for method in ('independent','bias','rank4','domain_map','mirror'):
            path=ROOT/'results'/'development'/f'{seed}_{method}.zip'
            with zipfile.ZipFile(path) as z:arrays={Path(n).stem:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')}
            model=exp.DomainModel(method);model.load_state_dict({k:torch.from_numpy(arrays[k].astype(np.float32)) for k in model.state_dict()})
            got=exp.score(model,splits['test']);expected=summary['methods'][method]['test'];assert got==expected
            blob=path.read_bytes();checks.append({'seed':seed,'method':method,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'arrays':sorted(arrays)})
    return checks


if __name__=='__main__':
    rows=verify();print(json.dumps({'payloads_replayed':len(rows),'checks':rows},indent=2))
