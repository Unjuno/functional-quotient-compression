"""Replay held-out and seen metrics from serialized FP16 models."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import torch
import run_experiment as exp

ROOT=Path(__file__).resolve().parents[1]


def verify():
    rows=[]
    for seed in (39501,39502):
        splits,_=exp.make_world(seed);summary=json.loads((ROOT/'results'/'development'/f'{seed}_result.json').read_text())
        for method in ('independent','shared','post_rank4','post_full','mirror'):
            path=ROOT/'results'/'development'/f'{seed}_{method}.zip'
            with zipfile.ZipFile(path) as z:arrays={Path(n).stem:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')}
            model=exp.FactorizedDomainModel(method);model.load_state_dict({k:torch.from_numpy(arrays[k].astype(np.float32)) for k in model.state_dict()})
            got=exp.score(model,splits['test']);expected=summary['methods'][method]['test'];assert got==expected
            blob=path.read_bytes();rows.append({'seed':seed,'method':method,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'arrays':sorted(arrays)})
    return rows


if __name__=='__main__':
    rows=verify();print(json.dumps({'payloads_replayed':len(rows),'checks':rows},indent=2))
