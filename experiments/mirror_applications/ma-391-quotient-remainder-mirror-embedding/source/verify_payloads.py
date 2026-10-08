"""Replay all measured task metrics and unique-vector counts from FP16 archives."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import torch
import run_experiment as exp

ROOT=Path(__file__).resolve().parents[1]


def verify():
    checks=[]
    for seed in (39101,39102):
        for family in ('separable','xor'):
            data=exp.make_world(seed,family)
            summary=json.loads((ROOT/'results'/'development'/f'{seed}_{family}_result.json').read_text())
            for method in ('full','add','multiply','concat','mirror'):
                path=ROOT/'results'/'development'/f'{seed}_{family}_{method}.zip'
                with zipfile.ZipFile(path) as z:arrays={Path(n).stem:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')}
                model=exp.Composition(method);model.load_state_dict({k:torch.from_numpy(arrays[k].astype(np.float32)) for k in model.state_dict()})
                got=exp.evaluate(model,*data['test']);expected=summary['methods'][method]['test']
                assert got==expected
                blob=path.read_bytes();checks.append({'seed':seed,'family':family,'method':method,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'unique_embeddings':got['decoded_unique_embeddings']})
    return checks


if __name__=='__main__':
    rows=verify();print(json.dumps({'payloads_replayed':len(rows),'checks':rows},indent=2))
