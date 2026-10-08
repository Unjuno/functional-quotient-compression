"""Reload all inference archives and replay their final task metrics."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_experiment as exp  # noqa: E402


def verify(root):
    checks=[]
    for family,seed in (("aligned",38501),("aligned",38502),("unrelated",38501),("unrelated",38502)):
        data,_,_,_,_=exp.make_world(seed,family)
        summary=json.loads((root/f"{family}_{seed}_result.json").read_text())
        for method in ("explicit","tied","scalar","mirror","private_rank4","hyper"):
            path=root/f"{family}_{seed}_{method}.zip"
            with zipfile.ZipFile(path) as z:
                arrays={Path(n).stem:np.load(io.BytesIO(z.read(n)),allow_pickle=False)
                        for n in z.namelist() if n.endswith('.npy')}
            model=exp.ExpertPrompts(method)
            model.load_state_dict({k:torch.from_numpy(arrays[k.replace('.','_')].astype('float32'))
                                   for k in model.state_dict()})
            keys=torch.from_numpy(arrays['keys'].astype('float32'))
            classifier=torch.from_numpy(arrays['classifier'].astype('float32'))
            got=exp.evaluate(model,data['test'],keys,classifier,range(8))
            expected=summary['methods'][method]['test']
            assert got['per_task_accuracy']==expected['per_task_accuracy']
            assert got['retrieval_accuracy']==expected['retrieval_accuracy']
            delta=abs(got['mean_nll']-expected['mean_nll']);assert delta<5e-5
            blob=path.read_bytes()
            checks.append({'file':path.name,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'nll_delta':delta})
    return checks


if __name__=='__main__':
    checks=verify(HERE.parent/'results'/'development')
    print(json.dumps({'payloads_replayed':len(checks),'max_nll_delta':max(x['nll_delta'] for x in checks),'checks':checks},indent=2))
