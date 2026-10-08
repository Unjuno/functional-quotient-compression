"""Reload every development payload and replay held-out metrics from serialized state."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'))
import run_experiment as exp

def load(path):
    with zipfile.ZipFile(path) as z:
        meta=json.loads(z.read('metadata.json'));a={Path(n).stem:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')}
    shared=torch.from_numpy(a['shared'].astype(np.float32));m=meta['method']
    if m=='independent': recovered=shared
    elif m=='tied': recovered=shared.expand(exp.T,-1)
    elif m=='psp_sign':
        bits=np.unpackbits(a['context_bits'])[:exp.T*exp.D].reshape(exp.T,exp.D).astype(np.float32); signs=torch.from_numpy(bits*2-1);recovered=signs*shared
    elif m.startswith('mirror'):
        angles=torch.from_numpy(a['angles'].astype(np.float32));recovered=exp.retrieve_phase(shared,angles)
    else:
        codes=torch.from_numpy(a['dense_codes'].astype(np.float32));recovered=codes*shared
    return meta,a,recovered

def verify():
    rows=[]
    for seed in (25501,25502):
        summary=json.loads((ROOT/'results/development'/f'{seed}_result.json').read_text());_,data=exp.make_world(seed)
        for method,expected in summary['methods'].items():
            path=ROOT/'results/development'/f'{seed}_{method}.zip';blob=path.read_bytes()
            assert len(blob)==expected['actual_payload_bytes'] and hashlib.sha256(blob).hexdigest()==expected['payload_sha256']
            meta,arrays,recovered=load(path); replay=exp.evaluate(recovered,*data['test'])
            delta=abs(replay['mean_accuracy']-expected['quality']['mean_accuracy'])
            assert delta<=.01,(seed,method,delta)
            rows.append({'seed':seed,'method':method,'bytes':len(blob),'sha256':expected['payload_sha256'],'array_count':len(arrays),'mean_accuracy_replay_delta':delta})
    return rows
if __name__=='__main__': print(json.dumps({'payloads_replayed':len(verify()),'rows':verify()},indent=2))
