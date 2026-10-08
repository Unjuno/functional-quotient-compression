"""Reload serialized states and replay per-task metrics and actual-byte hashes."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'))
import run_experiment as exp

def decode(path):
    with zipfile.ZipFile(path) as z:
        meta=json.loads(z.read('metadata.json'));arr={Path(n).stem:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')}
    w=torch.from_numpy(arr['shared'].astype(np.float32));m=meta['method']
    if m=='independent_oracle':rec=w
    elif m=='tied':rec=w.expand(4,-1)
    elif m=='psp_sign':
        bits=np.unpackbits(arr['context_bits'])[:3*exp.D].reshape(3,exp.D).astype(np.float32);sign=torch.from_numpy(2*bits-1);rec=sign*w
    elif m=='factorized_mirror':
        c=torch.from_numpy(arr['context_codes'].astype(np.float32));a=torch.tensor([0.,c[0],c[1],c.sum()]);rec=exp.rotate(w.expand(4,-1),a)
    else:
        c=torch.from_numpy(arr['context_codes'].astype(np.float32));a=torch.tensor([c[0],c[1],c[2],c[1]+c[2]-c[0]]);rec=exp.rotate(w.expand(4,-1),a)
    return m,arr,rec

def verify():
    rows=[]
    for split,seeds in (('development',(25701,25702)),('fresh',(25711,25712,25713))):
      for seed in seeds:
        d=json.loads((ROOT/f'results/{split}'/f'{seed}_result.json').read_text());_,_,_,world=exp.make_world(seed)
        for method,expected in d['methods'].items():
            path=ROOT/f'results/{split}'/f'{seed}_{method}.zip';blob=path.read_bytes();assert len(blob)==expected['actual_payload_bytes'] and hashlib.sha256(blob).hexdigest()==expected['payload_sha256']
            _,arr,rec=decode(path);q=exp.eval_weights(rec[:3],world['test'][0][:3],world['test'][1][:3]);delta=abs(q['mean_accuracy']-expected['quality_observed_tasks']['mean_accuracy']);assert delta<.01,(seed,method,delta)
            if method!='psp_sign':
                qa=exp.eval_weights(rec,*world['test']);da=abs(qa['mean_accuracy']-expected['quality_all_tasks']['mean_accuracy']);assert da<.01
            rows.append({'split':split,'seed':seed,'method':method,'bytes':len(blob),'sha256':expected['payload_sha256'],'array_count':len(arr),'observed_accuracy_replay_delta':delta})
    return rows
if __name__=='__main__':print(json.dumps({'payloads_replayed':len(verify()),'rows':verify()},indent=2))
