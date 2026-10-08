"""Replay all fresh/development expert-bank metrics from serialized payloads."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'))
import run_experiment as exp

def verify():
    rows=[]
    for split,seeds in (('development',(25821,25822)),):
      for regime in ('aligned','unrelated'):
       for seed in seeds:
        d=json.loads((ROOT/f'results/{split}'/f'{regime}_{seed}_result.json').read_text());target,_=exp.make_world(seed,regime)
        for method,expected in d['methods'].items():
            path=ROOT/f'results/{split}'/f'{regime}_{seed}_{method}.zip';blob=path.read_bytes();assert len(blob)==expected['actual_payload_bytes'] and hashlib.sha256(blob).hexdigest()==expected['payload_sha256']
            _,_,rec=exp.decode(path);q=exp.score(rec,target,seed);delta=abs(q['mean_normalized_mse']-expected['quality']['mean_normalized_mse']);assert delta<1e-6,(split,seed,regime,method,delta)
            rows.append({'split':split,'seed':seed,'regime':regime,'method':method,'bytes':len(blob),'sha256':expected['payload_sha256'],'mean_mse_replay_delta':delta})
    return rows
if __name__=='__main__':print(json.dumps({'payloads_replayed':len(verify()),'rows':verify()},indent=2))
