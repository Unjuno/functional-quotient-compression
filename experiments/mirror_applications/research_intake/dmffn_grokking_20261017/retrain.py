"""Deterministic complete training replay of two frozen-primary models."""
from pathlib import Path
import os,json
import torch
from model import Config
from train import train_one
ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(1)
    p=ROOT/'results'/'fresh'
    out=[]
    for world,seed,method,wd in [(87011,101,'mirror',0.5),(87012,202,'gain',0.5)]:
        label=method
        prior=json.loads((p/f'result_{world}_{seed}_{label}.json').read_text())
        history=json.loads((p/f'history_{world}_{seed}_{label}.json').read_text())
        a=train_one(world,seed,method,Config(p=31,h=64,q=16),0.005,wd,0.60,8000,200,True)
        rows=a.pop('history');raw=a.pop('raw')
        assert rows==history,(world,seed,method,'metric history drift')
        assert a['payload_sha256']==prior['payload_sha256'],(world,seed,method,'payload hash drift')
        assert raw==(p/f'model_{world}_{seed}_{label}.pt').read_bytes()
        out.append({'world':world,'seed':seed,'method':method,'all_checkpoints_match':True,
                    'payload_sha256_match':True,'serialized_bytes_match':True,'history_points':len(rows)})
        print('retrain match',world,seed,method,flush=True)
    (p/'RETRAIN_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
