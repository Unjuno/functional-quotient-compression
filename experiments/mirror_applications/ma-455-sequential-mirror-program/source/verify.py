#!/usr/bin/env python3
"""Recheck MA-455 serialization, checksum, alias and metric replay."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from run import METHODS,data,predict

def load(method, path):
    with np.load(path,allow_pickle=False) as z:
        a={k:np.array(z[k]) for k in z.files if not k.startswith('__')}
    if method=='shared':p=[torch.tensor(a['shared_block'])]
    elif method in ('mirror','native_givens'):p=[torch.tensor(a['shared_block']),torch.tensor(a['step_angles'])]
    elif method=='rank1':p=[torch.tensor(a['shared_block']),torch.tensor(a['residual_left']),torch.tensor(a['residual_right'])]
    else:p=[torch.tensor(a['step_blocks'])]
    return p,a

def verify(root):
    errors=[];checks=[]
    for seed in (45501,45502):
        d=root/f'runs/dev_{seed}';metrics=json.loads((d/'metrics.json').read_text())
        x,y,_=data(seed,512,'eval')
        for m in METHODS:
            path=d/f'{m}_payload.npz';raw=path.read_bytes();p,_=load(m,path);actual=hashlib.sha256(raw).hexdigest()
            if len(raw)!=metrics['methods'][m]['payload_bytes'] or actual!=metrics['methods'][m]['payload_sha256']:errors.append(f'{seed}/{m}: serialized bytes/hash mismatch')
            with torch.no_grad():rmse=float(torch.sqrt(((predict(x,m,p)-y)**2).mean()))
            if abs(rmse-metrics['methods'][m]['query_rmse'])>1e-7:errors.append(f'{seed}/{m}: query replay difference {rmse-metrics["methods"][m]["query_rmse"]}')
        mp=d/'mirror_payload.npz';np_=d/'native_givens_payload.npz'
        if mp.read_bytes()!=np_.read_bytes():errors.append(f'{seed}: native Givens payload not byte-identical to Mirror')
        else:checks.append({'seed':seed,'mirror_native_payload_hash':hashlib.sha256(mp.read_bytes()).hexdigest(),'metric_replay_checked':True})
    report={'experiment_id':'MA-455','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'fresh_accessed':False,'checks':checks,'errors':errors}
    (root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);x=a.parse_args();r=verify(x.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
