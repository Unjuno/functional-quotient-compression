from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
import torch
from run import make_world, predict_from_payload

ROOT=Path(__file__).resolve().parents[1]

def verify():
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    rows=list(csv.DictReader((ROOT/'runs/RESULTS_CORE.csv').open()))
    checks=[]
    assert len(rows)==2*6*5, f'expected 60 result rows, found {len(rows)}'
    assert set(int(r['seed']) for r in rows)=={40101,40102}
    for seed in (40101,40102):
        x,h,y,w1,w2,mats=make_world(seed)
        for method in ('shared_identity','film','rank1','mirror','mirror_rank1','independent'):
            selected=[r for r in rows if int(r['seed'])==seed and r['method']==method]
            path=ROOT/'runs'/f'dev{seed}_{method}.npz'; raw=path.read_bytes()
            digest=hashlib.sha256(raw).hexdigest()
            assert all(r['artifact_sha256']==digest for r in selected)
            pred=torch.as_tensor(predict_from_payload(raw,x.numpy(),method))
            den=((y-y.mean(dim=0,keepdim=True))**2).mean(dim=(0,2)).sqrt().clamp_min(1e-8)
            nrmse=(((pred-y)**2).mean(dim=(0,2)).sqrt()/den).reshape(5,8).mean(dim=1).numpy()
            actual_bytes=len(raw)
            assert all(int(r['payload_bytes'])==actual_bytes for r in selected)
            for r,a in zip(selected,nrmse):
                assert abs(float(r['nrmse'])-float(a))<1e-7, (seed,method,r['alpha'],r['nrmse'],a)
            checks.append({'seed':seed,'method':method,'payload_sha256':digest,'payload_bytes':actual_bytes,'metric_rows_replayed':5})
    screen=json.loads((ROOT/'runs/screen.json').read_text())
    assert screen['fresh_accessed'] is False
    assert screen['development_gate_passed'] is False
    return {'experiment_id':'MA-401','status':'VERIFIED_DEVELOPMENT_FAIL','protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),'metric_replay':'PASS','payload_replay':'PASS','fresh_accessed':False,'checks':checks,'screen_gate':'FAIL: throughput and independent-byte ratio missed in both development seeds'}

if __name__=='__main__':
    result=verify();(ROOT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
