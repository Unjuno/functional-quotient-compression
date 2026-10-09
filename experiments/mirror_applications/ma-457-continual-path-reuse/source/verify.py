#!/usr/bin/env python3
"""Replay MA-457 metrics from actual serialized inference payloads."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from run import D,NTASK,task_data

def load(method,path):
    with np.load(path,allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files if not k.startswith('__')}
    if method=='pathnet':s={'modules':[torch.tensor(x) for x in a['module_bank']],'paths':a['task_path'].astype(int).tolist()}
    elif method in ('mirror','native_rank1'):
        priv={i:torch.tensor(a['private_modules'][idx]) for i,idx in enumerate(a['task_module_index']) if idx>=0}
        s={'w':torch.tensor(a['shared_module']),'u':torch.tensor(a['basis_left']),'v':torch.tensor(a['basis_right']),'codes':{i:float(a['task_codes'][i]) for i in range(NTASK)},'births':priv}
    elif method=='shared':s=torch.tensor(a['shared_module'])
    else:s=torch.tensor(a['task_modules'])
    return s

def pred(method,s,t,x):
    if method=='pathnet':return x@s['modules'][s['paths'][t]].T
    if method in ('mirror','native_rank1'):
        if t in s['births']:return x@s['births'][t].T
        return x@s['w'].T+s['codes'][t]*(x@s['v'])[:,None]*s['u']
    if method=='shared':return x@s.T
    return x@s[t].T

def verify(root):
    errors=[];checks=[];maxdiff=0.
    for seed in (45701,45702):
        d=root/f'runs/dev_{seed}';m=json.loads((d/'metrics.json').read_text());
        for method,rec in m['methods'].items():
            path=d/f'{method}_payload.npz';raw=path.read_bytes();s=load(method,path)
            if len(raw)!=rec['payload_bytes'] or hashlib.sha256(raw).hexdigest()!=rec['payload_sha256']:errors.append(f'{seed}/{method}: byte/hash mismatch')
            scores=[]
            with torch.no_grad():
                for t in range(NTASK):
                    x,y=task_data(seed,t,'query');scores.append(float(torch.sqrt(((pred(method,s,t,x)-y)**2).mean())))
            diff=max(abs(a-b) for a,b in zip(scores,rec['per_task_query_rmse']));maxdiff=max(maxdiff,diff)
            if diff>1e-7:errors.append(f'{seed}/{method}: metric replay diff {diff}')
        if (d/'mirror_payload.npz').read_bytes()!=(d/'native_rank1_payload.npz').read_bytes():errors.append(f'{seed}: rank-one native payload not byte-identical')
        else:checks.append({'seed':seed,'native_alias_sha256':hashlib.sha256((d/'mirror_payload.npz').read_bytes()).hexdigest(),'metric_replay':True})
    report={'experiment_id':'MA-457','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':maxdiff,'fresh_accessed':False,'checks':checks,'errors':errors}
    (root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=verify(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
