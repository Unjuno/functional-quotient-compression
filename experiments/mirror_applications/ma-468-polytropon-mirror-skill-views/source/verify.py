#!/usr/bin/env python3
"""Replay MA-468 task outputs from serialized skill banks."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from run import ALLOC,TASKS,world

def load(method,path):
    with np.load(path,allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files if not k.startswith('__')}
    if method in ('mirror','native_lowrank'):s={'base':torch.tensor(a['shared_physical_skills']),'u':torch.tensor(a['view_basis_left']),'v':torch.tensor(a['view_basis_right']),'codes':torch.tensor(a['logical_skill_view_codes']),'private':torch.tensor(a['private_skills'])}
    elif method=='physical_only':s={'modules':torch.tensor(a['physical_modules'])}
    elif method=='polytropon':s={'skills':torch.tensor(a['skill_bank'])}
    else:s={'tasks':torch.tensor(a['task_matrices'])}
    return s,a['task_skill_allocation'] if 'task_skill_allocation' in a else np.eye(TASKS,dtype=np.uint8)

def matrix(method,s,t,alloc):
    if method=='independent':return s['tasks'][t]
    ids=np.flatnonzero(alloc[t]);mats=[]
    for k in ids:
        if method in ('mirror','native_lowrank'):m=s['base'][k//2]+s['codes'][k]*torch.outer(s['u'],s['v']) if k<6 else s['private'][k-6]
        elif method=='physical_only':m=s['modules'][k//2 if k<6 else k-6+3]
        else:m=s['skills'][k]
        mats.append(m)
    return torch.stack(mats).sum(0)/(len(mats)**.5)

def verify(root):
    errors=[];checks=[];mx=0.
    for seed in (46801,46802):
        d=root/f'runs/dev_{seed}';metrics=json.loads((d/'metrics.json').read_text());teacher,data=world(seed);alloc=teacher['allocation']
        for method,rec in metrics['methods'].items():
            path=d/f'{method}_payload.npz';raw=path.read_bytes();s,a=load(method,path)
            if len(raw)!=rec['payload_bytes'] or hashlib.sha256(raw).hexdigest()!=rec['payload_sha256']:errors.append(f'{seed}/{method}: payload mismatch')
            vals=[]
            with torch.no_grad():
                for t in range(TASKS):
                    y=data['yq'][t];x=data['xq'][t];pred=x@matrix(method,s,t,alloc).T;vals.append(float(torch.sqrt(((pred-y)**2).mean())))
            diff=max(abs(x-y) for x,y in zip(vals,rec['per_task_rmse']));mx=max(mx,diff)
            if diff>1e-7:errors.append(f'{seed}/{method}: metric replay {diff}')
        if (d/'mirror_payload.npz').read_bytes()!=(d/'native_lowrank_payload.npz').read_bytes():errors.append(f'{seed}: native low-rank control is not byte-identical')
        else:checks.append({'seed':seed,'native_alias_sha256':hashlib.sha256((d/'mirror_payload.npz').read_bytes()).hexdigest(),'metric_replay':True})
    report={'experiment_id':'MA-468','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':mx,'fresh_accessed':False,'checks':checks,'errors':errors};(root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=verify(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
