#!/usr/bin/env python3
"""Replay MA-461 adapter outputs from serialized model states."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from run import D,TASKS,LAYERS,NEVAL,TRAIN,HELD,teacher

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k]) for k in z.files if not k.startswith('__')}

def predict(method,a,t,l):
    base=torch.tensor(a['base_matrix']);
    if method in ('mirror','native_lowrank','hyperformer'):
        x=torch.cat([torch.tensor(a['task_embedding'][t]),torch.tensor(a['layer_embedding'][l])])
        h=torch.tanh(torch.tensor(a['generator_fc1_weight'])@x+torch.tensor(a['generator_fc1_bias']))
        o=torch.tensor(a['generator_fc2_weight'])@h+torch.tensor(a['generator_fc2_bias'])
        if method in ('mirror','native_lowrank'):delta=torch.einsum('r,rij->ij',o,torch.tensor(a['adapter_basis']))
        else:delta=o.reshape(D,D)
        return base+delta
    if method=='shared':return base+torch.tensor(a['shared_delta'])
    return base+torch.tensor(a['adapter_table'][t,l])

def verify(root):
    errors=[];checks=[];maxdiff=0.
    for seed in (46101,46102):
        d=root/f'runs/dev_{seed}';m=json.loads((d/'metrics.json').read_text());_,targets=teacher(seed)
        for method,rec in m['methods'].items():
            path=d/f'{method}_payload.npz';raw=path.read_bytes();a=load(path)
            if len(raw)!=rec['payload_bytes'] or hashlib.sha256(raw).hexdigest()!=rec['payload_sha256']:errors.append(f'{seed}/{method}: payload mismatch')
            rng=np.random.default_rng(seed+46121);hs=[];ss=[]
            for t,l in HELD:
                x=torch.tensor(rng.normal(size=(NEVAL,D)).astype(np.float32));y=x@targets[t,l].T;xhat=x@predict(method,a,t,l).T;hs.append(float(torch.sqrt(((y-xhat)**2).mean())))
            for t,l in TRAIN:
                x=torch.tensor(rng.normal(size=(NEVAL,D)).astype(np.float32));y=x@targets[t,l].T;xhat=x@predict(method,a,t,l).T;ss.append(float(torch.sqrt(((y-xhat)**2).mean())))
            # independent table consumes the same random query stream but has no valid audit prediction; only seen pairs are authoritative.
            if method!='independent_seen':
                diff=max(abs(x-y) for x,y in zip(hs,rec['heldout_pair_rmse']));maxdiff=max(maxdiff,diff)
                if diff>1e-7:errors.append(f'{seed}/{method}: heldout metric replay {diff}')
            sd=max(abs(x-y) for x,y in zip(ss,rec['seen_pair_rmse']));maxdiff=max(maxdiff,sd)
            if sd>1e-7:errors.append(f'{seed}/{method}: seen metric replay {sd}')
        if (d/'mirror_payload.npz').read_bytes()!=(d/'native_lowrank_payload.npz').read_bytes():errors.append(f'{seed}: direct low-rank payload differs from Mirror')
        else:checks.append({'seed':seed,'mirror_native_sha256':hashlib.sha256((d/'mirror_payload.npz').read_bytes()).hexdigest(),'metric_replay':True})
    r={'experiment_id':'MA-461','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':maxdiff,'fresh_accessed':False,'checks':checks,'errors':errors};(root/'verification_report.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=verify(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
