#!/usr/bin/env python3
"""Replay MA-469 heldout edits/locality from serialized edit systems."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from run import D,EDITS,TRAIN_EDITS,world

def load(method,path):
    with np.load(path,allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files if not k.startswith('__')}
    if method in ('mend','mirror','native_lowrank'):
        g=torch.tensor(a['editor_fc2_weight'])
        out=D*D if method=='mend' else 2
        # hidden architecture is fixed at 32; evaluate directly from serialized tensors.
        def generate(sig):
            h=torch.tanh(torch.tensor(a['editor_fc1_weight'])@sig+torch.tensor(a['editor_fc1_bias']));v=g@h+torch.tensor(a['editor_fc2_bias'])
            return v.reshape(D,D) if method=='mend' else torch.einsum('r,rij->ij',v,torch.tensor(a['edit_basis']))
        return a,generate
    if method=='rome':return a,lambda sig,e:torch.tensor(a['rome_updates'][e])
    if method=='independent_fit':return a,lambda sig,e:torch.tensor(a['independent_updates'][e])
    return a,lambda sig,e:torch.zeros(D,D)

def verify(root):
    errors=[];checks=[];mx=0.
    for seed in (46901,46902):
        d=root/f'runs/dev_{seed}';metrics=json.loads((d/'metrics.json').read_text());w=world(seed)
        for method,rec in metrics['methods'].items():
            path=d/f'{method}_payload.npz';raw=path.read_bytes();a,fn=load(method,path)
            if len(raw)!=rec['payload_bytes'] or hashlib.sha256(raw).hexdigest()!=rec['payload_sha256']:errors.append(f'{seed}/{method}: payload mismatch')
            edit=[];loc=[]
            with torch.no_grad():
                for e in range(TRAIN_EDITS,EDITS):
                    sig=w['signals'][e];delta=fn(sig) if method in ('mend','mirror','native_lowrank') else fn(sig,e)
                    x=w['xq'][e];pred=x@(w['base']+delta).T;edit.append(float(torch.sqrt(((pred-w['yq'][e])**2).mean())))
                    xl=w['xl'][e];loc.append(float(torch.sqrt(((xl@delta.T)**2).mean())))
            diff=max(max(abs(x-y) for x,y in zip(edit,rec['per_edit_rmse'])),abs(float(np.mean(loc))-rec['locality_rmse']));mx=max(mx,diff)
            if diff>1e-7:errors.append(f'{seed}/{method}: edit/locality replay {diff}')
        if (d/'mirror_payload.npz').read_bytes()!=(d/'native_lowrank_payload.npz').read_bytes():errors.append(f'{seed}: native low-rank edit control not byte-identical')
        else:checks.append({'seed':seed,'native_alias_sha256':hashlib.sha256((d/'mirror_payload.npz').read_bytes()).hexdigest(),'edit_replay':True})
    r={'experiment_id':'MA-469','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':mx,'fresh_accessed':False,'checks':checks,'errors':errors};(root/'verification_report.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=verify(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
