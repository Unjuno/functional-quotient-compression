#!/usr/bin/env python3
"""Replay MA-470 local edit and locality metrics from exact serialized payloads."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from run import D,EDITS,TRAIN_EDITS,world,route

def load(path,method):
    with np.load(path,allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files if not k.startswith('__')}
    if method in ('mend','mirror','native_lowrank'):
        w1=torch.tensor(a['editor_fc1_weight']);b1=torch.tensor(a['editor_fc1_bias']);w2=torch.tensor(a['editor_fc2_weight']);b2=torch.tensor(a['editor_fc2_bias'])
        basis=torch.tensor(a['edit_basis']) if method!='mend' else None
        def update(sig):
            h=torch.tanh(w1@sig.reshape(-1)+b1);out=w2@h+b2
            return out.reshape(D,D) if method=='mend' else torch.einsum('r,rij->ij',out,basis)
        updates=None
    else:
        updates=torch.tensor(a['edit_updates'] if method=='mend' else a.get({'rome':'rome_updates','independent_fit':'independent_updates'}.get(method,''),np.zeros((EDITS,D,D),np.float32))) if method!='no_edit' else torch.zeros(EDITS,D,D)
        update=None
    return a,update,updates

def verify(root):
    errors=[];checks=[];maxdiff=0.
    for seed in (47001,47002):
        d=root/f'runs/dev_{seed}';metrics=json.loads((d/'metrics.json').read_text());w=world(seed)
        for method,rec in metrics['methods'].items():
            path=d/f'{method}_payload.npz';raw=path.read_bytes();a,make,stored=load(path,method)
            if len(raw)!=rec['payload_bytes'] or hashlib.sha256(raw).hexdigest()!=rec['payload_sha256']:errors.append(f'{seed}/{method}: byte/hash mismatch')
            updates=torch.stack([make(w['signals'][e]) for e in range(EDITS)]) if make else stored
            edits=[];route_ok=[];locals_=[];localroutes=[]
            with torch.no_grad():
                for e in range(TRAIN_EDITS,EDITS):
                    for x,y in zip(w['xq'][e],w['yq'][e]):
                        ix,on=route(x,torch.tensor(a['edit_keys']));route_ok.append(ix==e and on)
                        delta=updates[ix] if on else torch.zeros(D,D)
                        edits.append(float(torch.mean((x@(torch.tensor(a['base'])+delta).T-y)**2)))
                for x in w['xl']:
                    ix,on=route(x,torch.tensor(a['edit_keys']));localroutes.append(on)
                    delta=updates[ix] if on else torch.zeros(D,D)
                    locals_.append(float(torch.mean((x@(torch.tensor(a['base'])+delta).T-x@torch.tensor(a['base']).T)**2)))
            e_rmse=float(np.sqrt(np.mean(edits)));l_rmse=float(np.sqrt(np.mean(locals_)));wr=1-float(np.mean(route_ok));lr=float(np.mean(localroutes))
            ds=[abs(e_rmse-rec['heldout_edit_rmse']),abs(l_rmse-rec['locality_rmse']),abs(wr-rec['wrong_route_rate']),abs(lr-rec['locality_route_rate'])]
            diff=max(ds);maxdiff=max(maxdiff,diff)
            if diff>1e-7:errors.append(f'{seed}/{method}: metric replay difference {diff}')
        if (d/'mirror_payload.npz').read_bytes()!=(d/'native_lowrank_payload.npz').read_bytes():errors.append(f'{seed}: native control not byte identical')
        else:checks.append({'seed':seed,'native_alias_sha256':hashlib.sha256((d/'mirror_payload.npz').read_bytes()).hexdigest(),'replay':True})
    report={'experiment_id':'MA-470','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':maxdiff,'fresh_accessed':False,'checks':checks,'errors':errors}
    (root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=verify(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
