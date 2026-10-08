#!/usr/bin/env python3
"""Replay MA-473 edit/locality metrics from serialized bank payloads."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from run import D,LAYERS,EDITS,TRAIN,world,route,decode

def replay(root):
    errors=[];checks=[];mx=0.
    for seed in (47301,47302):
        d=root/f'runs/dev_{seed}';doc=json.loads((d/'metrics.json').read_text());w=world(seed)
        for method,m in doc['methods'].items():
            path=d/f'{method}_bank.npz';raw=path.read_bytes()
            with np.load(path,allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files if not k.startswith('__')}
            if len(raw)!=m['inference_payload_bytes'] or hashlib.sha256(raw).hexdigest()!=m['payload_sha256']:errors.append(f'{seed}/{method}: byte/hash mismatch')
            if method in ('mirror','native_cp'):
                basis=torch.tensor(a['shared_layer_basis']);codes=torch.tensor(a['edit_codes']);updates=torch.stack([decode(c,basis) for c in codes])
            elif method in ('memit_full','independent_fit'):updates=torch.tensor(a['layer_updates'])
            elif method=='rank2_per_edit':
                left=torch.tensor(a['left_factors']);right=torch.tensor(a['right_factors']);updates=left@right
            else:updates=torch.zeros(EDITS,LAYERS,D,D)
            edit=[];routeok=[];local=[];locroute=[];base=torch.tensor(a['base']);keys=torch.tensor(a['edit_keys'])
            with torch.no_grad():
                for e in range(TRAIN,EDITS):
                    for x,y in zip(w['xq'][e],w['yq'][e]):
                        i,on=route(x,keys);routeok.append(i==e and on);delta=updates[i] if on else torch.zeros(LAYERS,D,D)
                        pred=torch.stack([x@(base[l]+delta[l]).T for l in range(LAYERS)],dim=0);edit.append(float(torch.mean((pred-y)**2)))
                for x in w['xl']:
                    i,on=route(x,keys);locroute.append(on);delta=updates[i] if on else torch.zeros(LAYERS,D,D)
                    pred=torch.stack([x@(base[l]+delta[l]).T for l in range(LAYERS)],dim=0);orig=torch.stack([x@base[l].T for l in range(LAYERS)],dim=0);local.append(float(torch.mean((pred-orig)**2)))
            vals=[float(np.sqrt(np.mean(edit))),float(np.sqrt(np.mean(local))),1-float(np.mean(routeok)),float(np.mean(locroute))]
            ex=[m['heldout_edit_rmse'],m['locality_rmse'],m['wrong_route_rate'],m['locality_route_rate']];diff=max(abs(x-y) for x,y in zip(vals,ex));mx=max(mx,diff)
            if diff>1e-7:errors.append(f'{seed}/{method}: metric replay mismatch {diff}')
            ep=d/f'{method}_editor.npz'
            if method in ('mirror','native_cp'):
                if not ep.exists():errors.append(f'{seed}/{method}: missing editor payload')
                else:
                    eraw=ep.read_bytes()
                    if len(eraw)!=m['editor_payload_bytes'] or hashlib.sha256(eraw).hexdigest()!=m['editor_sha256']:errors.append(f'{seed}/{method}: editor byte/hash mismatch')
        if (d/'mirror_bank.npz').read_bytes()!=(d/'native_cp_bank.npz').read_bytes():errors.append(f'{seed}: native CP not byte identical')
        else:checks.append({'seed':seed,'native_cp_sha256':hashlib.sha256((d/'mirror_bank.npz').read_bytes()).hexdigest(),'replay':True})
    report={'experiment_id':'MA-473','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':mx,'fresh_accessed':False,'checks':checks,'errors':errors}
    (root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=replay(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
