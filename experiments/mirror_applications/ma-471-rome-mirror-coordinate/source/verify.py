#!/usr/bin/env python3
"""Replay MA-471 inference metrics from exact serialized bank payloads."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from run import D,EDITS,TRAIN_EDITS,world,route,delta_from_angle

def verify(root):
    errors=[];checks=[];mx=0.
    for seed in (47101,47102):
        d=root/f'runs/dev_{seed}';doc=json.loads((d/'metrics.json').read_text());w=world(seed)
        for method,m in doc['methods'].items():
            path=d/f'{method}_bank.npz';raw=path.read_bytes()
            with np.load(path,allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files if not k.startswith('__')}
            if len(raw)!=m['inference_payload_bytes'] or hashlib.sha256(raw).hexdigest()!=m['payload_sha256']:errors.append(f'{seed}/{method}: bank bytes/hash mismatch')
            if method in ('mirror','native_givens'):
                updates=torch.stack([delta_from_angle(torch.tensor(t),torch.tensor(a['template_left']),torch.tensor(a['template_right'])) for t in a['edit_angles']])
            elif method=='rome_factors':updates=torch.tensor(a['rome_left'])[:,:,None]*torch.tensor(a['rome_right'])[:,None,:]
            elif method=='independent_fit':updates=torch.tensor(a['independent_updates'])
            else:updates=torch.zeros(EDITS,D,D)
            edits=[];routes=[];locals_=[];locroute=[]
            with torch.no_grad():
                for e in range(TRAIN_EDITS,EDITS):
                    for x,y in zip(w['xq'][e],w['yq'][e]):
                        i,on=route(x,torch.tensor(a['edit_keys']));routes.append(i==e and on);delta=updates[i] if on else torch.zeros(D,D)
                        edits.append(float(torch.mean((x@(torch.tensor(a['base'])+delta).T-y)**2)))
                for x in w['xl']:
                    i,on=route(x,torch.tensor(a['edit_keys']));locroute.append(on);delta=updates[i] if on else torch.zeros(D,D)
                    locals_.append(float(torch.mean((x@(torch.tensor(a['base'])+delta).T-x@torch.tensor(a['base']).T)**2)))
            vals=[float(np.sqrt(np.mean(edits))),float(np.sqrt(np.mean(locals_))),1-float(np.mean(routes)),float(np.mean(locroute))]
            expected=[m['heldout_edit_rmse'],m['locality_rmse'],m['wrong_route_rate'],m['locality_route_rate']]
            diff=max(abs(x-y) for x,y in zip(vals,expected));mx=max(mx,diff)
            if diff>1e-7:errors.append(f'{seed}/{method}: replay difference {diff}')
            ep=d/f'{method}_editor.npz'
            if method in ('mirror','native_givens'):
                eb=ep.read_bytes()
                if len(eb)!=m['editor_payload_bytes'] or hashlib.sha256(eb).hexdigest()!=m['editor_sha256']:errors.append(f'{seed}/{method}: editor payload mismatch')
        if (d/'mirror_bank.npz').read_bytes()!=(d/'native_givens_bank.npz').read_bytes():errors.append(f'{seed}: native Givens not byte identical')
        else:checks.append({'seed':seed,'native_givens_sha256':hashlib.sha256((d/'mirror_bank.npz').read_bytes()).hexdigest(),'replay':True})
    report={'experiment_id':'MA-471','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':mx,'fresh_accessed':False,'checks':checks,'errors':errors}
    (root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=verify(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
