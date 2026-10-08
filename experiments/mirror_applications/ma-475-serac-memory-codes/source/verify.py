#!/usr/bin/env python3
"""Replay MA-475 value/retrieval metrics from serialized memories."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from run import D,N,TRAIN,HELD,NLOCAL,world,route

def replay(root):
    errors=[];checks=[];mx=0.
    for seed in (47501,47502):
        d=root/f'runs/dev_{seed}';doc=json.loads((d/'metrics.json').read_text());w=world(seed)
        for method,m in doc['methods'].items():
            path=d/f'{method}_payload.npz';raw=path.read_bytes()
            with np.load(path,allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files if not k.startswith('__')}
            if len(raw)!=m['inference_payload_bytes'] or hashlib.sha256(raw).hexdigest()!=m['payload_sha256']:errors.append(f'{seed}/{method}: payload bytes/hash mismatch')
            if method in ('serac_full','no_edit'):values=torch.tensor(a['values'])
            elif method in ('mirror_pca','native_pca'):values=torch.tensor(a['value_codes'])@torch.tensor(a['value_basis']).T
            else:values=torch.tensor(a['value_int8']).float()*torch.tensor(a['value_scales'])
            keys=torch.tensor(a['keys']);errs=[];routes=[];false=[]
            for e in range(TRAIN,N):
                for q in w['queries'][e]:
                    i,on=route(q,keys);routes.append(i==e and on);v=values[i] if on and method!='no_edit' else torch.zeros(D);errs.append(float(torch.mean((v-w['values'][e])**2)))
            for q in w['locality']:false.append(route(q,keys)[1])
            vals=[float(np.sqrt(np.mean(errs))),1-float(np.mean(routes)),float(np.mean(false))]
            ex=[m['heldout_value_rmse'],m['heldout_wrong_route_rate'],m['locality_false_trigger_rate']]
            diff=max(abs(x-y) for x,y in zip(vals,ex));mx=max(mx,diff)
            if diff>1e-7:errors.append(f'{seed}/{method}: replay difference {diff}')
        if (d/'mirror_pca_payload.npz').read_bytes()!=(d/'native_pca_payload.npz').read_bytes():errors.append(f'{seed}: native PCA bank not byte identical')
        else:checks.append({'seed':seed,'native_pca_sha256':hashlib.sha256((d/'mirror_pca_payload.npz').read_bytes()).hexdigest(),'replay':True})
    report={'experiment_id':'MA-475','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':mx,'fresh_accessed':False,'checks':checks,'errors':errors}
    (root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=replay(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
