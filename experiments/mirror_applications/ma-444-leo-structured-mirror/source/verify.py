#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(Path(__file__).resolve().parent));import run
errors=[]; checks=[]
for seed in (44401,44402):
    d=ROOT/'runs'/f'dev_{seed}'; data=json.loads((d/'metrics.json').read_text()); methods=data['methods']
    archives={}
    try:
        for name,m in methods.items():
            raw=(d/f'{name}_payload.npz').read_bytes()
            if len(raw)!=m['payload_bytes']:errors.append(f'{seed}/{name}: byte mismatch')
            if hashlib.sha256(raw).hexdigest()!=m['payload_sha256']:errors.append(f'{seed}/{name}: hash mismatch')
            archives[name]=np.load(d/f'{name}_payload.npz')
        a,b=archives['mirror'],archives['native_givens']
        for k in ('shared_base','task_codes'):
            if not np.array_equal(a[k],b[k]):errors.append(f'{seed}: mirror/native {k} differs')
        if methods['mirror']['payload_sha256']!=methods['native_givens']['payload_sha256']:errors.append(f'{seed}: native Givens payload is not byte-identical')
        rng=run.rng_for(seed+901); scores={name:[] for name in methods}
        for ix in range(16):
            _,x,y=run.sample_task(rng,run.SUPPORT+run.QUERY);xq,yq=x[run.SUPPORT:],y[run.SUPPORT:]
            for name,ar in archives.items():
                if name in ('mirror','native_givens'):w=run.rotate(torch.from_numpy(ar['shared_base'][0]),torch.from_numpy(ar['task_codes'][ix]))
                elif name=='leo':
                    base=torch.from_numpy(ar['shared_base'][0]);z=torch.from_numpy(ar['task_codes'][ix])
                    state=[torch.from_numpy(ar[f'decoder_{key}']) for key in ('w1','b1','w2','b2')];w=run.decode(name,base,z,state)
                elif name=='rank3':w=torch.from_numpy(ar['shared_base'][0])+torch.from_numpy(ar['basis'])@torch.from_numpy(ar['task_codes'][ix])
                elif name=='full':w=torch.from_numpy(ar['task_vectors'][ix])
                elif name=='independent':w=torch.from_numpy(ar['task_vectors'][ix])
                else:w=torch.from_numpy(ar['shared_base'][0])
                scores[name].append(float(torch.sqrt(run.loss(xq,yq,w))))
        for name,vals in scores.items():
            if abs(float(np.mean(vals))-methods[name]['query_rmse'])>1e-6:errors.append(f'{seed}/{name}: saved output RMSE mismatch')
        checks.append({'seed':seed,'mirror_bytes':methods['mirror']['payload_bytes'],'native_bytes':methods['native_givens']['payload_bytes'],'mirror_native_rmse_difference':abs(methods['mirror']['query_rmse']-methods['native_givens']['query_rmse'])})
    finally:
        for ar in archives.values():ar.close()
fresh=list(ROOT.joinpath('runs').glob('fresh_*'))
if fresh:errors.append('fresh runs opened')
report={'experiment_id':'MA-444','checks':checks,'serialization_roundtrip_checked':len(checks)==2,'payload_replay_checked':len(checks)==2 and not any('RMSE mismatch' in x for x in errors),'fresh_accessed':bool(fresh),'errors':errors}
(ROOT/'verification_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,sort_keys=True));raise SystemExit(bool(errors))
