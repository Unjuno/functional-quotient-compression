#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run
errors=[];checks=[]
for seed in (45301,45302):
    rd=ROOT/'runs'/f'dev_{seed}'
    d=json.loads((rd/'metrics.json').read_text());methods=d['methods'];archives={}
    try:
        for name,values in methods.items():
            raw=(rd/f'{name}_payload.npz').read_bytes()
            if len(raw)!=values['payload_bytes']:errors.append(f'{seed}/{name}: byte mismatch')
            if hashlib.sha256(raw).hexdigest()!=values['payload_sha256']:errors.append(f'{seed}/{name}: hash mismatch')
            archives[name]=np.load(rd/f'{name}_payload.npz')
        if methods['mirror']['payload_sha256']!=methods['native_givens']['payload_sha256']:
            errors.append(f'{seed}: native Givens payload mismatch')
        for key in ('router_table','function_contexts','physical_blocks','role_codes'):
            if not np.array_equal(archives['mirror'][key],archives['native_givens'][key]):
                errors.append(f'{seed}: Mirror/native {key} mismatch')
        true=run.teacher(seed);rng=np.random.default_rng(seed+8801);scores={name:[] for name in methods}
        for module in range(2):
            for role in range(2):
                x,y=run.data(rng,run.NEVAL,module,role,true)
                for name,ar in archives.items():
                    if name=='shared':w=torch.from_numpy(ar['shared_block'])
                    elif name=='independent':w=torch.from_numpy(ar['logical_blocks'][module,role])
                    elif name=='routing':w=torch.from_numpy(ar['physical_blocks'][module])
                    elif name in ('mirror','native_givens'):
                        w=run.rotate(torch.from_numpy(ar['physical_blocks'][module]),torch.from_numpy(ar['role_codes'][role]))
                    elif name=='rank2':
                        w=torch.from_numpy(ar['physical_blocks'][module])+torch.from_numpy(ar['residual_basis'])@torch.from_numpy(ar['role_codes'][role])
                    else:w=torch.from_numpy(ar['physical_blocks'][module])+torch.from_numpy(ar['role_vectors'][role])
                    scores[name].append(float(torch.sqrt(run.mse(x,y,w))))
        for name,values in scores.items():
            if abs(float(np.mean(values))-methods[name]['query_rmse'])>1e-6:
                errors.append(f'{seed}/{name}: inference replay mismatch')
        checks.append({'seed':seed,'mirror_bytes':methods['mirror']['payload_bytes'],'native_bytes':methods['native_givens']['payload_bytes'],'mirror_native_rmse_difference':abs(methods['mirror']['query_rmse']-methods['native_givens']['query_rmse'])})
    finally:
        for ar in archives.values():ar.close()
fresh=list((ROOT/'runs').glob('fresh_*'))
if fresh:errors.append('fresh seeds opened')
report={'experiment_id':'MA-453','checks':checks,'serialization_roundtrip_checked':len(checks)==2,'metric_replay_checked':len(checks)==2 and not any('inference replay mismatch' in e for e in errors),'fresh_accessed':bool(fresh),'errors':errors}
(ROOT/'verification_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,sort_keys=True))
raise SystemExit(bool(errors))
