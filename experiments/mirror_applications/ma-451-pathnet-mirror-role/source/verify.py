#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(Path(__file__).resolve().parent));import run
errors=[];checks=[]
for seed in (45101,45102):
    rd=ROOT/'runs'/f'dev_{seed}';d=json.loads((rd/'metrics.json').read_text());m=d['methods'];archives={}
    try:
        for name,v in m.items():
            raw=(rd/f'{name}_payload.npz').read_bytes()
            if len(raw)!=v['payload_bytes']:errors.append(f'{seed}/{name}: byte mismatch')
            if hashlib.sha256(raw).hexdigest()!=v['payload_sha256']:errors.append(f'{seed}/{name}: hash mismatch')
            archives[name]=np.load(rd/f'{name}_payload.npz')
        if m['mirror']['payload_sha256']!=m['native_givens']['payload_sha256']:errors.append(f'{seed}: native path-view payload not byte-identical')
        for key in ('role_codes','layer_a','layer_b','route_ids'):
            if not np.array_equal(archives['mirror'][key],archives['native_givens'][key]):errors.append(f'{seed}: native view differs in {key}')
        atrue,btrue=run.teacher(seed);rng=np.random.default_rng(seed+8801);scores={k:[] for k in m}
        for task,(path,angle) in enumerate((r,a) for r in run.ROUTES for a in run.ROLE_ANGLES):
            x,y=run.sample_task(rng,run.SUPPORT+run.QUERY,path,angle,atrue,btrue);xq,yq=x[run.SUPPORT:],y[run.SUPPORT:]
            for name,ar in archives.items():
                if name=='independent':w=torch.from_numpy(ar['task_vectors'][task])
                else:
                    aa=torch.from_numpy(ar['layer_a']);bb=torch.from_numpy(ar['layer_b']);route=tuple(int(v) for v in ar['route_ids'][task])
                    base=aa[route[0]]+bb[route[1]]
                    if name in ('mirror','native_givens'):w=run.rotate(base,torch.from_numpy(ar['role_codes'][task]))
                    elif name=='rank2':w=base+torch.from_numpy(ar['rank2_route_bases'])[route[0]*2+route[1]]@torch.from_numpy(ar['adapter_codes'][task])
                    else:w=base
                scores[name].append(float(torch.sqrt(run.mse(xq,yq,w))))
        for name,vals in scores.items():
            if abs(float(np.mean(vals))-m[name]['query_rmse'])>1e-6:errors.append(f'{seed}/{name}: metric replay mismatch')
        checks.append({'seed':seed,'mirror_bytes':m['mirror']['payload_bytes'],'native_bytes':m['native_givens']['payload_bytes'],'mirror_native_rmse_difference':abs(m['mirror']['query_rmse']-m['native_givens']['query_rmse'])})
    finally:
        for ar in archives.values():ar.close()
fresh=list((ROOT/'runs').glob('fresh_*'))
if fresh:errors.append('fresh data opened')
out={'experiment_id':'MA-451','checks':checks,'serialization_roundtrip_checked':len(checks)==2,'metric_replay_checked':len(checks)==2 and not any('metric replay mismatch' in e for e in errors),'fresh_accessed':bool(fresh),'errors':errors}
(ROOT/'verification_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,sort_keys=True));raise SystemExit(bool(errors))
