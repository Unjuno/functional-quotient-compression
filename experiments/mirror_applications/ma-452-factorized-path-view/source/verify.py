#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(Path(__file__).resolve().parent));import run
errors=[];checks=[]
for seed in (45201,45202):
 rd=ROOT/'runs'/f'dev_{seed}';d=json.loads((rd/'metrics.json').read_text());methods=d['methods'];ars={}
 try:
  for name,m in methods.items():
   raw=(rd/f'{name}_payload.npz').read_bytes()
   if len(raw)!=m['payload_bytes']:errors.append(f'{seed}/{name}: byte mismatch')
   if hashlib.sha256(raw).hexdigest()!=m['payload_sha256']:errors.append(f'{seed}/{name}: hash mismatch')
   ars[name]=np.load(rd/f'{name}_payload.npz')
  if methods['mirror']['payload_sha256']!=methods['native_givens']['payload_sha256']:errors.append(f'{seed}: native Givens payload mismatch')
  for k in ('layer_a','layer_b','route_ids','role_ids','role_codebook'):
   if not np.array_equal(ars['mirror'][k],ars['native_givens'][k]):errors.append(f'{seed}: mirror/native {k} mismatch')
  at,bt=run.teacher(seed);rng=np.random.default_rng(seed+8801);scores={k:[] for k in methods}
  for idx,pair in enumerate(run.TEST_PAIRS):
   x,y=run.make_data(rng,run.NSUPPORT+run.NQUERY,pair,at,bt);xs,xq=x[:run.NSUPPORT],x[run.NSUPPORT:];ys,yq=y[:run.NSUPPORT],y[run.NSUPPORT:]
   for name,ar in ars.items():
    if name=='independent':w=torch.linalg.lstsq(xs,ys).solution
    else:
     route=tuple(int(v) for v in ar['route_ids'][idx]);role=int(ar['role_ids'][idx]);aa=torch.from_numpy(ar['layer_a']);bb=torch.from_numpy(ar['layer_b']);base=aa[route[0]]+bb[route[1]]
     if name in ('mirror','native_givens'):w=run.rotate(base,torch.from_numpy(ar['role_codebook'][role]))
     elif name=='rank2':w=base+torch.from_numpy(ar['rank2_basis'])@torch.from_numpy(ar['role_codebook'][role])
     elif name=='role_bias':w=base+torch.from_numpy(ar['role_vectors'][role])
     else:w=base
    scores[name].append(float(torch.sqrt(run.mse(xq,yq,w))))
  for name,values in scores.items():
   if abs(float(np.mean(values))-methods[name]['query_rmse'])>1e-6:errors.append(f'{seed}/{name}: output replay mismatch')
  checks.append({'seed':seed,'mirror_bytes':methods['mirror']['payload_bytes'],'native_bytes':methods['native_givens']['payload_bytes'],'mirror_native_rmse_difference':abs(methods['mirror']['query_rmse']-methods['native_givens']['query_rmse'])})
 finally:
  for ar in ars.values():ar.close()
fresh=list((ROOT/'runs').glob('fresh_*'))
if fresh:errors.append('fresh seeds opened')
out={'experiment_id':'MA-452','checks':checks,'serialization_roundtrip_checked':len(checks)==2,'metric_replay_checked':len(checks)==2 and not any('replay mismatch' in e for e in errors),'fresh_accessed':bool(fresh),'errors':errors}
(ROOT/'verification_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,sort_keys=True));raise SystemExit(bool(errors))
