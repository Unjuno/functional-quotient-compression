#!/usr/bin/env python3
"""Replay MA-476 heldout value/retrieval metrics from serialized banks."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from run import D,N,TRAIN,NLOCAL,world,route

def replay(root):
 errors=[];checks=[];mx=0.
 for seed in (47601,47602):
  d=root/f'runs/dev_{seed}';doc=json.loads((d/'metrics.json').read_text());w=world(seed)
  for method,m in doc['methods'].items():
   path=d/f'{method}_payload.npz';raw=path.read_bytes()
   with np.load(path,allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files if not k.startswith('__')}
   if len(raw)!=m['inference_payload_bytes'] or hashlib.sha256(raw).hexdigest()!=m['payload_sha256']:errors.append(f'{seed}/{method}: bytes/hash mismatch')
   if method=='serac_full':vals=torch.tensor(a['values'])
   elif method=='no_edit':vals=torch.zeros(N,D)
   elif method in ('mirror_pca','native_pca'):vals=torch.tensor(a['value_codes'])@torch.tensor(a['value_basis']).T
   elif method=='latent_int8':vals=(torch.tensor(a['latent_int8']).float()*torch.tensor(a['latent_scales']))@torch.tensor(a['value_basis']).T
   else:vals=torch.tensor(a['vq_centers'])[torch.tensor(a['vq_indices']).long()]@torch.tensor(a['value_basis']).T
   keys=torch.tensor(a['keys']);errs=[];routes=[];false=[]
   for e in range(TRAIN,N):
    for q in w['queries'][e]:
     i,on=route(q,keys);routes.append(i==e and on);v=vals[i] if on and method!='no_edit' else torch.zeros(D);errs.append(float(torch.mean((v-w['values'][e])**2)))
   for q in w['locality']:false.append(route(q,keys)[1])
   got=[float(np.sqrt(np.mean(errs))),1-float(np.mean(routes)),float(np.mean(false))];want=[m['heldout_value_rmse'],m['heldout_wrong_route_rate'],m['locality_false_trigger_rate']]
   diff=max(abs(a-b) for a,b in zip(got,want));mx=max(mx,diff)
   if diff>1e-7:errors.append(f'{seed}/{method}: metric mismatch {diff}')
  if (d/'mirror_pca_payload.npz').read_bytes()!=(d/'native_pca_payload.npz').read_bytes():errors.append(f'{seed}: native PCA payload alias failed')
  else:checks.append({'seed':seed,'native_pca_sha256':hashlib.sha256((d/'mirror_pca_payload.npz').read_bytes()).hexdigest(),'replay':True})
 report={'experiment_id':'MA-476','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':mx,'fresh_accessed':False,'checks':checks,'errors':errors};(root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=replay(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
