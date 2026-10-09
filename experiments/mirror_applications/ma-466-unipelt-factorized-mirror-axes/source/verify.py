#!/usr/bin/env python3
"""Replay MA-466 gate outputs and component ablations from payloads."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from run import C,TASKS,LAYERS,world,predict

def load(method,path):
    with np.load(path,allow_pickle=False) as z:a={k:np.array(z[k]) for k in z.files if not k.startswith('__')}
    comp={k:torch.tensor(a[k]) for k in ('base','lora_a','lora_b','adapter_u','adapter_v','prefix')}
    if method=='unipelt':s={'logits':torch.tensor(a['unipelt_gate_logits'])}
    elif method in ('mirror','native_cp'):s={'task':torch.tensor(a['task_factors']),'layer':torch.tensor(a['layer_factors']),'component':torch.tensor(a['component_factors'])}
    else:s={'logits':torch.tensor(a['shared_gate_logits'])}
    return s,comp

def verify(root):
    errors=[];checks=[];mx=0.
    for seed in (46601,46602):
        d=root/f'runs/dev_{seed}';m=json.loads((d/'metrics.json').read_text());_,_,data,_=world(seed)
        for method,rec in m['methods'].items():
            path=d/f'{method}_payload.npz';raw=path.read_bytes();s,comp=load(method,path)
            if len(raw)!=rec['payload_bytes'] or hashlib.sha256(raw).hexdigest()!=rec['payload_sha256']:errors.append(f'{seed}/{method}: payload checksum/size mismatch')
            scores=[];abl={c:[] for c in range(C)}
            with torch.no_grad():
                for t in range(TASKS):
                    for l in range(LAYERS):
                        x=data['xq'][t,l];y=data['yq'][t,l];p=predict(method,s,comp,t,l,x);scores.append(float(torch.sqrt(((p-y)**2).mean())))
                        if method=='mirror':
                            for c in range(C):q=predict(method,s,comp,t,l,x,c);abl[c].append(float(torch.sqrt(((q-y)**2).mean())))
            diff=max(abs(a-b) for a,b in zip(scores,rec['per_context_rmse']));mx=max(mx,diff)
            if diff>1e-7:errors.append(f'{seed}/{method}: metric replay {diff}')
            if method=='mirror':
                for c in range(C):
                    ad=abs(float(np.mean(abl[c]))-rec['ablation_rmse'][str(c)]);mx=max(mx,ad)
                    if ad>1e-7:errors.append(f'{seed}/mirror component {c}: ablation replay {ad}')
        if (d/'mirror_payload.npz').read_bytes()!=(d/'native_cp_payload.npz').read_bytes():errors.append(f'{seed}: CP native control not exact Mirror alias')
        else:checks.append({'seed':seed,'native_alias_sha256':hashlib.sha256((d/'mirror_payload.npz').read_bytes()).hexdigest(),'metric_and_ablation_replay':True})
    report={'experiment_id':'MA-466','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':mx,'fresh_accessed':False,'checks':checks,'errors':errors};(root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=verify(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
