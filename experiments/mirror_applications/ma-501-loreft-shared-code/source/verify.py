#!/usr/bin/env python3
"""Replay MA-501 metrics from serialized intervention payloads."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np,torch
import run

def verify(root):
    errors=[];maxdiff=0.;numeric=('all_task_relative_rmse','heldout_task_relative_rmse','active_compute_proxy_per_example','active_compute_proxy_all_eval_examples','m_causal_max_output_change')
    for seed in (50101,50102):
        sd=root/'runs'/f'dev_{seed}';doc=json.loads((sd/'metrics.json').read_text())
        for rho in run.RHOS:
            w=run.world(seed,rho);rd=sd/f'rho_{rho:g}';rkey=str(rho)
            for method in run.METHODS:
                obs=doc['rhos'][rkey][method];p=rd/f'{method}_payload.npz';raw=p.read_bytes()
                if len(raw)!=obs['inference_payload_bytes'] or hashlib.sha256(raw).hexdigest()!=obs['payload_sha256']:errors.append(f'{seed}/{rho}/{method}: bytes/hash mismatch')
                with np.load(p,allow_pickle=False) as z:
                    objs={k:torch.from_numpy(np.array(z[k])) for k in z.files if k!='schema_json'}
                replay=run.score(method,objs,w)
                for key in numeric:
                    delta=abs(float(replay[key])-float(obs[key]));maxdiff=max(maxdiff,delta)
                    if delta>1e-6:errors.append(f'{seed}/{rho}/{method}/{key}: replay mismatch {delta}')
                if method=='full_matrix' and replay['heldout_task_relative_rmse']>1e-4:errors.append(f'{seed}/{rho}: full matrix upper invalid')
            if (rd/'shared_mirror_payload.npz').read_bytes()!=(rd/'native_shared_code_payload.npz').read_bytes():errors.append(f'{seed}/{rho}: native shared-code exact alias failed')
    report={'experiment_id':'MA-501','serialization_roundtrip_checked':not errors,'metric_replay_checked':not errors,'max_metric_difference':maxdiff,'full_matrix_upper_checked':not errors,'native_shared_code_alias_checked':not errors,'fresh_accessed':False,'errors':errors}
    (root/'verification_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=verify(a.root);print(json.dumps(r,sort_keys=True));raise SystemExit(bool(r['errors']))
if __name__=='__main__':main()
