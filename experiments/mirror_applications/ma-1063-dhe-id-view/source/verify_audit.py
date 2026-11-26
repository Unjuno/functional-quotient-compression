#!/usr/bin/env python3
"""Replay frozen MA-1063 payloads and verify aggregate metrics/byte manifests."""
import hashlib,json
from pathlib import Path
import numpy as np, torch
import train

ROOT=Path(__file__).resolve().parents[1]; RUNS=ROOT/'runs'; D=RUNS/'audit'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    freeze=train.load_freeze(); assert train.sha(RUNS/'DEVELOPMENT_SELECTION.json')==freeze['development_selection_sha256']; tr,_,audit=train.split(include_audit=True); maxu,maxi=6040,3952
    y=(audit[:,2]>=4).astype(np.float32); actual=json.loads((D/'results.json').read_text()); check={}; maxlog=maxmetric=0.0; pairs_train={(int(u),int(i)) for u,i,_ in tr}; pairs_audit={(int(u),int(i)) for u,i,_ in audit}
    assert not (pairs_train & pairs_audit)
    for method,seeds in actual.items():
        check[method]={}
        for seed,v in seeds.items():
            stem=f'{method}_{seed}'; ck=D/f'{stem}.pt'; mp=D/f'{stem}.json'; stored=np.load(D/f'{stem}.npz')['logits']
            obj=torch.load(ck,map_location='cpu',weights_only=False); model=train.Model(method,maxu,maxi); model.load_state_dict(obj['state_dict'])
            model.hash_u=torch.from_numpy(np.concatenate([np.zeros((1,train.K),np.float32),train.dense_hash(np.arange(1,maxu+1),0)],axis=0))
            model.hash_i=torch.from_numpy(np.concatenate([np.zeros((1,train.K),np.float32),train.dense_hash(np.arange(1,maxi+1),1000000)],axis=0))
            _,replay=train.eval_model(model,audit,(maxu,maxi)); delta=float(np.max(np.abs(stored-replay))); maxlog=max(maxlog,delta)
            metric_delta=max(abs(train.metric(stored,y)[k]-v['metrics'][k]) for k in ('auc','logloss')); maxmetric=max(maxmetric,float(metric_delta))
            size=ck.stat().st_size+mp.stat().st_size
            assert size==v['complete_payload_bytes']
            assert delta<=1e-6 and metric_delta<=1e-8
            assert sha(ck)==v['checkpoint_sha256'] and sha(mp)==v['metadata_sha256']
            check[method][seed]={'complete_payload_bytes':size,'max_logit_abs_diff':delta,'metric_max_abs_diff':float(metric_delta)}
    out={'experiment_id':'MA-1063','status':'PASS','audit_rows':int(len(audit)),'train_audit_pair_overlap':False,'audit_labels_or_ids_persisted':False,'payload_models_replayed':sum(map(len,check.values())),'max_logit_abs_difference':maxlog,'max_metric_difference':maxmetric,'replay_tolerances':{'logit_abs':1e-6,'metric_abs':1e-8},'per_model':check}
    (D/'verification.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
