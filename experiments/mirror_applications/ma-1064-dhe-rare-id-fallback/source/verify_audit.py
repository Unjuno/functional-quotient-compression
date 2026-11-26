#!/usr/bin/env python3
"""Replay MA-1064 saved payloads using only persisted training IDs/metadata."""
import hashlib,json
from pathlib import Path
import numpy as np, torch
import train

ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'runs'/'audit'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    train.load_freeze(); tr,_,audit,excluded=train.read_rows(True); results=json.loads((D/'results.json').read_text()); cut=int(results['selected_cutoff']); maxlog=maxmetric=0.0; rows={}
    expected=train.selected_ids(tr,cut)
    for method,seeds in results['methods'].items():
        rows[method]={}
        for seed,r in seeds.items():
            ck=D/f'{method}_{seed}.pt'; mp=D/'payload_metadata'/f'{method}_{seed}.json'; lp=D/f'{method}_{seed}.npz'; meta=json.loads(mp.read_text()); obj=torch.load(ck,map_location='cpu',weights_only=False); saved=np.load(lp)
            assert saved.files==['logits']; stored=saved['logits']; assert len(stored)==len(audit)
            if method=='full_tables': model=train.FullTables(); model.load_state_dict(obj['state_dict'])
            elif method=='dhe_shared': model=train.load_base(obj['state_dict'])
            else:
                ids=[meta['selected_user_ids'],meta['selected_item_ids']]
                assert ids==expected
                base_state={k.removeprefix('base.'):v for k,v in obj['state_dict'].items() if k.startswith('base.')}
                base=train.load_base(base_state); model=train.PrivateView(base,method,ids); model.load_state_dict(obj['state_dict'])
            replay_metrics,pred=train.evaluate(model,audit,tr); delta=float(np.max(np.abs(stored-pred))); maxlog=max(maxlog,delta)
            dm=max(abs(replay_metrics['overall'][k]-r['metrics']['overall'][k]) for k in ('auc','logloss'))
            dm=max(dm,abs(replay_metrics['slices']['item_freq_le5']['auc']-r['metrics']['slices']['item_freq_le5']['auc'])); maxmetric=max(maxmetric,float(dm))
            payload_bytes=ck.stat().st_size+mp.stat().st_size
            assert sha(ck)==r['checkpoint_sha256'] and sha(mp)==r['metadata_sha256'] and sha(lp)==r['logits_sha256']
            assert payload_bytes==r['payload_bytes'] and delta<=1e-6 and dm<=1e-8
            rows[method][seed]={'payload_bytes':payload_bytes,'max_logit_abs_diff':delta,'max_metric_abs_diff':float(dm),'checkpoint_sha256':sha(ck),'metadata_sha256':sha(mp),'logits_sha256':sha(lp)}
    out={'experiment_id':'MA-1064','status':'PASS','fresh_audit_rows':len(audit),'legacy_MA1063_audit_rows_excluded':excluded,'train_fresh_pair_overlap':False,'audit_row_ids_or_labels_persisted':False,'training_ID_addresses_in_private_payload_metadata':True,'training_ID_lists_verified_against_train_only_frequency':True,'payload_models_replayed':sum(map(len,rows.values())),'max_logit_abs_difference':maxlog,'max_metric_abs_difference':maxmetric,'tolerances':{'logit_abs':1e-6,'metric_abs':1e-8},'per_model':rows}
    (D/'verification.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
