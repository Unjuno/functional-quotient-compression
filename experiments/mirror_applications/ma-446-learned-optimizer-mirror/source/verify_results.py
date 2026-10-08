from __future__ import annotations
import csv,hashlib,json,math,statistics,sys
from pathlib import Path
import torch
from experiment import ROOT,make_world,make_episode,generator,decode_m

ROOT=Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(1)
    source=ROOT/'source'; summary=json.loads((source/'development_summary.json').read_text())
    raw=json.loads((source/'development_raw.json').read_text())
    methods=['hard_shared','mirror_sgd','mirror_adam','meta_sgd_mirror','lstm_mirror','full_adam','lstm_full']
    table={}
    with (ROOT/'RESULTS_CORE.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            key=(int(r['world_seed']),int(r['task_id']),r['method'],int(r['update']))
            assert key not in table
            table[key]=r
    maxpred=0.0; final_metric_delta=0.0; artifacts_checked=0
    for seed_s,items in summary['serialized_payloads'].items():
        seed=int(seed_s); w0,basis=make_world(seed); dev=make_episode(32,w0,basis,generator(seed+2_000_000))
        for method in methods:
            info=items[method]; loaded={}
            for kind,record in [('restartable',info),('inference',info['inference_payload'])]:
                path=ROOT/record['path']; blob=path.read_bytes()
                assert len(blob)==record['bytes'],(path,len(blob),record['bytes'])
                assert hashlib.sha256(blob).hexdigest()==record['sha256'],path
                obj=torch.load(path,map_location='cpu',weights_only=False)
                loaded[kind]=obj; artifacts_checked+=1
                if kind=='restartable':
                    assert obj['adaptation_updates']==4
                    assert obj['optimizer_step']==(4 if method in ('mirror_adam','full_adam') else None)
                else:
                    assert 'states' not in obj and 'learned_optimizer_state_dict' not in obj.get('shared',{})
            pred_restart=_predict(loaded['restartable'],dev)
            pred_inference=_predict(loaded['inference'],dev)
            diff=float((pred_restart-pred_inference).abs().max()); maxpred=max(maxpred,diff)
            assert diff==0.0,(seed,method,diff)
            mse=(pred_restart-dev['query_y'][:8]).square().mean(dim=(1,2)).tolist()
            for task_id,val in enumerate(mse):
                tableval=float(table[(seed,task_id,method,4)]['query_mse'])
                delta=abs(val-tableval); final_metric_delta=max(final_metric_delta,delta)
                assert delta<1e-7,(seed,method,task_id,val,tableval)
        for method in methods:
            rows=raw[seed_s][method]['task_query_mse_by_update']
            assert len(rows)==5 and all(len(x)==32 for x in rows)
            for task_id in range(32):
                for update in range(5):
                    row=table[(seed,task_id,method,update)]
                    assert abs(float(row['query_mse'])-rows[update][task_id])<1e-7
    assert len(table)==2240
    pooled={m:statistics.mean(summary['mean_query_mse_by_world'][str(s)][m] for s in summary['seeds']) for m in methods}
    assert all(abs(pooled[m]-summary['mean_query_mse_pooled'][m])<1e-10 for m in methods)
    assert not summary['fresh_accessed']
    # Recompute frozen gates, not just summary labels.
    for seed_s,g in summary['gates_by_world'].items():
        scores=summary['mean_query_mse_by_world'][seed_s]
        best=min(scores[x] for x in ('mirror_sgd','mirror_adam','meta_sgd_mirror'))
        mb=summary['storage_metrics'][seed_s]['lstm_mirror']['restartable_payload_bytes']
        fb=summary['storage_metrics'][seed_s]['lstm_full']['restartable_payload_bytes']
        assert g['learned_mirror_beats_simple_by_15pct']==(scores['lstm_mirror']<=.85*best)
        assert g['within_5pct_full_learned_quality']==(scores['lstm_mirror']<=1.05*scores['lstm_full'])
        assert g['at_least_30pct_payload_saving_vs_full_learned']==(mb<=.7*fb)
    assert summary['decision']=='FAIL' and not summary['all_pass']
    prior=source/'attempt_results/attempt_3/development_summary.json'
    old=json.loads(prior.read_text()) if prior.exists() else None
    replay_delta=None
    if old:
        replay_delta=max(abs(old['mean_query_mse_pooled'][m]-summary['mean_query_mse_pooled'][m]) for m in methods)
        assert replay_delta<1e-12
        assert old['selected_learning_rates']==summary['selected_learning_rates']
    out={'experiment_id':'MA-446','status':'VERIFIED_DEVELOPMENT_FAIL','artifacts_checked':artifacts_checked,'restartable_and_inference_payloads_checked':True,'max_inference_vs_restartable_prediction_difference':maxpred,'max_final_metric_csv_delta':final_metric_delta,'result_rows':len(table),'initial_run_quality_replay_max_delta':replay_delta,'fresh_accessed':False,'gate_recomputed':True}
    (source/'metric_replay.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

def _predict(obj,dev):
    if 'task_weights' in obj:
        return torch.einsum('bni,boi->bno',dev['query_x'][:8],obj['task_weights'])
    if 'task_codes' in obj:
        weights=decode_m(obj['shared']['w0'],obj['shared']['basis'],obj['task_codes'])
        return torch.einsum('bni,boi->bno',dev['query_x'][:8],weights)
    if 'states' in obj and 'params' in obj['states']:
        if obj['method'] in ('full_adam','lstm_full'): weights=obj['states']['params']
        else: weights=decode_m(obj['shared']['w0'],obj['shared']['basis'],obj['states']['params'])
        return torch.einsum('bni,boi->bno',dev['query_x'][:8],weights)
    return torch.einsum('bni,oi->bno',dev['query_x'][:8],obj['shared']['w0'])


if __name__=='__main__': main()
