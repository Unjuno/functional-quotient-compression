from __future__ import annotations
import csv, hashlib, json, statistics, sys, time
from pathlib import Path
import torch
from experiment import (ROOT, LRS, make_world, run_world, query_mse_m, query_mse_w,
                        decode_m, make_payload, make_inference_payload, save_library, save_inference_payload, tensor_bytes,
                        compute_proxy, RANK, HIDDEN, STEPS)

ART=ROOT/'source'/'artifacts'
ART.mkdir(parents=True,exist_ok=True)
SEEDS=(44601,44602)


def serialize_world(w:dict, selected:dict[str,str]) -> dict:
    seed=w['world_seed']; w0=w['w0']; basis=w['basis']; dev=w['dev']; methods=w['methods']
    method_map={
      'hard_shared':('hard_shared',None,None,None),
      'mirror_sgd':(f"sgd_{selected['sgd']:g}_mirror",None,None,selected['sgd']),
      'mirror_adam':(f"adam_{selected['adam']:g}_mirror",None,None,selected['adam']),
      'meta_sgd_mirror':('meta_sgd_mirror',None,(w['meta_init'],w['meta_alpha']),None),
      'lstm_mirror':('lstm_mirror',w['lstm_mirror'],None,None),
      'full_adam':(f"adam_{selected['full_adam']:g}_full",None,None,selected['full_adam']),
      'lstm_full':('lstm_full',w['lstm_full'],None,None),
    }
    infos={}
    for name,(key,model,meta,lr) in method_map.items():
        res=methods[key]
        payload=make_payload(name,seed,w0,basis,res,dev,learned_model=model,meta_sgd=meta,lr=lr)
        if name=='hard_shared':
            pred=torch.einsum('bni,oi->bno',dev['query_x'][:8],w0)
        elif name in ('full_adam','lstm_full'):
            pred=torch.einsum('bni,boi->bno',dev['query_x'][:8],res['params'][:8])
        else:
            pred=torch.einsum('bni,boi->bno',dev['query_x'][:8],decode_m(w0,basis,res['params'][:8]))
        file=ART/f'seed{seed}_{name}_K8.pt'
        info=save_library(file,payload,dev,w0,basis,pred)
        inference=make_inference_payload(name,seed,w0,basis,res)
        inference_file=ART/f'seed{seed}_{name}_K8_inference.pt'
        inference_info=save_inference_payload(inference_file,inference,dev,pred)
        info['inference_payload']=inference_info
        # Per-task serialized state bytes include optimizer state, not shared bases/optimizer weights.
        per_task=[]
        if name=='hard_shared':
            per_task=[0]*8
        else:
            for i in range(8):
                state={k:(v[i] if isinstance(v,torch.Tensor) and v.ndim>0 and v.shape[0]>=8 else v)
                       for k,v in payload['states'].items()}
                per_task.append(len(tensor_bytes({'schema':'MA-446/task-state-v1','task_id':i,'state':state})))
        infos[name]={**info,'per_task_state_bytes_mean':statistics.mean(per_task),'per_task_state_bytes':per_task,
                     'curve_mean':res['curve']}
    return infos


def main():
    torch.set_num_threads(1)
    begin=time.perf_counter()
    worlds=[]
    for seed in SEEDS:
        worlds.append(run_world(seed,ART))
    # One learning rate per optimizer, chosen on pooled development outcomes.
    selected={}
    for kind in ('sgd','adam'):
        selected[kind]=min(LRS,key=lambda lr:statistics.mean(w['methods'][f'{kind}_{lr:g}_mirror']['curve'][-1] for w in worlds))
    selected['full_adam']=min(LRS,key=lambda lr:statistics.mean(w['methods'][f'adam_{lr:g}_full']['curve'][-1] for w in worlds))
    artifacts={str(w['world_seed']):serialize_world(w,selected) for w in worlds}
    chosen_names=['hard_shared','mirror_sgd','mirror_adam','meta_sgd_mirror','lstm_mirror','full_adam','lstm_full']
    traces={}; rows=[]
    for w in worlds:
        seed=w['world_seed']; methods=w['methods']; traces[str(seed)]={}
        keys={'hard_shared':'hard_shared','mirror_sgd':f"sgd_{selected['sgd']:g}_mirror",'mirror_adam':f"adam_{selected['adam']:g}_mirror",'meta_sgd_mirror':'meta_sgd_mirror','lstm_mirror':'lstm_mirror','full_adam':f"adam_{selected['full_adam']:g}_full",'lstm_full':'lstm_full'}
        for name in chosen_names:
            res=methods[keys[name]]
            # Adaptation implementations return the mean curve over 32 tasks.
            traces[str(seed)][name]={'mean_query_mse_by_update':res['curve'],'task_query_mse_by_update':res['task_curves'],'task_count':32,'wall_seconds_per_task':res.get('wall_seconds_per_task',0)}
            is_mirror=name in ('mirror_sgd','mirror_adam','meta_sgd_mirror','lstm_mirror')
            if name=='hard_shared': nparam=0
            elif is_mirror: nparam=RANK
            else: nparam=12
            optname='lstm_mirror' if name=='lstm_mirror' else ('lstm_full' if name=='lstm_full' else ('adam' if 'adam' in name else name))
            mac=compute_proxy(optname,nparam,is_mirror)
            storage=artifacts[str(seed)][name]
            for tid in range(32):
                for step,mse in enumerate(res['curve']):
                    rows.append({'world_seed':seed,'task_id':tid,'method':name,'update':step,'query_mse':f"{res['task_curves'][step][tid]:.10g}",
                                 'restartable_payload_bytes':storage['bytes'] if tid<8 else '',
                                 'inference_payload_bytes':storage['inference_payload']['bytes'] if tid<8 else '',
                                 'per_task_state_bytes':storage['per_task_state_bytes'][tid] if tid<8 and tid<len(storage['per_task_state_bytes']) else '',
                                 'optimizer_compute_proxy':0 if name=='hard_shared' else mac,'wall_seconds_per_task':f"{res.get('wall_seconds_per_task',0):.8g}"})
    summary={
      'experiment_id':'MA-446','status':'DEVELOPMENT_COMPLETE_PENDING_GATE','seeds':list(SEEDS),
      'selected_learning_rates':selected,'development_tasks_per_world':32,'support_examples_per_update':16,'updates':4,'query_examples':128,
      'meta_training_outer_updates':256,'meta_training_batch_tasks':16,'meta_training_wall_seconds':{str(w['world_seed']):w['meta_train_seconds'] for w in worlds},
      'meta_training_final_outer_loss':{str(w['world_seed']):w['meta_train_loss_final'] for w in worlds},
      'serialized_payloads':artifacts,'storage_metrics':{seed:{name:{'inference_payload_bytes':x['inference_payload']['bytes'],'restartable_payload_bytes':x['bytes'],'inference_sha256':x['inference_payload']['sha256'],'restartable_sha256':x['sha256']} for name,x in items.items()} for seed,items in artifacts.items()},'mean_query_mse_by_world':{seed:{name:traces[seed][name]['mean_query_mse_by_update'][-1] for name in chosen_names} for seed in traces},
      'mean_query_mse_pooled':{name:statistics.mean(traces[str(w['world_seed'])][name]['mean_query_mse_by_update'][-1] for w in worlds) for name in chosen_names},
      'compute_proxy_per_task':{name:(0 if name=='hard_shared' else compute_proxy('lstm_mirror' if name=='lstm_mirror' else ('lstm_full' if name=='lstm_full' else ('adam' if 'adam' in name else name)),RANK if name in ('mirror_sgd','mirror_adam','meta_sgd_mirror','lstm_mirror') else (0 if name=='hard_shared' else 12),name in ('mirror_sgd','mirror_adam','meta_sgd_mirror','lstm_mirror'))) for name in chosen_names},
      'torch_num_threads':torch.get_num_threads(),'total_run_wall_seconds':time.perf_counter()-begin,'fresh_accessed':False
    }
    # Predeclared gates, applied independently per world using the pooled-selected controls.
    gates={}
    for seed in map(str,SEEDS):
        t=traces[seed]; best_simple=min(t[k]['mean_query_mse_by_update'][-1] for k in ('mirror_sgd','mirror_adam','meta_sgd_mirror'))
        mirror=t['lstm_mirror']['mean_query_mse_by_update'][-1]
        full=t['lstm_full']['mean_query_mse_by_update'][-1]
        bm=artifacts[seed]['lstm_mirror']['bytes']; bf=artifacts[seed]['lstm_full']['bytes']
        gates[seed]={'learned_mirror_beats_simple_by_15pct':mirror<=0.85*best_simple,
                     'within_5pct_full_learned_quality':mirror<=1.05*full,
                     'at_least_30pct_payload_saving_vs_full_learned':bm<=0.70*bf,
                     'best_simple_mse':best_simple,'mirror_lstm_mse':mirror,'full_lstm_mse':full,
                     'mirror_bytes':bm,'full_bytes':bf}
    summary['gates_by_world']=gates
    summary['all_pass']=all(all(v for k,v in checks.items() if isinstance(v,bool)) for checks in gates.values())
    summary['decision']='PROMISING' if summary['all_pass'] else 'FAIL'
    (ROOT/'source'/'development_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'source'/'development_raw.json').write_text(json.dumps(traces,indent=2)+'\n')
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['world_seed','task_id','method','update','query_mse','inference_payload_bytes','restartable_payload_bytes','per_task_state_bytes','optimizer_compute_proxy','wall_seconds_per_task'],lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    return summary

if __name__=='__main__':
    result=main()
    print(json.dumps({'decision':result['decision'],'selected_learning_rates':result['selected_learning_rates'],'mean_query_mse_pooled':result['mean_query_mse_pooled'],'gates_by_world':result['gates_by_world'],'total_run_wall_seconds':result['total_run_wall_seconds']},indent=2))
