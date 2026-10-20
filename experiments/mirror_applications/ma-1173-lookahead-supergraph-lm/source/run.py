"""MA-1173 preregistered CPU experiment: structure cuts as optimization interventions.

Usage:
  PYTHONPATH=source python source/run.py train [--world balanced|shortcut]
  PYTHONPATH=source python source/run.py audit

HOLD-OUT SAFETY: train never constructs fresh or fresh_ood. Audit may only
run after TRAIN_COMPLETE.json exists and all 54 checkpoints are present.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import TinyCausalLM, EDGES, D, NODES, VIEWS, STEPS, initialize_model
from data import corpus, batch_schedule

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
CHECKPOINTS=ROOT/'checkpoints'
PROTOCOL=ROOT/'PROTOCOL.json'
CFG=json.loads(PROTOCOL.read_text())
SEEDS=CFG['synthetic_causal_language_data']['seeds']
WORLDS=CFG['synthetic_causal_language_data']['worlds']
METHODS=CFG['models']
UPDATES=CFG['training']['updates']
EVENTS=CFG['cut_policy']['intervention_updates']
BATCH=CFG['training']['batch_size']
LOOK=CFG['cut_policy']['lookahead_steps_per_candidate']
torch.set_num_threads(1)


def optimizer(m):
    return torch.optim.AdamW(m.parameters(),lr=CFG['training']['learning_rate'],weight_decay=CFG['training']['weight_decay'])


def step(m, opt, samples, idx):
    z=samples[idx]
    opt.zero_grad(set_to_none=True)
    loss=m.model_loss(z[:,:-1],z[:,1:])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),CFG['training']['gradient_clip_norm'])
    opt.step()
    return float(loss.detach())


@torch.inference_mode()
def metrics(m,ds,eval_batch=128):
    m.eval();loss_sum=0.;qsum=0.;correct=0;total=0
    for chunk in ds.split(eval_batch):
        x=chunk[:,:-1];y=chunk[:,1:]; pred=m(x)
        ce=F.cross_entropy(pred.flatten(0,1),y.reshape(-1),reduction='none').reshape(y.shape)
        loss_sum+=float(ce.sum()); qsum+=float(ce[:,-1].sum());correct+=int((pred[:,-1].argmax(-1)==y[:,-1]).sum());total+=len(chunk)
    ntokens=ds.shape[1]-1
    nll=loss_sum/(total*ntokens);q=qsum/total
    m.train()
    return {'plain_nll':nll,'query_nll':q,'query_accuracy':correct/total,'weighted_loss':nll+2*q}


def shortlist(m,seed,world,event):
    indices=torch.nonzero(m.graph.active).flatten().numpy()
    salt=0 if world=='balanced' else 300001
    rng=np.random.default_rng(170000+seed*13+event*31+salt)
    indices=rng.permutation(indices).tolist()
    assert len(indices)>=8
    return [tuple(sorted(indices[2*i:2*i+2])) for i in range(4)]


def clone_continue_score(m,opt,remove,train,schedule,start,dev):
    """All clones see identical 8 minibatches and optimizer state.

    There is no RNG in a forward pass and no fresh data are consumed here.
    """
    trial=copy.deepcopy(m)
    trialopt=optimizer(trial)
    trialopt.load_state_dict(copy.deepcopy(opt.state_dict()))
    if remove:
        for j in remove:trial.graph.cut(j)
    for t in range(start,start+LOOK):step(trial,trialopt,train,schedule[t])
    return metrics(trial,dev)['weighted_loss']


def intervene(m,opt,method,seed,world,event,step_i,train,schedule,dev):
    groups=shortlist(m,seed,world,event)
    active_before=m.graph.active_count()
    dev_before=metrics(m,dev)['weighted_loss']
    removed=();candidate_scores={};reference=None;probe_updates=0
    if method in ('mirror_random',):
        removed=groups[0]
    elif method=='mirror_low_gate':
        gates=m.graph.logits.detach().sigmoid()
        removed=min(groups,key=lambda pair:float(gates[list(pair)].sum()))
    else:
        assert method in ('mirror_lookahead','mirror_guarded','direct_tanh_lookahead','affine_lookahead')
        reference=clone_continue_score(m,opt,(),train,schedule,step_i,dev)
        probe_updates+=LOOK
        for pair in groups:
            s=clone_continue_score(m,opt,pair,train,schedule,step_i,dev)
            candidate_scores[','.join(map(str,pair))]=s
            probe_updates+=LOOK
        removed=min(groups,key=lambda g:candidate_scores[','.join(map(str,g))])
        if method=='mirror_guarded' and candidate_scores[','.join(map(str,removed))]>=reference-0.0005:
            removed=()
    for e in removed:m.graph.cut(e)
    dev_after=metrics(m,dev)['weighted_loss']
    return {'update':step_i,'event':event,'policy':method,'candidate_pairs':[list(g) for g in groups],
            'removed':list(removed),'active_before':active_before,'active_after':m.graph.active_count(),
            'dev_before_cut':dev_before,'dev_immediate_after_cut':dev_after,
            'lookahead_keep_future':reference,'lookahead_candidates':candidate_scores,
            'probe_optimizer_steps':probe_updates}


def variant_for(method):
    if method=='native_tied':return 'native'
    if method=='direct_tanh_lookahead':return 'direct_tanh'
    if method=='affine_lookahead':return 'affine'
    return 'mirror'


def train_one(world,seed,method):
    t0=time.perf_counter()
    model=initialize_model(seed,variant_for(method))
    opt=optimizer(model)
    train=corpus(world,'train',seed,CFG['synthetic_causal_language_data']['train'])
    dev=corpus(world,'dev',seed,CFG['synthetic_causal_language_data']['development'])
    max_updates=UPDATES + (120 if method=='mirror_keep_extra' else 0)
    schedule=batch_schedule(len(train),seed,max_updates+100,BATCH)
    events=[]; nprobes=0;loss0=None
    for t in range(max_updates):
        if t in EVENTS and method not in ('mirror_keep','mirror_keep_extra','native_tied'):
            event=EVENTS.index(t)
            record=intervene(model,opt,method,seed,world,event,t,train,schedule,dev)
            events.append(record);nprobes+=record['probe_optimizer_steps']
        l=step(model,opt,train,schedule[t]);
        if t==0:loss0=l
    dev_stats=metrics(model,dev)
    checkpoint={'world':world,'seed':seed,'method':method,'variant':variant_for(method),
                'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'optimizer':opt.state_dict(), 'protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                'train_updates':max_updates,'probe_updates':nprobes,'events':events,
                'dev_stats':dev_stats,'loss_step0':loss0,
                'model_learned_coefficients':sum(p.numel() for p in model.parameters()),
                'final_hard_active_edges':model.graph.active_count() if hasattr(model,'graph') else None,
                'deployment_serialized_bytes':model.deploy_bytes(),
                'duration_train_s':time.perf_counter()-t0}
    out=CHECKPOINTS/f'{world}_{seed}_{method}.pt'
    torch.save(checkpoint,out)
    return {k:checkpoint[k] for k in ('world','seed','method','train_updates','probe_updates','dev_stats','model_learned_coefficients','final_hard_active_edges','deployment_serialized_bytes','duration_train_s')}


@torch.inference_mode()
def latency(model,world,seed,batch,nwarm=15,nruns=75):
    samples=corpus(world,'dev',seed,max(batch,128))[:batch,:-1]
    model.eval()
    for _ in range(nwarm):model(samples)
    ts=[]
    for _ in range(nruns):
        t0=time.perf_counter_ns();model(samples);ts.append((time.perf_counter_ns()-t0)/1000.)
    return float(np.median(ts))


def train_all(world=None):
    RESULTS.mkdir(exist_ok=True);CHECKPOINTS.mkdir(exist_ok=True)
    print('MA-1173 TRAIN ONLY, fresh sealed, protocol SHA256',hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),flush=True)
    worlds=[world] if world is not None else WORLDS
    for w in worlds:
        for seed in SEEDS:
            for method in METHODS:
                p=CHECKPOINTS/f'{w}_{seed}_{method}.pt'
                if p.exists():
                    print('skip existing',w,seed,method,flush=True);continue
                row=train_one(w,seed,method)
                print('train',w,seed,method, 'dev',round(row['dev_stats']['plain_nll'],5),
                      'query_acc',round(row['dev_stats']['query_accuracy'],4),
                      'remaining',row['final_hard_active_edges'],
                      'probe',row['probe_updates'],'sec',round(row['duration_train_s'],1),flush=True)
    paths=[CHECKPOINTS/f'{w}_{s}_{m}.pt' for w in WORLDS for s in SEEDS for m in METHODS]
    if all(x.exists() for x in paths):
        outputs={'all_train_models':len(paths),'source':'synthetic_autoregressive_grammar',
                 'protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                 'fresh_accessed':False,'checkpoint_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
        (RESULTS/'TRAIN_COMPLETE.json').write_text(json.dumps(outputs,indent=2)+'\n')
        print('TRAIN COMPLETE (fresh never opened)',len(paths),flush=True)


def audit():
    guard=RESULTS/'TRAIN_COMPLETE.json'
    assert guard.exists(), 'Fresh seal not opened: training incomplete'
    report=json.loads(guard.read_text());assert report['all_train_models']==len(SEEDS)*len(WORLDS)*len(METHODS)
    all_rows=[];event_rows=[];vjp=[]
    for w in WORLDS:
        for seed in SEEDS:
            # Fresh generation ONLY after all training has been completed.
            fresh=corpus(w,'fresh',seed,CFG['synthetic_causal_language_data']['fresh'])
            ood=corpus(w,'fresh_ood',seed,CFG['synthetic_causal_language_data']['fresh_ood'])
            for method in METHODS:
                path=CHECKPOINTS/f'{w}_{seed}_{method}.pt'
                checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                assert hashlib.sha256(path.read_bytes()).hexdigest()==report['checkpoint_sha256'][path.name]
                m=initialize_model(seed,checkpoint['variant']);m.load_state_dict(checkpoint['model'])
                replay=metrics(m,corpus(w,'dev',seed,CFG['synthetic_causal_language_data']['development']))
                assert abs(replay['plain_nll']-checkpoint['dev_stats']['plain_nll'])<1e-7
                f=metrics(m,fresh);o=metrics(m,ood)
                active=m.graph.active_count() if hasattr(m,'graph') else 0
                fixed_buffers=0
                if hasattr(m,'graph'):
                    g=m.graph
                    fixed_buffers=sum(t.numel()*t.element_size() for t in (g.perms,g.signs,g.house,g.sources,g.targets,g.views))
                edge_runtime=active*(2*D*D)*4
                b1=latency(m,w,seed,1);b32=latency(m,w,seed,32)
                row={'world':w,'seed':seed,'method':method,'train_updates':checkpoint['train_updates'],
                     'probe_updates':checkpoint['probe_updates'],
                     'all_optimizer_steps':checkpoint['train_updates']+checkpoint['probe_updates'],
                     'physical_trainable_coefficients':checkpoint['model_learned_coefficients'],
                     'active_virtual_edge_mlps':active,'logic_edge_mlp_weights':active*2*D*D,
                     'active_edge_mlp_mac_per_token':active*2*D*D*STEPS,
                     'serialized_inference_bytes':m.deploy_bytes(), 'fixed_decoder_runtime_buffer_bytes':fixed_buffers,
                     'decoded_active_weights_runtime_bytes':edge_runtime,
                     'fresh_plain_nll':f['plain_nll'],'fresh_query_nll':f['query_nll'],
                     'fresh_query_accuracy':f['query_accuracy'],
                     'fresh_ood_plain_nll':o['plain_nll'],'fresh_ood_query_nll':o['query_nll'],
                     'fresh_ood_query_accuracy':o['query_accuracy'],
                     'cpu_forward_B1_us_median':b1,'cpu_forward_B32_us_median':b32,
                     'dev_plain_nll':replay['plain_nll'],
                     'train_duration_seconds':checkpoint['duration_train_s']}
                all_rows.append(row)
                for e in checkpoint['events']:
                    event_rows.append({'world':w,'seed':seed,'method':method,**e})
                print('audit',w,seed,method,'fresh',round(f['plain_nll'],4),
                      'ruleAcc',round(f['query_accuracy'],3),'OODacc',round(o['query_accuracy'],3),
                      'active',active,flush=True)
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    with (RESULTS/'INTERVENTIONS.json').open('w') as f:json.dump(event_rows,f,indent=2)
    n=int(sum(r['active_virtual_edge_mlps'] for r in all_rows))
    print('all models',len(all_rows),'total active graph edge instances',n)
    from statistics import median
    summ={}
    for method in METHODS:
        rows=[r for r in all_rows if r['method']==method]
        summ[method]={k:round(median(r[k] for r in rows),6) for k in ['fresh_plain_nll','fresh_query_accuracy','fresh_ood_query_accuracy','active_virtual_edge_mlps','serialized_inference_bytes','cpu_forward_B1_us_median','all_optimizer_steps']}
    (RESULTS/'SUMMARY.json').write_text(json.dumps(summ,indent=2)+'\n')
    print('SUMMARY',json.dumps(summ,indent=2))
    return all_rows

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['train','audit']);parser.add_argument('--world',choices=WORLDS)
    args=parser.parse_args()
    if args.phase=='train':train_all(args.world)
    else:audit()
