"""Frozen MA-1165 CPU feasibility experiment and causal step diagnostics."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import torch
import torch.nn.functional as F

from model import VirtualLoop, make_teacher, N, M

MODES=('tied','same','switch','linear','gain','untied')
TASKS=('aligned','independent')
LOOPS=(2,4)
SEEDS=(31,47,59)
STEPS=240
BATCH=128
LR=.006


def inputs(n:int, seed:int, ood:bool=False)->torch.Tensor:
    gen=torch.Generator().manual_seed(seed)
    if ood:
        return torch.randn(n,8,generator=gen)*1.25+0.5
    return torch.randn(n,8,generator=gen)


def task_nmse(model,teacher,x):
    with torch.no_grad():
        y=teacher(x)
        pred=model(x)
        scale=(y-x).square().mean().clamp_min(1e-14)
        return ((pred-y).square().mean()/scale).item(),scale.item()


def step_gradients(model,teacher,x):
    if model.mode=='untied':return {'count':0, 'negative_fraction':None,'mean_cosine':None}
    y=teacher(x).detach()
    weights=model.virtual()
    gradients=[]
    for index in range(model.loops):
        chosen=[w if j==index else w.detach() for j,w in enumerate(weights)]
        residual=model(x,weight_override=chosen)-y
        loss=residual.square().mean()
        g=torch.autograd.grad(loss,model.u,retain_graph=True,allow_unused=True)[0]
        gradients.append(g.detach())
    cos=[]
    for a,b in itertools.combinations(gradients,2):
        value=float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
        cos.append(value)
    return {'count':len(cos),'negative_fraction':sum(z<0 for z in cos)/len(cos),
        'mean_cosine':sum(cos)/len(cos),'raw_cosines':cos}


def interventions(model,teacher,x):
    y=teacher(x).detach()
    ref=model(x).detach()
    scale=(y-x).square().mean().clamp_min(1e-14)
    before=((ref-y).square().mean()/scale).item()
    weights=model.virtual()
    swapped=model(x,weight_override=list(reversed(weights)))
    swapped_nmse=((swapped-y).square().mean()/scale).item()
    responses=[]
    for idx in range(model.loops):
        ablated=[w if j!=idx else torch.zeros_like(w) for j,w in enumerate(weights)]
        a=model(x,weight_override=ablated)
        responses.append((((a-y).square().mean()/scale).item()-before))
    return {'swap_nmse_delta':swapped_nmse-before,'ablated_nmse_deltas':responses,
            'functional_weight_differences':[(weights[j]-weights[j-1]).norm().item() for j in range(1,model.loops)]}


def single_run(root:Path,mode:str,task:str,loops:int,seed:int)->dict:
    torch.manual_seed(seed+1200)
    model=VirtualLoop(mode,loops,seed=seed+1200)
    teacher=make_teacher(task,loops,seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.0)
    rng=torch.Generator().manual_seed(100000+seed)
    train_curve=[]
    start=time.perf_counter()
    for step in range(STEPS):
        x=torch.randn(BATCH,8,generator=rng)
        with torch.no_grad():y=teacher(x)
        prediction=model(x)
        loss=(prediction-y).square().mean()
        if not torch.isfinite(loss):raise RuntimeError(f'non-finite loss {mode,task,loops,seed,step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (step+1)%60==0:train_curve.append({'update':step+1,'loss_mse':loss.item()})
    duration=time.perf_counter()-start
    dev=inputs(2048,200000+seed)
    fresh=inputs(4096,300000+seed)
    ood=inputs(4096,400000+seed,ood=True)
    train=inputs(2048,100000+seed)
    metrics={}
    for name,x in [('train',train),('dev',dev),('fresh',fresh),('ood',ood)]:
        metrics[name+'_nmse'], metrics[name+'_target_residual_variance']=task_nmse(model,teacher,x)
    diag_x=fresh[:128]
    # No adjustments or model selection on audit data; measurements only.
    gradients=step_gradients(model,teacher,diag_x)
    intervention=interventions(model,teacher,diag_x)
    label=f'{task}_{mode}_L{loops}_s{seed}'
    payload={'model':model.state_dict(),'mode':mode,'loops':loops,'seed':seed,
       'teacher_kind':task,'physical_trainable_scalars':model.learned_count()}
    weights_file=root/'checkpoints'/f'{label}.pt'
    torch.save(payload,weights_file)
    ck_hash=hashlib.sha256(weights_file.read_bytes()).hexdigest()
    row={'label':label,'task':task,'mode':mode,'loops':loops,'seed':seed,
        'training_updates':STEPS,'batch':BATCH,'learning_rate':LR,'dtype':'FP32',
        'train_seconds':round(duration,4),'parameters':model.learned_count(),
        'inference_bytes':model.payload_bytes(),'checkpoint_bytes':weights_file.stat().st_size,
        'checkpoint_sha256':ck_hash,
        'step_grad_negative_fraction':gradients['negative_fraction'],
        'step_grad_cosine_mean':gradients['mean_cosine'],
        'swap_nmse_delta':intervention['swap_nmse_delta'],
        'mean_ablation_nmse_delta':sum(intervention['ablated_nmse_deltas'])/loops,
        'mean_virtual_weight_difference':(sum(intervention['functional_weight_differences'])/len(intervention['functional_weight_differences'])) if intervention['functional_weight_differences'] else 0.0,
        **metrics}
    audit_file=root/'results'/f'{label}.json'
    audit_file.write_text(json.dumps({'row':row,'train_curve':train_curve,'step_gradient':gradients,
          'step_intervention':intervention,'environment':{'torch':torch.__version__,'cpu_threads':torch.get_num_threads(),
          'platform':platform.platform(),'python':sys.version,'cpu_clock_fixed':False}},indent=2,allow_nan=False))
    return row


def run(root:Path,quick:bool=False):
    root.mkdir(parents=True,exist_ok=True)
    (root/'results').mkdir(parents=True,exist_ok=True)
    (root/'checkpoints').mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    srcdir=Path(__file__).parent
    paths=[srcdir/'model.py',srcdir/'run_experiment.py']
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (root/'SOURCE_SHA256.json').write_text(json.dumps(manifest,indent=2))
    # Explicit ordering; aligned tasks cannot choose unaligned hyperparameters.
    full=[(task,loops,mode,seed) for task in TASKS for loops in LOOPS for mode in MODES for seed in SEEDS]
    todo=full[:1] if quick else full
    completed={}
    csvpath=root/'results'/'RESULTS_CORE.csv'
    if csvpath.exists():
        with csvpath.open(newline='') as f:completed={r['label']:r for r in csv.DictReader(f)}
    for i,(task,loops,mode,seed) in enumerate(todo):
        label=f'{task}_{mode}_L{loops}_s{seed}'
        if label in completed:continue
        row=single_run(root,mode,task,loops,seed)
        completed[label]=row
        with csvpath.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=sorted(next(iter(completed.values())).keys()))
            writer.writeheader()
            for rec in completed.values():writer.writerow(rec)
        if (i+1)%6==0 or quick:print(f'completed {len(completed)}/{len(todo)}: {label}, fresh={row["fresh_nmse"]:.5f}',flush=True)
    return len(completed)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path(__file__).parents[1]);parser.add_argument('--quick',action='store_true')
    args=parser.parse_args();run(args.out,args.quick)
