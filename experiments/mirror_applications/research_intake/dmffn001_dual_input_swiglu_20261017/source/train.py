"""Frozen paired CPU experiment for gate/up dual-axis SwiGLU Mirror views."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import time
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from model import DualSwiGLU, generate_world,serialize_inference

ROOT=Path(__file__).resolve().parents[1]
METHODS=['tied','mirror1','mirror4','mirror12','gain12','gate_mirror12','up_mirror12','pair_mirror4','pair_gain4','factor_lora1','independent','auto_mirror']
FAMILIES=['aligned_full','aligned_holdout','partial_full','unrelated_full']
HELDO=[2,3,7]
ALL_INDICES=list(range(9))
TRAINED=[i for i in ALL_INDICES if i not in HELDO]
STEPS=[300,600,1200,2400]


def create(name, init_seed):
    if name in ('auto_mirror','mirror12'):method,q='mirror',12
    elif name.startswith('mirror') and name[6:].isdigit():method,q='mirror',int(name[6:])
    elif name=='gain12':method,q='gain',12
    elif name=='gate_mirror12':method,q='mirror_gate',12
    elif name=='up_mirror12':method,q='mirror_up',12
    elif name=='pair_mirror4':method,q='pair_mirror',4
    elif name=='pair_gain4':method,q='pair_gain',4
    else:method,q=name,12
    # Seeded base W (all methods share exactly the same tensor coordinates).
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        shared=DualSwiGLU('tied',d=8,h=24,K=3,q=12)
        model=DualSwiGLU(method,d=8,h=24,K=3,q=q)
        with torch.no_grad():
            if method=='independent':
                for key in ['wg','wu','wd']:
                    getattr(model,key).copy_(getattr(shared,key).unsqueeze(0).expand_as(getattr(model,key)))
            else:
                for key in ['wg','wu','wd']:
                    getattr(model,key).copy_(getattr(shared,key))
    if name=='auto_mirror':model.set_active(1)
    return model


def train_single(arg):
    world_id,init_id,family,label,steps,lr,output=arg
    torch.set_num_threads(1)
    wd=family.replace('_full','').replace('_holdout','')
    world=generate_world(world_id,wd,d=8,h=24,K=3)
    train_mask=TRAINED if family.endswith('holdout') else ALL_INDICES
    model=create(label,911000+init_id)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.0001)
    g=torch.Generator().manual_seed(551100 + world_id*7 + init_id)
    choices=torch.randint(len(world['train'][0]),(steps,64),generator=g)
    name=f'{family}_w{world_id}_s{init_id}_{label}_lr{lr:g}'
    outdir=Path(output)
    (outdir/'models').mkdir(parents=True,exist_ok=True)
    (outdir/'runs').mkdir(parents=True,exist_ok=True)
    log=[];grows=[]
    scale2=float(world['scale']**2)
    train_x,train_y=world['train'];val_x,val_y=world['val'];test_x,test_y=world['test']
    clock=time.perf_counter()
    prev_val=None
    for step in range(1,steps+1):
        idx=choices[step-1]
        out=model(train_x[idx]);y=train_y[idx]
        loss=((out[:,train_mask,:]-y[:,train_mask,:])**2).mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step==steps or step in STEPS or (label=='auto_mirror' and step in (400,800,1200,1600,2000)):
            model.eval()
            with torch.no_grad():
                train_pred=model(train_x)
                val_pred=model(val_x)
                test_pred=model(test_x)
                v=float(((val_pred[:,train_mask]-val_y[:,train_mask])**2).mean()/scale2)
                tr=float(((train_pred[:,train_mask]-train_y[:,train_mask])**2).mean()/scale2)
                tst=float(((test_pred-test_y)**2).mean()/scale2)
                tst_train=float(((test_pred[:,train_mask]-test_y[:,train_mask])**2).mean()/scale2)
                tst_hold=float(((test_pred[:,HELDO]-test_y[:,HELDO])**2).mean()/scale2)
                mode_worst=float(((test_pred-test_y)**2).mean(dim=(0,2)).max()/scale2)
                diversity=float(test_pred.var(dim=1).mean().sqrt())
                target_div=float(test_y.var(dim=1).mean().sqrt())
                div_ratio=diversity/max(target_div,1e-12)
                row={'name':name,'world':world_id,'seed':init_id,'family':family,'method':label,'step':step,
                     'train_nmse':tr,'val_nmse':v,'test_nmse':tst,'test_trained_pairs_nmse':tst_train,
                     'test_heldout_pairs_nmse':tst_hold,'worst_pair_nmse':mode_worst,
                     'output_diversity_ratio':div_ratio,'active_q':model.active_q,
                     'active_parameters':model.active_trainable_params()}
                log.append(row)
                if label=='auto_mirror' and step in (400,800,1200,1600,2000):
                    if prev_val is not None and (v >= prev_val*0.90) and v>0.002 and model.active_q<12:
                        old=model.active_q
                        model.set_active(4 if old==1 else 12)
                        grows.append({'step':step,'before_q':old,'after_q':model.active_q,'validation_nmse':v,
                                      'previous_validation_nmse':prev_val})
                    prev_val=v
            model.train()
    elapsed=time.perf_counter()-clock
    data=serialize_inference(model)
    blob_name=f'{name}.pt'
    (outdir/'models'/blob_name).write_bytes(data)
    final=log[-1].copy();final.update({'serialized_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                                      'elapsed_s':elapsed,'updates':steps,'examples_presented':steps*64,
                                      'training_unique_inputs':len(train_x),'validation_inputs':len(val_x),
                                      'test_inputs':len(test_x),'training_pairs':train_mask,'heldout_pairs':HELDO,
                                      'expansions':grows,'checkpoints':log,'lr':lr,
                                      'model_path':'models/'+blob_name})
    (outdir/'runs'/f'{name}.json').write_text(json.dumps(final,indent=2)+'\n')
    return {'name':name,'test':final['test_nmse'],'hold':final['test_heldout_pairs_nmse'],'q':final['active_q'],'bytes':len(data),'elapsed_s':round(elapsed,2)}


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--phase',choices=['dev','fresh'],required=True)
    pa.add_argument('--workers',type=int,default=3)
    pa.add_argument('--steps',type=int)
    pa.add_argument('--lr',type=float)
    pa.add_argument('--out')
    args=pa.parse_args()
    torch.set_num_threads(1)
    if args.phase=='dev':
        worlds=[73001]; families=['aligned_full','unrelated_full'];seeds=[0]; methods=['mirror4','mirror12','gain12','factor_lora1','independent','tied'];steps=args.steps or 800;rates=[0.008,0.018]
    else:
        worlds=[73111,73112,73113,73114,73115];families=FAMILIES;seeds=[0,1];methods=METHODS;steps=args.steps or 2400;rates=[args.lr]
        assert args.lr is not None
    jobs=[]
    for w in worlds:
        for family in families:
            for seed in seeds:
                for lr in rates:
                    for method in methods:
                        jobs.append((w,seed,family,method,steps,lr,args.out or str(ROOT/'results'/args.phase)))
    from concurrent.futures import ProcessPoolExecutor,as_completed
    start=time.perf_counter()
    errors=[];done=0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures={ex.submit(train_single,j):j for j in jobs}
        for future in as_completed(futures):
            try:
                value=future.result()
                done+=1
                if done%10==0 or done==len(jobs):
                    print(json.dumps({'progress':done,'total':len(jobs),'sec':round(time.perf_counter()-start,1),'latest':value},ensure_ascii=False),flush=True)
            except Exception as e:
                errors.append({'job':futures[future],'error':repr(e)})
                print('ERROR',repr(e),'job',futures[future],flush=True)
    print(json.dumps({'phase':args.phase,'done':done,'total':len(jobs),'error_count':len(errors),'elapsed_s':time.perf_counter()-start},ensure_ascii=False),flush=True)
    if errors:(Path(args.out or str(ROOT/'results'/args.phase))/'ERRORS.json').write_text(json.dumps(errors,indent=2));sys.exit(1)

if __name__=='__main__':main()
