"""Pre-specified synthetic teacher / off-orbit study of one-scalar virtual MLP.

Training and evaluation datasets are disjoint and generated deterministically by
independent CPU RNG streams. The scalar is the ONLY learned number in the proposed
models. All 337 conventional MLP weights are transient outputs of the Mirror program.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from source.model import VirtualMLP, DenseMLP, WEIGHTS

KINDS = ('mirror','spectral','linear_view','scalar_gain','dense')
TASKS = ('aligned','off_orbit')
DEV = (618320,)
FRESH = (618321,618322,618323)
SEEDS = (0,1)


def get_sstar(world, seed):
    # One scalar truth per independent world/seed, no learned teacher extras.
    return 0.15+0.45*math.sin(world*0.93+seed*0.51)


def dataset(task, world, seed, which, n=512):
    g=torch.Generator().manual_seed(1000000 + world*23 + seed*987 + (0 if which=='train' else 414145))
    x=(torch.rand(n,2,generator=g,dtype=torch.float64)*4-2)
    if task=='aligned':
        true=VirtualMLP('mirror',init_s=get_sstar(world,seed))
        with torch.no_grad(): y=true(x)
    elif task=='off_orbit':
        phase=0.65*math.sin(world*.23+seed*.17)
        y=(0.55*torch.sin(2.4*x[:,0]+phase)+
           0.30*torch.cos(1.8*x[:,1]-0.7*phase)+
           0.40*x[:,0]*x[:,1]).unsqueeze(-1)
    else: raise ValueError(task)
    return x,y


def make_dense_from_mirror_start():
    d=DenseMLP()
    v=VirtualMLP('mirror',init_s=.2)
    with torch.no_grad():
        for mod,(w,b) in zip(d.layers,v.virtual_weights()):
            mod.weight.copy_(w);mod.bias.copy_(b)
    return d


def run_one(task,kind,world,seed,lr,updates,batch=64):
    torch.manual_seed(11000+seed)
    torch.set_num_threads(1)
    trainx, trainy=dataset(task,world,seed,'train',n=512)
    valx,valy=dataset(task,world,seed,'val',n=512)
    if kind=='dense':
        model=make_dense_from_mirror_start()
    else:
        model=VirtualMLP(kind,init_s=.2)
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
    x_gen=torch.Generator().manual_seed(world*299 + seed*17 + 1357)
    with torch.no_grad():
        before=F.mse_loss(model(valx),valy).item()
    start=time.perf_counter()
    for step in range(updates):
        select=torch.randint(0,len(trainx),(batch,),generator=x_gen)
        opt.zero_grad(set_to_none=True)
        loss=F.mse_loss(model(trainx[select]),trainy[select])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        opt.step()
    train_seconds=time.perf_counter()-start
    model.eval()
    with torch.no_grad():
        pred=model(valx)
        mse=F.mse_loss(pred,valy).item()
        var=valy.var(unbiased=False).item()
        r2=1-mse/var if var>1e-12 else float('nan')
        probe=valx[:64]
        for _ in range(5):model(probe)
        times=[]
        for _ in range(20):
            a=time.perf_counter()
            model(probe)
            times.append((time.perf_counter()-a)*1000)
    scalar=model.s.item() if kind!='dense' else None
    param_total=sum(p.numel() for p in model.parameters())
    return dict(task=task,model=kind,world=world,seed=seed,lr=lr,updates=updates,
                batch=batch,train_samples=512,eval_samples=512,teacher_scalar=get_sstar(world,seed) if task=='aligned' else '',
                learned_scalar=scalar if scalar is not None else '',learned_scalar_count=param_total,
                virtual_mlp_weights=WEIGHTS,initial_val_mse=before,val_mse=mse,eval_variance=var,val_r2=r2,
                train_wall_seconds=train_seconds,median_forward_ms_batch64=sorted(times)[len(times)//2],
                inference_checkpoint_bytes=model.checkpoint_bytes(),torch=str(torch.__version__),dtype='float64',
                cpu_threads=torch.get_num_threads())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['dev','fresh'],required=True)
    p.add_argument('--lr',type=float,default=.03)
    p.add_argument('--updates',type=int,default=400)
    p.add_argument('--kinds',default=','.join(KINDS))
    p.add_argument('--tasks',default=','.join(TASKS))
    p.add_argument('--seeds',default='0,1')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    kinds=tuple(a.kinds.split(','));tasks=tuple(a.tasks.split(','));seeds=tuple(map(int,a.seeds.split(',')))
    assert set(kinds).issubset(KINDS) and set(tasks).issubset(TASKS) and set(seeds).issubset(SEEDS)
    worlds=DEV if a.phase=='dev' else FRESH
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(str(path))
    rows=[]
    for task in tasks:
        for world in worlds:
            for seed in seeds:
                for kind in kinds:
                    row=run_one(task,kind,world,seed,a.lr,a.updates)
                    rows.append(row)
                    print(f'{task:9} {world} {seed} {kind:12} MSE={row["val_mse"]:.6g} R2={row["val_r2"]:+.4f} train={row["train_wall_seconds"]:.2f}s',flush=True)
                    with path.open('w',newline='') as f:
                        wr=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
                        wr.writeheader();wr.writerows(rows)
    path.with_suffix('.meta.json').write_text(json.dumps(dict(args=vars(a),worlds=worlds,seeds=seeds,
                                    torch=str(torch.__version__),cpu=platform.processor(),
                                    platform=platform.platform(),rows=len(rows)),indent=2))


if __name__=='__main__':main()
