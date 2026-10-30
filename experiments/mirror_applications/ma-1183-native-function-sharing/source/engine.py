"""Frozen experimental core. Controlled generator separates Train/DEV from Fresh/OOD."""
from __future__ import annotations
import json,hashlib,io,time,copy,os,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from core import ConditionalMLP,Teacher,dataset,DIN,H,DOUT,K

ARMS=['shared','film','mirror','coupled','ind_budget','ind_upper','fixed_mirror']
ACTS=['relu','silu']
GAMMAS=[0.,0.35,1.]

def hash_tensor(model):
    b=io.BytesIO();torch.save(model.state_dict(),b);return hashlib.sha256(b.getvalue()).hexdigest()

def configure():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

@torch.no_grad()
def eval_loss(model,tensors,batch=1024):
    model.eval();x,t,y=tensors
    ys=torch.cat([model(x[i:i+batch],t[i:i+batch]) for i in range(0,len(x),batch)])
    mse=(ys-y).square().mean().item()
    denom=(y-y.mean(0)).square().mean().item()
    return mse/max(denom,1e-12)

@torch.no_grad()
def activation_effect(model,x):
    model.eval();n=min(len(x),256);xx=x[:n]
    preds=[]
    for j in range(K):preds.append(model(xx,torch.full((n,),j,dtype=torch.long)))
    q=torch.stack(preds)
    norm=float(q.square().mean().sqrt().item())
    return float((q[0]-q[1]).square().mean().sqrt().item()/max(norm,1e-8))

@torch.no_grad()
def swap_loss(model,tensors):
    model.eval();x,t,y=tensors
    perm=(t+1)%K
    return eval_loss(model,(x,perm,y))

def train_one(arm,act,gamma,seed,proto,outdir,save=True):
    # Different teacher generation seeds; fixed generated train/dev per task.
    tstart=time.perf_counter()
    teacher=Teacher(seed+int(10000*gamma)+({'relu':0,'silu':55000}[act]),act,gamma)
    train=dataset(teacher,seed+int(10000*gamma), 'train',proto['n_train'])
    dev=dataset(teacher,seed+int(10000*gamma),'dev',proto['n_dev'])
    model=ConditionalMLP(arm,act,initseed=40000+seed)
    params=model.physical_params()
    with torch.no_grad():
        init_div=activation_effect(model,dev[0])
        init_dev=eval_loss(model,dev)
    optimizer=torch.optim.AdamW(model.parameters(),lr=proto['lr'],weight_decay=proto['weight_decay'])
    rg=torch.Generator(device='cpu').manual_seed(900000+seed*19+int(gamma*10000)+(0 if act=='relu' else 6666))
    best=float('inf');bstep=-1;best_state=None
    hist=[]
    idxlim=len(train[0]);batch=proto['batch'];steps=proto['steps'];every=proto['eval_every']
    for step in range(1,steps+1):
        ix=torch.randint(idxlim,(batch,),generator=rg)
        x,t,y=(v[ix] for v in train)
        model.train();pred=model(x,t)
        loss=F.mse_loss(pred,y)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step%every==0:
            e=eval_loss(model,dev)
            hist.append([step,e])
            if e<best:bstep=step;best=e;best_state=copy.deepcopy(model.state_dict())
    assert best_state is not None
    model.load_state_dict(best_state)
    final_dev=eval_loss(model,dev)
    final_div=activation_effect(model,dev[0])
    dev_swap=swap_loss(model,dev)
    elapsed=time.perf_counter()-tstart
    key=f'{act}_g{gamma:.2f}_seed{seed}_{arm}'
    checkpoint=str(Path(outdir)/'checkpoints'/f'{key}.pt')
    if save:
        Path(checkpoint).parent.mkdir(parents=True,exist_ok=True)
        torch.save({'state':best_state,'key':key,'params':params,'step':bstep,'dev':best},checkpoint)
        # Count an actual serialized inference packet, including architecture metadata.
        packet=io.BytesIO()
        torch.save({"arm":arm,"act":act,"K":K,"DIN":DIN,"H":H,"DOUT":DOUT,"state":best_state},packet)
        packetbytes=len(packet.getvalue())
        cksha=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    else:packetbytes=-1;cksha=None
    # Eager CPU FP32 batch128 latency, serial execution; no fixed clock or GPU.
    with torch.no_grad():
        model.eval(); bx=dev[0][:128];bt=dev[1][:128]
        for _ in range(25):model(bx,bt)
        batches=[]
        for _ in range(5):
            t0=time.perf_counter()
            for j in range(80): model(bx,bt)
            batches.append((time.perf_counter()-t0)*1e6/80)
        latency=float(np.median(batches))
    return dict(key=key,arm=arm,act=act,gamma=gamma,seed=seed,params=params,packetbytes=packetbytes,checkpoint_sha=cksha,checkpoint=checkpoint,selected_step=bstep,dev=final_dev,init_dev=init_dev,init_div=init_div,final_div=final_div,dev_swap=dev_swap,elapsed_s=elapsed,latency_batch128_us=latency,history=hist)

@torch.no_grad()
def verify_and_evaluate(row,proto):
    info=torch.load(row['checkpoint'],weights_only=True,map_location='cpu')
    if hashlib.sha256(Path(row['checkpoint']).read_bytes()).hexdigest()!=row['checkpoint_sha']:raise RuntimeError('checkpoint hash mismatch')
    model=ConditionalMLP(row['arm'],row['act'],initseed=40000+row['seed'])
    model.load_state_dict(info['state'])
    gamma=row['gamma'];seed=row['seed'];act=row['act']
    teacher=Teacher(seed+int(10000*gamma)+({'relu':0,'silu':55000}[act]),act,gamma)
    dev=dataset(teacher,seed+int(10000*gamma),'dev',proto['n_dev'])
    devscore=eval_loss(model,dev)
    if abs(devscore-row['dev'])>1e-12:raise RuntimeError('dev replay mismatch')
    fresh=dataset(teacher,seed+int(10000*gamma),'fresh',proto['n_fresh'])
    ood=dataset(teacher,seed+int(10000*gamma),'ood',proto['n_ood'])
    freshscore=eval_loss(model,fresh);oodscore=eval_loss(model,ood)
    swap=swap_loss(model,fresh)
    return {**row,'fresh':freshscore,'ood':oodscore,'fresh_swap':swap,'fresh_swap_delta':swap-freshscore}
