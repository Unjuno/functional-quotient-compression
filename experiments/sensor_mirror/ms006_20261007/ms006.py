from __future__ import annotations
import time
import math
import torch
from ms005 import MirrorView, make_mirror_bank, mirror_view_theta, model_forward

def make_symmetric_bank(sh, seed:int, rho:float, dtype=torch.float32):
    base=make_mirror_bank(sh.hidden,8,seed+99,rho=rho,dtype=dtype)
    out=[]
    for v in base:
        out += [v, MirrorView(v.a1,v.a2,-v.rho)]
    return out

def multiview_mean_loss(th,x,y,sh,bank):
    losses=[]
    for view in bank:
        pred=model_forward(mirror_view_theta(th,sh,view),x,sh)
        losses.append((pred-y).square().mean())
    return torch.stack(losses).mean()

def identity_loss(th,x,y,sh):
    return (model_forward(th,x,sh)-y).square().mean()

def adamw_adapt(parent,world,sh,bank,steps:int,batch:int,lr:float,seed:int,objective='mirror_mean',weight_decay=1e-4,clip=1.0):
    th=parent.detach().clone().requires_grad_(True)
    opt=torch.optim.AdamW([th],lr=lr,weight_decay=weight_decay)
    g=torch.Generator().manual_seed(seed)
    t0=time.perf_counter()
    for _ in range(int(steps)):
        idx=torch.randint(len(world.train_x),(batch,),generator=g)
        x=world.train_x[idx]; y=world.train_y1[idx]
        if objective=='mirror_mean': loss=multiview_mean_loss(th,x,y,sh,bank)
        elif objective=='identity': loss=identity_loss(th,x,y,sh)
        else: raise ValueError(objective)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([th],clip); opt.step()
    return th.detach(), {'optimizer_steps':int(steps),'wall_seconds':time.perf_counter()-t0,'objective':objective,'lr':float(lr)}

def es_adapt_time(parent,world,sh,bank,wall_budget:float,batch:int,sigma:float,step:float,pairs:int,seed:int,mode='independent_eps'):
    from ms005 import es_estimate
    th=parent.detach().clone(); data_g=torch.Generator().manual_seed(seed)
    t0=time.perf_counter(); gens=0; branches=0
    while True:
        idx=torch.randint(len(world.train_x),(batch,),generator=data_g)
        x=world.train_x[idx]; y=world.train_y1[idx]
        ge,meta=es_estimate(th,x,y,sh,bank,pairs,sigma,seed=seed*100000+gens*101+len(bank)*7,mode=mode)
        branches += meta['branch_evals']; gn=ge.norm()
        if gn>0 and torch.isfinite(gn): th=th-step*ge/(gn+1e-12)
        gens += 1
        if time.perf_counter()-t0 >= wall_budget: break
    return th.detach(), {'generations':gens,'wall_seconds':time.perf_counter()-t0,'branch_evals':branches,'pairs':int(pairs),'sigma':float(sigma),'step':float(step),'mode':mode,'wall_budget':float(wall_budget)}

def adamw_adapt_time_cosine(parent,world,sh,bank,wall_budget:float,batch:int,lr_start:float,lr_end:float,seed:int,objective='mirror_mean',weight_decay=1e-4,clip=1.0):
    th=parent.detach().clone().requires_grad_(True)
    opt=torch.optim.AdamW([th],lr=lr_start,weight_decay=weight_decay)
    g=torch.Generator().manual_seed(seed)
    t0=time.perf_counter(); steps=0
    while True:
        elapsed=time.perf_counter()-t0
        frac=min(1.0, elapsed/max(wall_budget,1e-12))
        lr=lr_end + 0.5*(lr_start-lr_end)*(1.0+math.cos(math.pi*frac))
        for pg in opt.param_groups: pg['lr']=lr
        idx=torch.randint(len(world.train_x),(batch,),generator=g)
        x=world.train_x[idx]; y=world.train_y1[idx]
        if objective=='mirror_mean': loss=multiview_mean_loss(th,x,y,sh,bank)
        elif objective=='identity': loss=identity_loss(th,x,y,sh)
        else: raise ValueError(objective)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([th],clip); opt.step(); steps+=1
        if time.perf_counter()-t0 >= wall_budget: break
    return th.detach(), {'optimizer_steps':steps,'wall_seconds':time.perf_counter()-t0,'objective':objective,'lr_start':float(lr_start),'lr_end':float(lr_end),'wall_budget':float(wall_budget)}
