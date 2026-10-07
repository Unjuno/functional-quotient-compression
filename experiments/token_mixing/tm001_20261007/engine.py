from __future__ import annotations
import json, math, os, time
import torch
from torch.nn import functional as F
from tm001 import *


def config_for(world,P,d=32,layers=2,ff=64):
    v=vocab_info(world)
    return dict(states=world['states'],rules=world['rules'],vocab=v['vocab'],d=d,heads=4,layers=layers,ff=ff,max_len=4+P,P=P)


def batch_loss(model,world,mode,P,batch,seed):
    b=sample_batch(world,batch,P,mode,seed)
    if isinstance(model,TinyAR):
        seq=torch.cat((b['context'],b['targets']),1)
        logits=full_ar_logits(model,seq,b['context'].shape[1],P)
    else:
        logits=model(b['context'])
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),b['targets'].reshape(-1))


def train(method,world,mode,P,seed,steps=800,lr=3e-3,batch=128,d=32,layers=2,ff=64,checkpoints=(200,400,800)):
    c=config_for(world,P,d,layers,ff);torch.manual_seed(seed)
    if method=='ar':m=TinyAR(c)
    elif method=='direct':m=PeriodModel(c,mixer=False)
    elif method=='mixer':m=PeriodModel(c,mixer=True)
    else:raise ValueError(method)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps,eta_min=lr*.05)
    curve=[];t0=time.perf_counter()
    for step in range(1,steps+1):
        opt.zero_grad(set_to_none=True)
        loss=batch_loss(m,world,mode,P,batch,seed*100000+step)
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();sched.step()
        if step in checkpoints or step==steps:
            ev=evaluate(m,world,mode,P,batch_eval=1024)
            curve.append(dict(step=step,loss=float(loss.detach()),**{k:v for k,v in ev.items() if isinstance(v,(float,int))}))
    return m,curve,time.perf_counter()-t0


def exhaustive(world,mode,P):
    s,r=world['states'],world['rules'];v=vocab_info(world)
    rules=[];xs=[];zs=[]
    branch_values=(0,) if mode=='deterministic' else (0,1)
    for rr in range(r):
        for x in range(s):
            for z in branch_values:
                rules.append(rr);xs.append(x);zs.append(z)
    rules=torch.tensor(rules);x=torch.tensor(xs);z=torch.tensor(zs)
    bz=torch.where(z==0,torch.full_like(z,v['branch0']),torch.full_like(z,v['branch1']))
    if mode=='hidden':bz=torch.full_like(z,v['hidden'])
    context=torch.stack((torch.full_like(x,v['bos']),rules+v['rule0'],bz,x),1)
    cur=x.clone();ys=[]
    for _ in range(P):
        cur=world['table'][z,rules,cur];ys.append(cur.clone())
    return context,torch.stack(ys,1),rules,x,z


def greedy_ar(m,context,P):
    log,caches=m.prefill(context);outs=[]
    tok=log[:,0].argmax(-1);outs.append(tok)
    C=context.shape[1]
    for i in range(1,P):
        log,caches=m.step(tok,caches,C+i-1);tok=log[:,0].argmax(-1);outs.append(tok)
    return torch.stack(outs,1)


def evaluate(m,world,mode,P,batch_eval=2048):
    m.eval();context,target,rules,x,z=exhaustive(world,mode,P);C=context.shape[1]
    all_logits=[];all_gen=[]
    with torch.no_grad():
        for st in range(0,len(context),batch_eval):
            cx=context[st:st+batch_eval];ty=target[st:st+batch_eval]
            if isinstance(m,TinyAR):
                seq=torch.cat((cx,ty),1);logits=full_ar_logits(m,seq,C,P);gen=greedy_ar(m,cx,P)
            else:
                logits=m(cx);gen=logits.argmax(-1)
            all_logits.append(logits.cpu());all_gen.append(gen.cpu())
    logits=torch.cat(all_logits);gen=torch.cat(all_gen)
    ce=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),reduction='none').reshape(-1,P)
    slot_acc=(gen==target).float().mean(0)
    out=dict(nll_token=float(ce.mean()),nll_sequence=float(ce.sum(1).mean()),token_acc=float((gen==target).float().mean()),joint_acc=float((gen==target).all(1).float().mean()))
    for i,a in enumerate(slot_acc.tolist(),1):out[f'slot{i}_acc']=float(a)
    if mode=='hidden':
        unique=context[::2]
        with torch.no_grad():
            if isinstance(m,TinyAR):g=greedy_ar(m,unique,P).cpu()
            else:g=m(unique).argmax(-1).cpu()
        t0=target[::2];t1=target[1::2]
        valid=((g==t0).all(1)|(g==t1).all(1)).float().mean()
        out['valid_branch_rate']=float(valid)
        out['oracle_joint_entropy_nats']=float(((t0!=t1).any(1).float().mean()*math.log(2)).item())
    m.train();return out


def benchmark_pair(ar,macro,context,P,batch_sizes=(1,32,128),warmup=20,iters=100):
    rows=[];ar.eval();macro.eval();torch.set_num_threads(1)
    with torch.no_grad():
        for B in batch_sizes:
            cx=context[:B].clone()
            if len(cx)<B:cx=cx.repeat((B+len(cx)-1)//len(cx),1)[:B]
            def f_ar():greedy_ar(ar,cx,P)
            def f_ma():macro(cx)
            for fn in (f_ar,f_ma):
                for _ in range(warmup):fn()
            for name,fn in [('ar_cached',f_ar),('period_one_forward',f_ma)]:
                times=[]
                for _ in range(iters):
                    t=time.perf_counter();fn();times.append(time.perf_counter()-t)
                med=float(torch.tensor(times).median())
                rows.append(dict(method=name,batch=B,P=P,median_ms=med*1000,tokens_per_s=B*P/med))
    return rows


def train_latent(world,P,seed,steps=1600,lr=3e-3,batch=128,d=32,layers=2,ff=64,mixer=True,checkpoints=()):
    c=config_for(world,P,d,layers,ff);c['latents']=2;torch.manual_seed(seed);m=LatentPeriodModel(c,mixer=mixer)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps,eta_min=lr*.05)
    curve=[];t0=time.perf_counter()
    for step in range(1,steps+1):
        b=sample_batch(world,batch,P,'hidden',seed*100000+step)
        opt.zero_grad(set_to_none=True);logits=m(b['context']);loss=latent_joint_nll(logits,b['targets'],m.mixture_logits(batch));loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();sched.step()
        if step in checkpoints or step==steps:curve.append(dict(step=step,loss=float(loss.detach()),**evaluate_latent(m,world,P)))
    return m,curve,time.perf_counter()-t0


def evaluate_latent(m,world,P,batch_eval=512):
    m.eval();context,target,rules,x,z=exhaustive(world,'hidden',P);nlls=[];all_pred=[]
    with torch.no_grad():
        for st in range(0,len(context),batch_eval):
            cx=context[st:st+batch_eval];ty=target[st:st+batch_eval]
            logits=m(cx);nlls.append(latent_joint_nll(logits,ty,m.mixture_logits(len(cx)),reduction='none').cpu());all_pred.append(logits.argmax(-1).cpu())
    nll=torch.cat(nlls);pred=torch.cat(all_pred);p=pred[::2];t0=target[::2];t1=target[1::2]
    valid=((p==t0[:,None,:]).all(-1)|(p==t1[:,None,:]).all(-1))
    both=((p[:,0]==t0).all(-1)&(p[:,1]==t1).all(-1))|((p[:,1]==t0).all(-1)&(p[:,0]==t1).all(-1))
    distinct=(p[:,0]!=p[:,1]).any(-1);probs=F.softmax(m.mix_logits.detach().cpu(),-1);k=int(probs.argmax());g=p[:,k]
    map_valid=((g==t0).all(-1)|(g==t1).all(-1)).float().mean();gd=g.repeat_interleave(2,0);joint_acc=(gd==target).all(-1).float().mean();m.train()
    return dict(nll_sequence=float(nll.mean()),coverage_both=float(both.float().mean()),latent_valid_rate=float(valid.float().mean()),latent_distinct_rate=float(distinct.float().mean()),map_valid_branch_rate=float(map_valid),map_joint_acc=float(joint_acc),mix_p0=float(probs[0]),mix_p1=float(probs[1]))
