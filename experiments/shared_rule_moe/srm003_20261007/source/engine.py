"""SRM003 runner; all training input streams are model-independent."""
from __future__ import annotations
import copy, hashlib, json, math, os, platform, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from core import make_world, make_model, encode_model, decode_model, compact

BASE=dict(kind='dense',d=32,ff=48,final_ff=48,heads=4,layers=2,vocab=43,length=5,atoms=8,arank=2,experts=8,rank=4,expert_ff=16,topk=2)
KINDS=['dense','moe','lora','shared','hybrid']

def select_configs():
    target=dict(BASE,kind='hybrid')
    cap=len(encode_model(make_model(target,0)))
    configs={'hybrid':target}
    search={'dense':('final_ff',range(48,160)), 'moe':('expert_ff',range(1,64)),
            'lora':('rank',range(1,16)), 'shared':('arank',range(1,16))}
    for kind,(key,values) in search.items():
        valid=[]
        for v in values:
            c=dict(BASE,kind=kind,**{key:v})
            if kind=='moe':c['experts']=4
            size=len(encode_model(make_model(c,0)))
            if size<=cap:valid.append((size,c))
        configs[kind]=max(valid,key=lambda a:a[0])[1]
    return configs,cap


def stream(world:dict,steps:int,seed:int,pair_only:bool=False,batch:int=96):
    g=torch.Generator().manual_seed(seed)
    n_atomic=0 if pair_only else batch//2
    n_pairs=batch-n_atomic
    # Every atomic entry is visited in each complete cycle.
    na=len(world['atomic_targets']);nt=len(world['train_targets'])
    ai=(torch.cat([torch.randperm(na,generator=g) for _ in range((steps*n_atomic+na-1)//na)])[:steps*n_atomic]
        if n_atomic else torch.empty(0,dtype=torch.long))
    pi=torch.randint(nt,(steps*n_pairs,),generator=g)
    a=world['atomic_tokens'][ai].reshape(steps,n_atomic,5)
    p=world['train_tokens'][pi].reshape(steps,n_pairs,5)
    x=torch.cat([a,p],1)
    y=torch.cat([world['atomic_targets'][ai].reshape(steps,n_atomic),world['train_targets'][pi].reshape(steps,n_pairs)],1)
    return x,y


@torch.no_grad()
def metrics(m,w,which='audit'):
    m.eval();out={}
    for name in (which,'atomic'):
        tok=w[name+'_tokens'];labels=w[name+'_targets'];pred=[];loss=[]
        for i in range(0,len(labels),256):
            z=m(tok[i:i+256]);pred.append(z.argmax(-1));loss.append(F.cross_entropy(z,labels[i:i+256],reduction='none'))
        pred=torch.cat(pred);correct=(pred==labels)
        out[name+'_correct']=int(correct.sum());out[name+'_n']=len(labels)
        out[name+'_acc']=correct.float().mean().item();out[name+'_ce']=torch.cat(loss).mean().item()
        if name=='atomic':
            per=correct.reshape(24,16).sum(1)
            out['exact_rules']=int((per==16).sum());out['rule_correct_counts']=per.tolist()
    if which=='audit':
        # Both orders are independently predicted; pair list is lexicographically ordered.
        pair=w['audit_pairs'];mapping={tuple(p):i for i,p in enumerate(pair.tolist())}
        reverse=torch.tensor([mapping[(b,a)] for a,b in pair.tolist()])
        p=[]
        for i in range(0,len(w['audit_tokens']),256):p.append(m(w['audit_tokens'][i:i+256]).argmax(-1))
        ok=(torch.cat(p)==w['audit_targets']).reshape(-1,16)
        out['reverse_both_acc']=(ok & ok[reverse]).float().mean().item()
    return out


def train(m,w,x,y,start,end,lr,path,checkpoints=(),evaluate='dev'):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    record=[];wall=cpu=0.;count=0
    for step in range(start,end):
        t=time.perf_counter();p=time.process_time()
        m.train();opt.zero_grad(set_to_none=True)
        rate=lr*(.1+.9*.5*(1+math.cos(math.pi*(step-start)/max(end-start-1,1))))
        for group in opt.param_groups:group['lr']=rate
        logits=m(x[step]);loss=F.cross_entropy(logits,y[step]);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        wall+=time.perf_counter()-t;cpu+=time.process_time()-p;count+=1
        if not math.isfinite(float(loss.detach())):raise RuntimeError('nonfinite loss')
        if step+1 in checkpoints or step+1==end:
            ck=path/f'step{step+1}.bin';ck.write_bytes(encode_model(m))
            ev=metrics(m,w,evaluate)
            row=dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(norm),wall=wall,cpu=cpu,metrics=ev,sha256=hashlib.sha256(ck.read_bytes()).hexdigest())
            record.append(row)
            print(path.name,step+1,round(ev[evaluate+'_acc'],4),ev['exact_rules'],round(wall,2),flush=True)
    torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'config':m.config},path/'resume.pt')
    return dict(records=record,wall_seconds=wall,cpu_seconds=cpu,updates=count,examples=count*len(x[0]),input_sha256=hashlib.sha256(x[start:end].numpy().tobytes()+y[start:end].numpy().tobytes()).hexdigest())


def prune_validation(m,w,keep_count=4):
    """Greedy leave-one-expert-out CE; no audit or true private mask access."""
    current=m;history=[];extra_seconds=0.
    while current.residual.private.n>keep_count:
        candidates=[]
        for j in range(current.residual.private.n):
            keep=[i for i in range(current.residual.private.n) if i!=j]
            t=time.perf_counter();candidate=compact(current,keep);ev=metrics(candidate,w,'dev');extra_seconds+=time.perf_counter()-t
            candidates.append((ev['dev_ce'],j,candidate,ev))
        loss,j,current,ev=min(candidates,key=lambda v:(v[0],v[1]))
        history.append(dict(ce=loss,remaining=current.residual.private.slot_ids.tolist(),dev_acc=ev['dev_acc']))
    return current,dict(history=history,selection_wall_seconds=extra_seconds,
                        candidate_evaluations=sum(range(keep_count+1,m.residual.private.n+1)),
                        note='dev atomic metrics recorded, but selection uses dev pair CE only')


def run_world(out,world_seed,init_seed,configs,lr=.003,fresh=False,pair_only=False,pruning=True,methods=None):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    w=make_world(world_seed);torch.save(w,out/'world.pt')
    x,y=stream(w,1600,init_seed+10000,pair_only)
    torch.save({'tokens':x,'targets':y},out/'stream.pt')
    base=make_model(BASE,init_seed)
    parent_log=train(base,w,x,y,0,400,lr,out/'parent',evaluate='dev')
    (out/'parent.bin').write_bytes(encode_model(base))
    results=[];use_eval='audit' if fresh else 'dev'
    for kind in (methods or KINDS):
        model=make_model(configs[kind],init_seed+10,base)
        r=train(model,w,x,y,400,1200,lr,out/kind,checkpoints=(600,800,1200),evaluate=use_eval)
        if kind=='hybrid' and pruning:
            parent=copy.deepcopy(model)
            selected,sel_log=prune_validation(parent,w,4)
            selected_ids=selected.residual.private.slot_ids.tolist()
            rng=np.random.default_rng(init_seed+20000);keep=sorted(rng.choice(8,4,replace=False).tolist())
            random_model=compact(parent,keep)
            for label,child in [('selected',selected),('random',random_model)]:
                immediate=metrics(child,w,use_eval)
                before=encode_model(child);(out/f'{label}_before.bin').write_bytes(before)
                log=train(child,w,x,y,1200,1600,lr/3,out/label,evaluate=use_eval)
                results.append(dict(kind=label,config=child.config,bytes=len(encode_model(child)),metrics=metrics(child,w,use_eval),before=immediate,training=log,selection=sel_log if label=='selected' else {'kept':keep}))
            c=dict(configs['hybrid'],experts=4)
            small=make_model(c,init_seed+10,base)
            log1=train(small,w,x,y,400,1200,lr,out/'small',checkpoints=(1200,),evaluate=use_eval)
            log2=train(small,w,x,y,1200,1600,lr/3,out/'small',evaluate=use_eval)
            results.append(dict(kind='small',config=c,bytes=len(encode_model(small)),metrics=metrics(small,w,use_eval),training=log1,continuation=log2))
        cont=train(model,w,x,y,1200,1600,lr/3,out/kind,evaluate=use_eval)
        results.append(dict(kind=kind,config=model.config,bytes=len(encode_model(model)),metrics=metrics(model,w,use_eval),training=r,continuation=cont))
    report=dict(world=world_seed,init=init_seed,lr=lr,pair_only=pair_only,parent=parent_log,results=results)
    (out/'result.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--world',type=int,required=True);p.add_argument('--init',type=int,required=True);p.add_argument('--lr',type=float,default=.003);p.add_argument('--fresh',action='store_true');p.add_argument('--pair-only',action='store_true');p.add_argument('--no-prune',action='store_true');p.add_argument('--config');p.add_argument('--methods',nargs='+',choices=KINDS)
    a=p.parse_args();c=json.loads(Path(a.config).read_text())['configs'] if a.config else select_configs()[0]
    run_world(a.out,a.world,a.init,c,a.lr,a.fresh,a.pair_only,not a.no_prune,a.methods)
