"""Post-hoc separate exploratory experiment, with paired teacher kernels across gamma."""
from __future__ import annotations
import os,sys,time,json,hashlib,copy,statistics,io,math
from pathlib import Path
import torch
import torch.nn.functional as F
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/'source'))
from core import Teacher,ConditionalMLP,dataset,K
from engine import eval_loss,activation_effect,swap_loss
HERE=Path(__file__).resolve().parent
PROTO=HERE/'PROTOCOL.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o):Path(p).write_text(json.dumps(o,indent=2,sort_keys=True)+'\n')
def inputs_for(teacher,seed,split,n):return dataset(teacher,seed,split,n)

def run_train():
    torch.set_num_threads(1);p=json.loads(PROTO.read_text())
    if (HERE/'SEAL.json').exists():raise RuntimeError('Already sealed')
    rows=[];start=time.perf_counter()
    for seed in p['seed_set_new']:
        for gamma in p['gamma_grid_fixed']:
            teacher=Teacher(seed,'relu',gamma)
            train=inputs_for(teacher,seed,'train',p['n_train']);dev=inputs_for(teacher,seed,'dev',p['n_dev'])
            for arm in p['arms']:
                model=ConditionalMLP(arm,'relu',initseed=40000+seed)
                optimizer=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
                gen=torch.Generator().manual_seed(900000+seed*19)
                best=float('inf');step_best=0;state_best=None
                for step in range(1,p['steps']+1):
                    ix=torch.randint(len(train[0]),(p['batch'],),generator=gen)
                    x,t,y=[v[ix] for v in train]
                    loss=F.mse_loss(model(x,t),y)
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                    if step%p['eval_every']==0:
                        with torch.no_grad():e=eval_loss(model,dev)
                        if e<best:best=e;step_best=step;state_best=copy.deepcopy(model.state_dict())
                model.load_state_dict(state_best)
                path=HERE/'checkpoints'/f'{seed}_{gamma:.2f}_{arm}.pt'
                torch.save(state_best,path)
                rows.append({'seed':seed,'gamma':gamma,'arm':arm,'step':step_best,'dev':eval_loss(model,dev),'ck':str(path),'sha':sha(path)})
        print('SWEEP TRAIN',len(rows),'of',len(p['seed_set_new'])*len(p['gamma_grid_fixed'])*len(p['arms']),flush=True)
    dump(HERE/'DEV_ONLY.json',rows)
    dump(HERE/'SEAL.json',{'type':p['type'],'fresh_generated':False,'rows':[{'ck':r['ck'],'sha':r['sha']} for r in rows],'protocol_sha256':sha(PROTO),'sweep_source_sha256':sha(__file__),'core_sha256':sha(BASE/'source'/'core.py'),'train_wall':time.perf_counter()-start})
    print('SWEEP SEALED',len(rows),flush=True)

def run_score():
    torch.set_num_threads(1);p=json.loads(PROTO.read_text());seal=json.loads((HERE/'SEAL.json').read_text())
    assert seal['fresh_generated'] is False and seal['protocol_sha256']==sha(PROTO) and seal['sweep_source_sha256']==sha(__file__) and seal['core_sha256']==sha(BASE/'source'/'core.py')
    rows=json.loads((HERE/'DEV_ONLY.json').read_text());assert len(rows)==140
    for r in rows:assert sha(r['ck'])==r['sha']
    new=[]
    for r in rows:
        teacher=Teacher(r['seed'],'relu',r['gamma']);model=ConditionalMLP(r['arm'],'relu',initseed=40000+r['seed'])
        model.load_state_dict(torch.load(r['ck'],weights_only=True));model.eval()
        dev=eval_loss(model,inputs_for(teacher,r['seed'],'dev',p['n_dev']))
        assert abs(dev-r['dev'])<1e-12
        fresh=eval_loss(model,inputs_for(teacher,r['seed'],'fresh',p['n_fresh']))
        ood=eval_loss(model,inputs_for(teacher,r['seed'],'ood',p['n_ood']))
        new.append({**r,'fresh':fresh,'ood':ood})
    dump(HERE/'FULL_RESULTS.json',new)
    table=[]
    for gamma in p['gamma_grid_fixed']:
        for arm in p['arms']:
            data=[r for r in new if r['gamma']==gamma and r['arm']==arm]
            table.append({'gamma':gamma,'arm':arm,'dev_median':statistics.median(r['dev'] for r in data),'fresh_median':statistics.median(r['fresh'] for r in data),'ood_median':statistics.median(r['ood'] for r in data)})
        mirror=sorted([r for r in new if r['gamma']==gamma and r['arm']=='mirror'],key=lambda x:x['seed'])
        for control in ('film','coupled','ind_budget'):
            ctrl=sorted([r for r in new if r['gamma']==gamma and r['arm']==control],key=lambda x:x['seed'])
            paired=[a['fresh']/b['fresh'] for a,b in zip(mirror,ctrl)]
            gm=math.exp(statistics.mean(map(math.log,paired)))
            print('SWEEP',gamma,'Mirror/',control,'gmean',round(gm,4),'wins10pct',sum(x<=.9 for x in paired),flush=True)
    dump(HERE/'ANALYSIS.json',{'type':p['type'],'summary':table,'rows':140})
    print('SWEEP scored',len(new),flush=True)

def verify():
    torch.set_num_threads(1);p=json.loads(PROTO.read_text());rows=json.loads((HERE/'FULL_RESULTS.json').read_text());seal=json.loads((HERE/'SEAL.json').read_text());assert seal['fresh_generated'] is False
    assert len(rows)==140; err=0
    for r in rows:
        assert sha(r['ck'])==r['sha']
        teacher=Teacher(r['seed'],'relu',r['gamma']);model=ConditionalMLP(r['arm'],'relu',initseed=40000+r['seed']);model.load_state_dict(torch.load(r['ck'],weights_only=True))
        for split in ('dev','fresh','ood'):
            v=eval_loss(model,inputs_for(teacher,r['seed'],split,p['n_'+split]))
            err=max(err,abs(v-r[split]))
    assert err<=1e-12
    dump(HERE/'VERIFICATION.json',{'rows_replayed':len(rows),'max_abs_error':err,'seal_sha256':sha(HERE/'SEAL.json'),'status':'POSTHOC_EXPLORATORY_ONLY'})
    print('SWEEP VERIFIED',len(rows),'err',err,flush=True)

if __name__=='__main__':
    if sys.argv[1]=='train':run_train()
    elif sys.argv[1]=='score':run_score()
    elif sys.argv[1]=='verify':verify()
    else:raise ValueError(sys.argv[1])
