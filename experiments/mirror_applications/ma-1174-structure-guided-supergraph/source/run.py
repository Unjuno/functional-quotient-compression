"""MA-1174 frozen structure-intervention experiment: train has no fresh access."""
from __future__ import annotations
import argparse,copy,csv,hashlib,io,json,math,os,time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import D,NODES,VIEWS,EDGES,LOOPS,new_model,VirtualGraph,DenseControl
from data import dataset,teacher,batch_schedule

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'PROTOCOL.json').read_text())
P_SHA=hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest()
RESULTS=ROOT/'results';CHECKPOINTS=ROOT/'checkpoints'
SEEDS=CFG['seeds'];WORLDS=CFG['worlds'];METHODS=CFG['methods']
UPDATES=CFG['training']['updates'];EVENTS=CFG['training']['events'];BATCH=CFG['training']['batch']
LOOK=CFG['training']['lookahead_updates_per_candidate'];CUT=CFG['training']['cut_count_per_event']
torch.set_num_threads(1)

@lru_cache(maxsize=16)
def samples(world,split,seed):
    assert split in ('train','dev'),'TRAIN SEALED: fresh data may not be accessed'
    count=CFG['data']['train_n'] if split=='train' else CFG['data']['dev_n']
    return dataset(world,split,seed,count)

def variant(method):
    if method=='native_dense':return 'native'
    if method.startswith('affine_'):return 'affine'
    if method.startswith('direct_tanh_'):return 'direct_tanh'
    return 'mirror'

def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=CFG['training']['learning_rate'],weight_decay=CFG['training']['weight_decay'])

def loss(m,data,indices,denom):
    x,y=data
    return (m(x[indices])-y[indices]).square().mean()/denom

def train_step(m,opt,train,ids,denom):
    m.train(); opt.zero_grad(set_to_none=True)
    err=loss(m,train,ids,denom)
    err.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),CFG['training']['gradient_clip'])
    opt.step()
    return float(err.detach())

@torch.no_grad()
def metric(m,data,denom):
    m.eval();x,y=data
    return float((m(x)-y).square().mean()/denom)

def mask_indices(m):return m.active.nonzero(as_tuple=False).flatten().tolist()

def cost_proxy(m):
    if isinstance(m,DenseControl):return 2*D*18
    return m.active_count()*2*D*D*LOOPS

def per_batch_grad(m,train,ids,denom,parameter):
    val=loss(m,train,ids,denom)
    g=torch.autograd.grad(val,parameter,retain_graph=False,allow_unused=True)[0]
    return torch.zeros_like(parameter) if g is None else g.detach().clone()

def shortlist(m,train,sched,t,denom,seed,world,event):
    eligible=mask_indices(m)
    # Two disjoint training minibatches; this tests gradient *sign*, not a proven causal interference estimator.
    g1=per_batch_grad(m,train,sched[t],denom,m.gates)
    g2=per_batch_grad(m,train,sched[t+1],denom,m.gates)
    gavg=(g1+g2)/2
    conflict=(-g1*g2).clamp_min(0)
    # Tie breaker: large task-sensitive gradients win if no disagreement signal exists.
    conflict_score=conflict + 1e-8*gavg.abs()
    picks_conflict=sorted(eligible,key=lambda i:(-float(conflict_score[i]),i))[:CUT]
    picks_gate=sorted(eligible,key=lambda i:(float(m.gates[i].detach()),i))[:CUT]
    picks_taylor=sorted(eligible,key=lambda i:(float(gavg[i].abs()),i))[:CUT]
    salt=0 if world=='aligned_sparse_mirror' else 765431
    rng=np.random.default_rng(seed*31+event*103+salt+99233)
    picks_random=sorted(rng.choice(eligible,size=CUT,replace=False).tolist())
    groups=[tuple(sorted(x)) for x in (picks_conflict,picks_gate,picks_taylor,picks_random)]
    cross=float(((g1[eligible]*g2[eligible])<0).float().mean())
    return {'conflict':groups[0],'low_gate':groups[1],'low_taylor':groups[2],'random':groups[3]},cross

def trial(m,opt,cut_group,train,sched,t,dev,denom):
    a=copy.deepcopy(m)
    op=optimizer(a);op.load_state_dict(copy.deepcopy(opt.state_dict()))
    if cut_group:a.cut(cut_group)
    work=0
    for k in range(t,t+LOOK):
        train_step(a,op,train,sched[k],denom)
        work+=cost_proxy(a)*BATCH
    return metric(a,dev,denom),work

def gradient_rotation(m,group,train,ids,denom):
    if not group:return None
    before=per_batch_grad(m,train,ids,denom,m.core).flatten()
    mask_before=m.active.clone()
    m.cut(group)
    after=per_batch_grad(m,train,ids,denom,m.core).flatten()
    m.active.copy_(mask_before)
    norm=before.norm()*after.norm()
    if norm.item()<1e-20:return None
    return math.degrees(math.acos(float(torch.clamp((before@after)/norm,-1,1))))

def intervention(m,opt,method,train,dev,sched,t,denom,seed,world,event):
    before_dev=metric(m,dev,denom)
    groups,conflict_fraction=shortlist(m,train,sched,t,denom,seed,world,event)
    choice=None;scored={};probes=0;probe_cost=0;reference=None
    if method=='mirror_random':choice='random'
    elif method=='mirror_low_gate':choice='low_gate'
    elif method=='mirror_conflict':choice='conflict'
    else:
        assert method in ('mirror_lookahead','mirror_guarded','direct_tanh_lookahead','affine_lookahead')
        reference,c=trial(m,opt,(),train,sched,t,dev,denom);probes+=LOOK;probe_cost+=c
        for name,edges in groups.items():
            score,c=trial(m,opt,edges,train,sched,t,dev,denom)
            scored[name]=score;probes+=LOOK;probe_cost+=c
        choice=min(scored,key=lambda k:(scored[k],k))
        if method=='mirror_guarded' and scored[choice] >= reference-0.0005:choice=None
    removed=list(groups[choice]) if choice is not None else []
    theta_grad_angle=gradient_rotation(m,removed,train,sched[t],denom)
    if removed:m.cut(removed)
    return {'event':event,'update':t,'policy':method,'groups':{k:list(v) for k,v in groups.items()},
            'gradient_sign_disagreement':conflict_fraction,'scores':scored,'keep_probe_dev':reference,
            'chosen':choice,'cut_edges':removed,'immediate_dev_before':before_dev,
            'immediate_dev_after':metric(m,dev,denom),'core_gradient_rotation_deg':theta_grad_angle,
            'probe_updates':probes,'probe_mac_lower_bound':probe_cost,
            'active_after':m.active_count()}

def cpath(world,seed,method):return CHECKPOINTS/f'{world}_{seed}_{method}.pt'

def train_one(world,seed,method):
    t0=time.perf_counter()
    m=new_model(variant(method),seed)
    opt=optimizer(m)
    tr=samples(world,'train',seed);dev=samples(world,'dev',seed)
    denom=float(tr[1].square().mean().clamp_min(1e-8))
    num=CFG['training']['extra_update_control'] if method=='mirror_keep_extra' else UPDATES
    sched=batch_schedule(len(tr[0]),seed,num+LOOK,BATCH)
    events=[]; probes=0; pcost=0;real_cost=0
    loss0=None
    for t in range(num):
        if t in EVENTS and method not in ('mirror_keep','mirror_keep_extra','native_dense'):
            event=intervention(m,opt,method,tr,dev,sched,t,denom,seed,world,EVENTS.index(t))
            events.append(event);probes+=event['probe_updates'];pcost+=event['probe_mac_lower_bound']
        real_cost+=cost_proxy(m)*BATCH
        lossval=train_step(m,opt,tr,sched[t],denom)
        if t==0:loss0=lossval
    row={'world':world,'seed':seed,'method':method,'variant':variant(method),
         'updates':num,'probe_updates':probes,'real_mlp_mac_lower_bound':real_cost,
         'probe_mlp_mac_lower_bound':pcost,
         'physical_trainable_coefficients':sum(p.numel() for p in m.parameters()),
         'remaining_edges':m.active_count() if isinstance(m,VirtualGraph) else -1,
         'deployment_bytes':m.packet_bytes(),'dev_nmse':metric(m,dev,denom),
         'train_nmse':metric(m,tr,denom),'train0_loss':loss0,
         'train_elapsed_sec':time.perf_counter()-t0}
    state={'protocol_sha':P_SHA,'row':row,'events':events,'model':m.state_dict(),
           'optimizer':opt.state_dict(),'denom':denom,
           'active_mask':m.active.detach().cpu().clone() if isinstance(m,VirtualGraph) else None}
    torch.save(state,cpath(world,seed,method))
    return row

def train_same_start_mask(world,seed):
    source=torch.load(cpath(world,seed,'mirror_lookahead'),map_location='cpu',weights_only=False)
    if source['row']['updates']!=UPDATES:raise ValueError('invalid lookahead checkpoint')
    mask=source['active_mask']
    m=new_model('mirror',seed)
    m.active.copy_(mask)
    opt=optimizer(m)
    tr=samples(world,'train',seed);dev=samples(world,'dev',seed);denom=float(tr[1].square().mean().clamp_min(1e-8))
    sched=batch_schedule(len(tr[0]),seed,UPDATES+LOOK,BATCH)
    cost=0;t0=time.perf_counter()
    for t in range(UPDATES):
        cost+=cost_proxy(m)*BATCH
        train_step(m,opt,tr,sched[t],denom)
    row={'world':world,'seed':seed,'method':'mirror_from_start_same_mask','variant':'mirror','updates':UPDATES,
         'probe_updates':0,'real_mlp_mac_lower_bound':cost,'probe_mlp_mac_lower_bound':0,
         'physical_trainable_coefficients':sum(p.numel() for p in m.parameters()),
         'remaining_edges':m.active_count(),'deployment_bytes':m.packet_bytes(),
         'dev_nmse':metric(m,dev,denom),'train_nmse':metric(m,tr,denom),
         'train0_loss':None,'train_elapsed_sec':time.perf_counter()-t0}
    torch.save({'protocol_sha':P_SHA,'row':row,'events':[],'model':m.state_dict(),'optimizer':opt.state_dict(),
                'denom':denom,'active_mask':mask.detach().clone()},cpath(world,seed,row['method']))
    return row

def train_all(world_filter=None,seed_filter=None,method_filter=None):
    RESULTS.mkdir(exist_ok=True);CHECKPOINTS.mkdir(exist_ok=True)
    print('TRAIN ONLY: fresh not generated; preregistered protocol SHA',P_SHA,flush=True)
    for world in WORLDS:
        if world_filter is not None and world!=world_filter:continue
        for seed in SEEDS:
            if seed_filter is not None and seed!=seed_filter:continue
            for method in METHODS:
                if method_filter is not None and method!=method_filter:continue
                p=cpath(world,seed,method)
                if p.exists():
                    print('skip',p.name,flush=True);continue
                row=train_one(world,seed,method)
                print('train',world,seed,method,'dev',round(row['dev_nmse'],6),'edges',row['remaining_edges'],
                      'probes',row['probe_updates'],'sec',round(row['train_elapsed_sec'],2),flush=True)
            p=cpath(world,seed,'mirror_from_start_same_mask')
            if not p.exists() and (method_filter is None or method_filter=='mirror_from_start_same_mask') and cpath(world,seed,'mirror_lookahead').exists():
                row=train_same_start_mask(world,seed)
                print('train',world,seed,row['method'],'dev',round(row['dev_nmse'],6),'edges',row['remaining_edges'],flush=True)
    files=[cpath(w,s,m) for w in WORLDS for s in SEEDS for m in METHODS+['mirror_from_start_same_mask']]
    if not all(f.exists() for f in files):
        print('TRAIN PARTIAL',sum(f.exists() for f in files),'/',len(files),'no fresh',flush=True)
        return
    complete={'checkpoint_count':len(files),'all_complete':True,'protocol_sha':P_SHA,'fresh_accessed':False,
              'checkpoints':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (RESULTS/'TRAIN_COMPLETE.json').write_text(json.dumps(complete,indent=2)+'\n')
    print('TRAIN ALL COMPLETE',len(files),'fresh never used',flush=True)

@torch.inference_mode()
def latency(m,batch=1,reps=51,warm=12):
    x=torch.randn(batch,D)
    m.eval()
    for _ in range(warm):m(x)
    measurements=[]
    for _ in range(reps):
        t=time.perf_counter_ns();m(x);measurements.append((time.perf_counter_ns()-t)/1000)
    return float(np.median(measurements))

def oracle_check(world,seed):
    if world!='aligned_sparse_mirror':return None
    t=teacher(world,seed);s=new_model('mirror',123)
    s.load_state_dict(t.state_dict())
    x,_=dataset(world,'dev',seed,128)
    with torch.no_grad():return float((s(x)-t(x)).abs().max())

def audit():
    assert (RESULTS/'TRAIN_COMPLETE.json').exists(),'Fresh evaluation forbidden until all checkpoints committed'
    complete=json.loads((RESULTS/'TRAIN_COMPLETE.json').read_text())
    assert complete['all_complete'] and complete['checkpoint_count']==len(WORLDS)*len(SEEDS)*(len(METHODS)+1)
    assert complete['protocol_sha']==P_SHA
    results=[];events=[];oracle={}
    for w in WORLDS:
        for seed in SEEDS:
            # First time fresh / OOD is constructed: after all training files + hashes exist.
            fresh=dataset(w,'fresh',seed,CFG['data']['fresh_n'])
            ood=dataset(w,'fresh_ood',seed,CFG['data']['fresh_ood_n'])
            dev=samples(w,'dev',seed)
            oracle[f'{w}_{seed}']=oracle_check(w,seed)
            for method in METHODS+['mirror_from_start_same_mask']:
                path=cpath(w,seed,method)
                assert hashlib.sha256(path.read_bytes()).hexdigest()==complete['checkpoints'][path.name]
                ck=torch.load(path,map_location='cpu',weights_only=False)
                assert ck['protocol_sha']==P_SHA
                m=new_model(ck['row']['variant'],seed)
                m.load_state_dict(ck['model'])
                assert abs(metric(m,dev,ck['denom'])-ck['row']['dev_nmse'])<1e-6
                if isinstance(m,VirtualGraph):
                    packet=m.packet();n=new_model(m.kind,seed+11);n.reload_packet(packet)
                    with torch.no_grad():assert torch.allclose(n(fresh[0][:32]),m(fresh[0][:32]),atol=1e-6)
                    buffers=sum(t.numel()*t.element_size() for t in (m.perms,m.signs,m.house,m.bases,m.src,m.dst))
                    decoded=m.active_count()*2*D*D*4
                else:buffers=decoded=0
                row={**ck['row'],'fresh_nmse':metric(m,fresh,ck['denom']),'ood_nmse':metric(m,ood,ck['denom']),
                     'checkpoint_sha256':complete['checkpoints'][path.name],
                     'mac_per_sample':cost_proxy(m),'fixed_buffer_bytes':buffers,'decoded_runtime_bytes':decoded,
                     'latency_B1_us':latency(m,batch=1),'latency_B64_us':latency(m,batch=64)}
                results.append(row)
                for rec in ck['events']:events.append({'world':w,'seed':seed,'method':method,**rec})
                print('audit',w,seed,method,'fresh',round(row['fresh_nmse'],6),'ood',round(row['ood_nmse'],6),flush=True)
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    (RESULTS/'INTERVENTIONS.json').write_text(json.dumps(events,indent=2)+'\n')
    (RESULTS/'ORACLE_REPRESENTABILITY.json').write_text(json.dumps(oracle,indent=2)+'\n')
    # Comparisons specified BEFORE fresh opening (PROTOCOL.json).
    paired=[]
    from statistics import median
    for w in WORLDS:
        for seed in SEEDS:
            by={r['method']:r for r in results if r['world']==w and r['seed']==seed}
            target=by['mirror_lookahead']['fresh_nmse']
            pair={'world':w,'seed':seed,
                  'lookahead_fresh':target,
                  'ratio_vs_random':target/by['mirror_random']['fresh_nmse'],
                  'ratio_vs_low_gate':target/by['mirror_low_gate']['fresh_nmse'],
                  'ratio_vs_from_start':target/by['mirror_from_start_same_mask']['fresh_nmse'],
                  'ratio_vs_direct_tanh':target/by['direct_tanh_lookahead']['fresh_nmse'],
                  'ratio_vs_affine':target/by['affine_lookahead']['fresh_nmse']}
            pair['trajectory_pass']=all(pair[k]<=0.95 for k in ('ratio_vs_random','ratio_vs_low_gate','ratio_vs_from_start'))
            pair['mirror_specific_pass']=all(pair[k]<=0.95 for k in ('ratio_vs_direct_tanh','ratio_vs_affine'))
            paired.append(pair)
    summary={'n_checkpoints':len(results),'n_pairs':len(paired),
             'trajectory_gate_wins':sum(x['trajectory_pass'] for x in paired),
             'mirror_specific_gate_wins':sum(x['mirror_specific_pass'] for x in paired),
             'trajectory_pass':sum(x['trajectory_pass'] for x in paired)>=5,
             'mirror_specific_pass':sum(x['mirror_specific_pass'] for x in paired)>=5,
             'method_medians':{m:{k:round(median(r[k] for r in results if r['method']==m),7) for k in ('fresh_nmse','ood_nmse','remaining_edges','deployment_bytes','latency_B1_us','probe_updates','train_elapsed_sec')} for m in METHODS+['mirror_from_start_same_mask']},
             'paired':paired,'oracle_max_abs_error':oracle,
             'protocol_sha':P_SHA,'frozen_fresh_after_all_training':True}
    # Gradient-angle mechanism separately from quality. Zero-cut guarded records report null.
    mir=[e for e in events if e['method']=='mirror_lookahead']
    summary['rotation_degrees_by_pair']={f'{w}_{s}':[e['core_gradient_rotation_deg'] for e in mir if e['world']==w and e['seed']==s] for w in WORLDS for s in SEEDS}
    mech=sum(any(v is not None and v>=5 for v in angles) for angles in summary['rotation_degrees_by_pair'].values())
    summary['gradient_rotation_pair_ge5deg']=mech
    summary['mechanism_rotation_pass']=mech>=4
    (RESULTS/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('GATES',json.dumps({k:summary[k] for k in ('trajectory_gate_wins','mirror_specific_gate_wins','trajectory_pass','mirror_specific_pass','gradient_rotation_pair_ge5deg')},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','audit']);p.add_argument('--world',choices=WORLDS);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--method',choices=METHODS+['mirror_from_start_same_mask']);args=p.parse_args()
    if args.phase=='train':train_all(args.world,args.seed,args.method)
    else:audit()
