"""MA-1175: preregistered, no-cut-aware gradient-VJP guided Mirror intervention.

Train never imports/constructs held-out fresh or OOD data. Audit is a separate CLI step.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, io, json, math, platform, time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from model import D, EDGES, NODES, VIEWS, LOOPS, DenseControl, VirtualGraph, new_model
from data import dataset, teacher, batch_schedule

ROOT=Path(__file__).resolve().parents[1]
CONFIG_BYTES=(ROOT/'PROTOCOL.json').read_bytes()
CONFIG=json.loads(CONFIG_BYTES)
PHASH=hashlib.sha256(CONFIG_BYTES).hexdigest()
WORLDS=CONFIG['data']['worlds'];SEEDS=CONFIG['data']['seeds']
METHODS=CONFIG['methods'];TRAIN=CONFIG['training']
TOTAL=TRAIN['total_real_updates']; EVENTS=TRAIN['cut_events'];BATCH=TRAIN['train_batch']
CUT=TRAIN['cut_four_edges_per_event']; MARGIN=TRAIN['dev_margin_nmse']
torch.set_num_threads(1)
CP=ROOT/'checkpoints'; RES=ROOT/'results'

@lru_cache(maxsize=32)
def train_data(world,split,seed):
    assert split in ('train','dev'),'FRESH SEALED before training completes'
    return dataset(world,split,seed,CONFIG['data'][split+'_n'])

def init_variant(method):
    if method=='native_dense':return 'native'
    if method=='direct_guard_long':return 'direct_tanh'
    return 'mirror'

def opt_new(m):
    return torch.optim.AdamW(m.parameters(),lr=TRAIN['lr'],weight_decay=TRAIN['weight_decay'])

def nmse_loss(m,pair,ids,denom,return_weights=False):
    x,y=pair
    if return_weights:
        pred,w,edge_ids=m(x[ids],return_weights=True)
        return (pred-y[ids]).square().mean()/denom,w,edge_ids
    return (m(x[ids])-y[ids]).square().mean()/denom

@torch.no_grad()
def quality(m,pair,denom):
    x,y=pair
    m.eval()
    return float(((m(x)-y)**2).mean()/denom)

def step(m,opt,train_pair,schedule,t,denom):
    m.train();opt.zero_grad(set_to_none=True)
    loss=nmse_loss(m,train_pair,schedule[t],denom)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),TRAIN['clip_grad_norm'])
    opt.step()
    return float(loss.detach())

def model_cost(m):
    return 2*D*18 if isinstance(m,DenseControl) else 2*D*D*LOOPS*m.active_count()

def core_vjp_per_edge(m,pair,ids,denom):
    """Exact shared physical core gradients g_e for each logical path.

    Computes dL/dW_e including all recurrent appearances of W_e, then
    (dW_e/dcore)^T * dL/dW_e; does NOT use gate gradient sign.
    """
    val, w, edge_ids = nmse_loss(m,pair,ids,denom,return_weights=True)
    g_w=torch.autograd.grad(val,w,retain_graph=True)[0]
    direct=torch.autograd.grad(val,m.core,retain_graph=True)[0].detach().flatten()
    parts=[]
    for n in range(w.shape[0]):
        part=torch.autograd.grad(w[n],m.core,grad_outputs=g_w[n],retain_graph=n+1<w.shape[0],allow_unused=False)[0]
        parts.append(part.detach().flatten())
    g=torch.stack(parts) if parts else direct.new_zeros((0,direct.numel()))
    rel_err=float((g.sum(0)-direct).norm()/(direct.norm()+1e-12))
    return edge_ids.detach().cpu().tolist(),g,rel_err

def intervention_groups(m,train_pair,sched,t,denom,world,seed,event,need_grad):
    eligible=m.active.nonzero().flatten().tolist()
    assert len(eligible)>=CUT
    salt=0 if world=='aligned_sparse_mirror' else 7591
    rng=np.random.default_rng(1175+seed*131+event*1009+salt)
    pick_random=tuple(sorted(int(v) for v in rng.choice(eligible,size=CUT,replace=False)))
    pick_gate=tuple(sorted(sorted(eligible,key=lambda i:(float(m.gates.detach()[i]),i))[:CUT]))
    if need_grad:
        a,ga,err_a=core_vjp_per_edge(m,train_pair,sched[t],denom)
        b,gb,err_b=core_vjp_per_edge(m,train_pair,sched[t+1],denom)
        if a!=b:raise AssertionError('edge IDs changed within VJP scan')
        g=(ga+gb)/2
        total=g.sum(0)
        gradnorm=g.norm(dim=-1)
        scores=[]
        for n,i in enumerate(a):
            other=total-g[n]; cost=float(g[n].norm()*other.norm())
            cosine=float((g[n]@other)/cost) if cost>1e-14 else 0.0
            rel=float(gradnorm[n]/(gradnorm.max()+1e-12))
            score= -cosine*float(gradnorm[n]) if (cosine<0 and rel>1e-5) else 0.0
            scores.append((i,score,cosine,float(gradnorm[n])))
        rank=sorted(scores,key=lambda x:(-x[1],x[0]))[:CUT]
        pick_conflict=tuple(sorted(x[0] for x in rank))
        pick_low_vjp=tuple(sorted(x[0] for x in sorted(scores,key=lambda x:(x[3],x[0]))[:CUT]))
        valid=sum(s[1]>0 for s in scores)
        assert max(err_a,err_b)<=1e-4,(err_a,err_b)
        neg_pairs=[]
        unit=g/(gradnorm[:,None]+1e-12)
        sims=unit@unit.T
        tri=torch.triu_indices(len(a),len(a),offset=1)
        if tri.shape[1]:negfrac=float((sims[tri[0],tri[1]]<0).float().mean())
        else:negfrac=0.0
        audit={'relative_vjp_error_max':max(err_a,err_b),'negative_grad_pair_fraction':negfrac,
               'anti_aligned_edge_count':valid,'top_scores':rank,
               'core_grad_norm':float(total.norm())}
    else:
        pick_conflict=tuple(sorted(eligible[:CUT]))
        pick_low_vjp=tuple(sorted(eligible[-CUT:]))
        audit={'relative_vjp_error_max':None,'negative_grad_pair_fraction':None,
               'anti_aligned_edge_count':None,'top_scores':[], 'core_grad_norm':None}
    groups={'true_shared_gradient_conflict':pick_conflict,'low_gate':pick_gate,
            'low_edge_vjp_norm':pick_low_vjp,'random':pick_random}
    return groups,audit

def probe(m,opt,removed,train_pair,sched,t,dev,denom,horizon):
    copy_m=copy.deepcopy(m)
    copy_opt=opt_new(copy_m)
    copy_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    if removed:copy_m.cut(removed)
    mac=0
    for k in range(t,t+horizon):
        mac+=model_cost(copy_m)*BATCH
        step(copy_m,copy_opt,train_pair,sched,k,denom)
    return quality(copy_m,dev,denom),mac

def core_rotation(m,group,train_pair,ids,denom):
    if not group:return 0.0
    l1=nmse_loss(m,train_pair,ids,denom)
    u=torch.autograd.grad(l1,m.core)[0].detach().flatten()
    before=m.active.clone()
    m.cut(group)
    l2=nmse_loss(m,train_pair,ids,denom)
    v=torch.autograd.grad(l2,m.core)[0].detach().flatten()
    with torch.no_grad():m.active.copy_(before)
    nv=float(u.norm()*v.norm())
    return math.degrees(math.acos(float(torch.clamp((u@v)/nv,-1.,1.)))) if nv>1e-15 else None

def decide(m,opt,method,world,seed,event,t,train_pair,dev,sched,denom):
    active_before=m.active_count()
    need_grad=method not in ('mirror_random','mirror_low_gate')
    groups,audit=intervention_groups(m,train_pair,sched,t,denom,world,seed,event,need_grad=need_grad)
    scores={};cost=0;used=0
    if method=='mirror_random':label='random'
    elif method=='mirror_low_gate':label='low_gate'
    elif method=='mirror_trueconflict':label='true_shared_gradient_conflict'
    else:
        assert method in ('mirror_guard_short','mirror_guard_long','direct_guard_long')
        H=TRAIN['probe_horizons_for_training_choice']['short' if method=='mirror_guard_short' else 'long']
        ref,c=probe(m,opt,(),train_pair,sched,t,dev,denom,H)
        scores['keep']=ref;used+=H;cost+=c
        for name,ids in groups.items():
            score,c=probe(m,opt,ids,train_pair,sched,t,dev,denom,H)
            scores[name]=score;used+=H;cost+=c
        best=min(groups,key=lambda k:(scores[k],k))
        label=best if scores[best] <= ref-MARGIN else None
    selected=list(groups[label]) if label else []
    angle=core_rotation(m,selected,train_pair,sched[t],denom)
    immediate_pre=quality(m,dev,denom)
    if selected:m.cut(selected)
    immediate_post=quality(m,dev,denom)
    return {'world':world,'seed':seed,'method':method,'event':event,'update':t,
            'candidates':{k:list(v) for k,v in groups.items()},'selected':label,
            'edges_cut':selected,'active_before':active_before,'active_after':m.active_count(),
            'reference_and_probes_dev_nmse':scores,'lookahead_probe_updates':used,
            'lookahead_mac_proxy':cost,'shared_core_grad_rotation_deg':angle,
            'immediate_dev_pre':immediate_pre,'immediate_dev_post':immediate_post,'vjp':audit}

def diagnostic_h120(m,opt,world,seed,t,train_pair,dev,sched,denom,first_event):
    groups,diag=intervention_groups(m,train_pair,sched,t,denom,world,seed,first_event,True)
    outputs={};totalmac=0
    for name,cut_ids in [('keep',()),*groups.items()]:
        v,c=probe(m,opt,cut_ids,train_pair,sched,t,dev,denom,120)
        outputs[name]=v;totalmac+=c
    return {'world':world,'seed':seed,'t':t,'H':120,'DEV_ONLY':True,
            'dev_scores':outputs,'candidate_groups':{k:list(v) for k,v in groups.items()},
            'diagnostic_probe_updates':600,'diagnostic_mac_proxy':totalmac,'vjp':diag}

def ckpt_path(world,seed,method):return CP/f'{world}_{seed}_{method}.pt'

def train_one(world,seed,method):
    variant=init_variant(method)
    m=new_model(variant,seed)
    opt=opt_new(m)
    tr=train_data(world,'train',seed);dev=train_data(world,'dev',seed)
    den=float(tr[1].square().mean().clamp_min(1e-8))
    n=TOTAL+400 if method=='mirror_keep_extra' else TOTAL
    schedule=batch_schedule(len(tr[0]),seed,n+120+41,BATCH)
    interventions=[];diagnostics=[];probes=0;diag_updates=0
    trainmac=0;probemac=0;diagmac=0
    t0=time.perf_counter();first=None
    for t in range(n):
        if t in EVENTS and method in ('mirror_random','mirror_low_gate','mirror_trueconflict','mirror_guard_short','mirror_guard_long','direct_guard_long'):
            if method=='mirror_guard_long' and t==EVENTS[0]:
                # Independent diagnostic: never feeds the selection policy; computed before mask change.
                diag=diagnostic_h120(m,opt,world,seed,t,tr,dev,schedule,den,0)
                diagnostics.append(diag);diag_updates+=diag['diagnostic_probe_updates'];diagmac+=diag['diagnostic_mac_proxy']
            ev=decide(m,opt,method,world,seed,EVENTS.index(t),t,tr,dev,schedule,den)
            interventions.append(ev);probes+=ev['lookahead_probe_updates'];probemac+=ev['lookahead_mac_proxy']
        trainmac+=model_cost(m)*BATCH
        v=step(m,opt,tr,schedule,t,den)
        if first is None:first=v
    packet_bytes=m.packet_bytes()
    row={'world':world,'seed':seed,'method':method,'kind':variant,'real_updates':n,
         'probe_updates':probes,'diagnostic_probe_updates':diag_updates,
         'train_mlp_mac_proxy':trainmac,'probe_mlp_mac_proxy':probemac,
         'diagnostic_mlp_mac_proxy':diagmac,'total_accounted_mlp_mac_proxy':trainmac+probemac+diagmac,
         'trainable_float_coefficients':sum(p.numel() for p in m.parameters()),
         'active_edges':-1 if isinstance(m,DenseControl) else m.active_count(),
         'packet_bytes':packet_bytes,'dev_nmse':quality(m,dev,den),
         'train_nmse':quality(m,tr,den),'initial_batch_nmse':first,
         'train_wall_s':time.perf_counter()-t0}
    ck={'protocol_sha':PHASH,'row':row,'model':m.state_dict(),'optimizer':opt.state_dict(),
        'denom':den,'events':interventions,'diagnostics':diagnostics}
    CP.mkdir(exist_ok=True);torch.save(ck,ckpt_path(world,seed,method))
    return row

def from_start(world,seed):
    original=torch.load(ckpt_path(world,seed,'mirror_guard_long'),map_location='cpu',weights_only=False)
    mask=original['model']['active'].clone()
    m=new_model('mirror',seed);m.active.copy_(mask)
    opt=opt_new(m)
    tr=train_data(world,'train',seed);dev=train_data(world,'dev',seed)
    den=float(tr[1].square().mean().clamp_min(1e-8));schedule=batch_schedule(len(tr[0]),seed,TOTAL+121,BATCH)
    cost=0;t0=time.perf_counter()
    for t in range(TOTAL):
        cost+=model_cost(m)*BATCH
        step(m,opt,tr,schedule,t,den)
    row={'world':world,'seed':seed,'method':'mirror_from_start_mask','kind':'mirror','real_updates':TOTAL,
        'probe_updates':0,'diagnostic_probe_updates':0,'train_mlp_mac_proxy':cost,
        'probe_mlp_mac_proxy':0,'diagnostic_mlp_mac_proxy':0,
        'total_accounted_mlp_mac_proxy':cost,'trainable_float_coefficients':sum(p.numel() for p in m.parameters()),
        'active_edges':m.active_count(),'packet_bytes':m.packet_bytes(),
        'dev_nmse':quality(m,dev,den),'train_nmse':quality(m,tr,den),'initial_batch_nmse':None,
        'train_wall_s':time.perf_counter()-t0}
    torch.save({'protocol_sha':PHASH,'row':row,'model':m.state_dict(),'optimizer':opt.state_dict(),
                'denom':den,'events':[],'diagnostics':[],
                'final_mask_source':'mirror_guard_long'},ckpt_path(world,seed,'mirror_from_start_mask'))
    return row

def train():
    CP.mkdir(exist_ok=True);RES.mkdir(exist_ok=True)
    print('MA1175 PREREG SHA',PHASH,'FRESH SEALED',flush=True)
    jobs=[(w,s,m) for w in WORLDS for s in SEEDS for m in METHODS]
    for index,(world,seed,method) in enumerate(jobs):
        path=ckpt_path(world,seed,method)
        if path.exists():
            print('SKIP EXISTING',index+1,len(jobs),path.name,flush=True);continue
        row=from_start(world,seed) if method=='mirror_from_start_mask' else train_one(world,seed,method)
        print('TRAIN',index+1,len(jobs),world,seed,method,'dev',round(row['dev_nmse'],6),
              'edges',row['active_edges'],'real',row['real_updates'],'probe',row['probe_updates'],
              'diag',row['diagnostic_probe_updates'],'sec',round(row['train_wall_s'],2),flush=True)
    assert all(ckpt_path(w,s,m).exists() for w,s,m in jobs)
    artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ckpt_path(w,s,m) for w,s,m in jobs]}
    done={'id':'MA-1175','all_complete':True,'checkpoint_count':len(jobs),
          'protocol_sha':PHASH,'fresh_accessed':False,'checkpoint_sha256':artifacts}
    (RES/'TRAIN_COMPLETE.json').write_text(json.dumps(done,indent=2)+'\n')
    print('COMPLETE',len(jobs),'FRESH SEALED until AUDIT',flush=True)

@torch.no_grad()
def timing(m,b,seed):
    gen=torch.Generator().manual_seed(seed+30017+b)
    x=torch.randn((b,D),generator=gen)
    m.eval();n=35
    for _ in range(10):m(x)
    t=[]
    for _ in range(n):
        t0=time.perf_counter_ns();m(x);t.append((time.perf_counter_ns()-t0)/1000)
    return float(np.median(t))

def audit():
    file=RES/'TRAIN_COMPLETE.json'
    assert file.exists(),'Cannot access fresh before COMPLETE manifest'
    manifest=json.loads(file.read_text());assert manifest['all_complete']
    assert manifest['protocol_sha']==PHASH
    jobs=[(w,s,m) for w in WORLDS for s in SEEDS for m in METHODS]
    assert manifest['checkpoint_count']==len(jobs)
    for w,s,m in jobs:
        p=ckpt_path(w,s,m)
        assert manifest['checkpoint_sha256'][p.name]==hashlib.sha256(p.read_bytes()).hexdigest()
    rows=[];interventions=[];diags=[]
    for w in WORLDS:
        for seed in SEEDS:
            fresh=dataset(w,'fresh',seed,CONFIG['data']['fresh_n'])
            ood=dataset(w,'fresh_ood',seed,CONFIG['data']['ood_n'])
            dev=train_data(w,'dev',seed)
            for method in METHODS:
                path=ckpt_path(w,seed,method)
                ck=torch.load(path,map_location='cpu',weights_only=False)
                assert ck['protocol_sha']==PHASH
                m=new_model(ck['row']['kind'],seed);m.load_state_dict(ck['model'])
                expected=ck['row']['dev_nmse']
                assert abs(quality(m,dev,ck['denom'])-expected)<1e-6,(w,seed,method)
                if isinstance(m,VirtualGraph):
                    packet=m.packet()
                    q=new_model(m.kind,seed+991);q.reload_packet(packet)
                    with torch.no_grad():
                        assert torch.allclose(m(fresh[0][:16]),q(fresh[0][:16]),atol=1e-6)
                    bytes_fixed=sum(t.numel()*t.element_size() for t in (m.perms,m.signs,m.house,m.bases,m.src,m.dst))
                    virtual_runtime=m.active_count()*2*D*D*4
                else:bytes_fixed=virtual_runtime=0
                row={**ck['row'],'fresh_nmse':quality(m,fresh,ck['denom']),
                     'ood_nmse':quality(m,ood,ck['denom']),
                     'forward_b1_us':timing(m,1,seed),'forward_b64_us':timing(m,64,seed),
                     'runtime_fixed_program_bytes':bytes_fixed,
                     'temporary_decoded_weights_bytes':virtual_runtime,
                     'active_mlp_mac_per_example':model_cost(m),
                     'checkpoint_sha256':manifest['checkpoint_sha256'][path.name]}
                rows.append(row)
                interventions+=ck['events'];diags+=ck['diagnostics']
                print('AUDIT',w,seed,method,'fresh',round(row['fresh_nmse'],5),'ood',round(row['ood_nmse'],5),flush=True)
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (RES/'INTERVENTIONS.json').write_text(json.dumps(interventions,indent=2)+'\n')
    (RES/'H120_DEV_ONLY.json').write_text(json.dumps(diags,indent=2)+'\n')
    comparisons=[]
    for w in WORLDS:
        for seed in SEEDS:
            x={r['method']:r for r in rows if r['world']==w and r['seed']==seed}
            y=x['mirror_guard_long']['fresh_nmse']
            comparisons.append({'world':w,'seed':seed,
               'long_to_short':y/x['mirror_guard_short']['fresh_nmse'],
               'long_to_keep':y/x['mirror_keep']['fresh_nmse'],
               'long_to_random':y/x['mirror_random']['fresh_nmse'],
               'long_to_startmask':y/x['mirror_from_start_mask']['fresh_nmse'],
               'long_to_direct':y/x['direct_guard_long']['fresh_nmse'],
               'long_to_extra':y/x['mirror_keep_extra']['fresh_nmse'],
               'learning_gate':y<=.95*x['mirror_guard_short']['fresh_nmse'] and y<=.95*x['mirror_keep']['fresh_nmse'],
               'mirror_gate':y<=.95*x['direct_guard_long']['fresh_nmse']})
    (RES/'PAIRED_RESULTS.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    return rows,comparisons,interventions,diags

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','audit'])
    mode=ap.parse_args().mode
    if mode=='train':train()
    else:audit()
