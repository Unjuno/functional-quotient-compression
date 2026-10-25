"""MA-1179: preregistered correlated independently pretrained MLP family distillation.

Run: python source/run.py --out <experiment_folder>
Requires PyTorch CPU.  Saves/hashes ALL checkpoints before Fresh/OOD generation.
Only the frozen PROTOCOL.json defines decisions.  No Fresh-based hyperparameter selection.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

D,H,R=4,16,3
SEEDS=(101,211,307,401,503)
WORLDS=('mirror_positive','correlated_independent','unrelated_independent')
ARMS=('mirror_cold','mirror_warm','film_cold','film_warm','rank1_cold','rank1_warm','independent_cold','shared_cold')
BATCH=128
TEACHER_STEPS=700
WARM_STEPS=250
TOTAL_STEPS=750


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<19),b''):h.update(block)
    return h.hexdigest()


def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')


def count(model):return sum(p.numel() for p in model.parameters())


def reflection(x,v):
    den=(v*v).sum(dim=-1,keepdim=True).clamp_min(1e-12)
    return x - 2*v*((x*v).sum(dim=-1,keepdim=True)/den)


def householder(v):
    den=v.square().sum().clamp_min(1e-12)
    return torch.eye(len(v),dtype=v.dtype)-2*torch.outer(v,v)/den


class DenseMLP(nn.Module):
    def __init__(self,h=H):
        super().__init__()
        self.up=nn.Linear(D,h)
        self.down=nn.Linear(h,D)
    def forward(self,x):return self.down(torch.tanh(self.up(x)))


class SharedStudent(nn.Module):
    def __init__(self,kind:str):
        super().__init__()
        assert kind in ('mirror','film','rank1','shared')
        self.kind=kind
        self.up=nn.Linear(D,H)
        self.down=nn.Linear(H,D)
        if kind!='shared':self.codes=nn.Parameter(torch.randn(R,H)*0.45)
    def forward(self,x,rule):
        a=self.up(x)
        if self.kind=='mirror':
            v=self.codes[rule]
            y=reflection(torch.tanh(reflection(a,v)),v)
        elif self.kind=='film':
            y=torch.tanh(a*(1+0.85*torch.tanh(self.codes[rule])))
        elif self.kind=='rank1':
            v=self.codes[rule]
            proj=v*((a*v).sum(dim=-1,keepdim=True)/v.square().sum(dim=-1,keepdim=True).clamp_min(1e-12))
            y=torch.tanh(a+0.85*proj)
        else:y=torch.tanh(a)
        return self.down(y)


class IndependentStudent(nn.Module):
    """Three ordinary independent 4->7->4 MLPs; 201 coefficients total."""
    def __init__(self):
        super().__init__()
        h=7
        self.up=nn.Parameter(torch.randn(R,h,D)*math.sqrt(2/(D+h)))
        self.up_bias=nn.Parameter(torch.zeros(R,h))
        self.down=nn.Parameter(torch.randn(R,D,h)*math.sqrt(2/(D+h)))
        self.down_bias=nn.Parameter(torch.zeros(R,D))
    def forward(self,x,rule):
        y=torch.einsum('bhd,bd->bh',self.up[rule],x)+self.up_bias[rule]
        return torch.einsum('bdh,bh->bd',self.down[rule],torch.tanh(y))+self.down_bias[rule]


def new_student(arm,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return IndependentStudent() if arm=='independent' else SharedStudent(arm)


def teacher_normal(x,task,seed):
    """Fixed nonlinear targets; common task==0, specific task==1,2,3."""
    g=torch.Generator().manual_seed(73_000_000+seed*1307+task*4099)
    w=torch.randn(12,D,generator=g)*1.18
    b=torch.randn(12,generator=g)*.35
    v=torch.randn(D,12,generator=g)*.25
    return torch.tanh(x@w.T+b)@v.T+0.15*torch.sin(x.square()@w.square().T)@v.T


def draw(seed,n,scale=1.):
    g=torch.Generator().manual_seed(int(seed))
    return (torch.rand(n,D,generator=g)*2-1)*scale,torch.randint(R,(n,),generator=g)


def teacher_fit(seed,task,kind,dry=False):
    init=11_000_000+seed*101+task*31+(0 if kind=='common' else (1 if kind=='correlated' else 2))*5009
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init)
        t=DenseMLP()
    xs,_=draw(17_000_000+seed*100+task*1000+(kind=='unrelated')*400+(kind=='correlated')*800,768 if dry else 3072)
    xd,_=draw(18_000_000+seed*100+task*1000+(kind=='unrelated')*400+(kind=='correlated')*800,256 if dry else 768)
    def target(x):
        return .75*teacher_normal(x,0,seed)+.25*teacher_normal(x,task,seed) if kind=='correlated' else teacher_normal(x,task,seed)
    ys=target(xs);yd=target(xd)
    gen=torch.Generator().manual_seed(19_000_000+seed*100+task*1000+(kind=='unrelated')*400+(kind=='correlated')*800)
    steps=8 if dry else TEACHER_STEPS
    idx=torch.randint(len(xs),(steps,BATCH),generator=gen)
    opt=torch.optim.AdamW(t.parameters(),lr=.012,weight_decay=1e-4)
    start=time.perf_counter();best=float('inf');best_state=None
    for i in range(steps):
        ix=idx[i];loss=F.mse_loss(t(xs[ix]),ys[ix]);opt.zero_grad(set_to_none=True)
        loss.backward();nn.utils.clip_grad_norm_(t.parameters(),1.);opt.step()
        if (i+1) in ([4,8] if dry else [350,700]):
            with torch.inference_mode(): v=F.mse_loss(t(xd),yd).item()
            if v<best:best=v;best_state={k:z.detach().clone() for k,z in t.state_dict().items()}
    t.load_state_dict(best_state);t.eval().requires_grad_(False)
    return t,{'kind':kind,'task':task,'seed':seed,'fit_dev_mse':best,'train_wall_s':time.perf_counter()-start,'parameter_count':count(t)}


def reflect_teachers(base,seed):
    g=torch.Generator().manual_seed(32_000_000+seed)
    dirs=torch.randn(R,H,generator=g)
    group=[]
    for r in range(R):
        hh=householder(dirs[r]);teacher=DenseMLP()
        with torch.no_grad():
            teacher.up.weight.copy_(hh@base.up.weight)
            teacher.up.bias.copy_(hh@base.up.bias)
            teacher.down.weight.copy_(base.down.weight@hh)
            teacher.down.bias.copy_(base.down.bias)
        teacher.eval().requires_grad_(False);group.append(teacher)
    return group,dirs


def output(teachers,x,r):
    with torch.inference_mode():
        y=torch.stack([t(x) for t in teachers],1)
        return y[torch.arange(len(x)),r]


def student_data(teachers,seed,world,split,n,scale=1.):
    key= {'train':31_000_000,'dev':41_000_000,'fresh':51_000_000,'ood':61_000_000}[split]
    x,r=draw(key+seed*337+WORLDS.index(world)*10_003,n,scale)
    return x,r,output(teachers,x,r)


def diversity(teachers,seed):
    x,_=draw(81_000_000+seed,512)
    with torch.inference_mode():
        v=[t(x) for t in teachers]
        d=torch.stack([F.mse_loss(v[i],v[j]) for i,j in ((0,1),(0,2),(1,2))])
        norm=torch.stack(v).square().mean().clamp_min(1e-5)
    return (d/norm).tolist()


def packet_bytes(model):
    buffer=io.BytesIO()
    torch.save({'kind':type(model).__name__,'code':getattr(model,'kind','independent'),'state':model.state_dict()},buffer)
    return len(buffer.getvalue())


def train_model(model,train,dev,norm,stream,train_steps,offset=0,dev_steps=(500,750)):
    x,r,y=train;xd,rd,yd=dev
    opt=torch.optim.AdamW(model.parameters(),lr=.012,weight_decay=1e-4)
    best=float('inf');best_state=None;best_step=None;curve={}
    start=time.perf_counter()
    for n in range(train_steps):
        ii=stream[n+offset]
        loss=(model(x[ii],r[ii])-y[ii]).square().mean()/norm
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        step=offset+n+1
        if step in dev_steps:
            with torch.inference_mode():v=((model(xd,rd)-yd).square().mean()/norm).item()
            curve[str(step)]=v
            if v<best:best=v;best_state={k:z.detach().clone() for k,z in model.state_dict().items()};best_step=step
    if best_state is None:raise RuntimeError('No DEV checkpoint')
    model.load_state_dict(best_state)
    return {'dev_nmse':best,'selected_step':best_step,'dev_curve':curve,'train_wall_s':time.perf_counter()-start}


def warm_core(train,dev,norm,stream,seed):
    """One 148-coefficient MLP pretrained exclusively on the fixed train teacher-output labels."""
    x,r,y=train;xd,dr,dy=dev
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        m=DenseMLP()
    opt=torch.optim.AdamW(m.parameters(),lr=.012,weight_decay=1e-4)
    start=time.perf_counter()
    for n in range(WARM_STEPS):
        ii=stream[n]
        loss=(m(x[ii])-y[ii]).square().mean()/norm
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    wall=time.perf_counter()-start
    with torch.inference_mode():v=((m(xd)-dy).square().mean()/norm).item()
    return m,wall,v


def metrics(model,data,norm):
    x,r,y=data
    with torch.inference_mode():
        pred=model(x,r);sq=(pred-y).square().mean(dim=1)
        by=[(sq[r==j].mean().item()/norm) for j in range(R)]
        return {'nmse':sq.mean().item()/norm,'max_rule_nmse':max(by),'rule_nmse':by}


def latency(model,seed):
    model.eval();g=torch.Generator().manual_seed(96_000_000+seed)
    out={}
    with torch.inference_mode():
        for n in (1,128):
            x=torch.randn(n,D,generator=g);r=torch.randint(R,(n,),generator=g)
            for _ in range(5):model(x,r)
            blocks=[]
            for _ in range(7):
                tic=time.perf_counter()
                for _ in range(20):model(x,r)
                blocks.append((time.perf_counter()-tic)*1e6/20)
            out['batch'+str(n)+'_us_median']=statistics.median(blocks)
    return out


def build_teachers(seed,dry=False):
    base,meta=teacher_fit(seed,0,'common',dry=dry)
    corr=[];unrel=[];costs=[meta]
    for task in range(1,R+1):
        t,rec=teacher_fit(seed,task,'correlated',dry=dry);corr.append(t);costs.append(rec)
        t,rec=teacher_fit(seed,task,'unrelated',dry=dry);unrel.append(t);costs.append(rec)
    mirror,dirs=reflect_teachers(base,seed)
    return {'mirror_positive':mirror,'correlated_independent':corr,'unrelated_independent':unrel},dirs,costs,base


def reload_teachers(checkpoint,seed):
    t=torch.load(checkpoint,weights_only=True,map_location='cpu')
    base=DenseMLP();base.load_state_dict(t['base']);base.eval().requires_grad_(False)
    group={'mirror_positive':reflect_teachers(base,seed)[0]}
    for world in ('correlated_independent','unrelated_independent'):
        arr=[]
        for s in t[world]:
            m=DenseMLP();m.load_state_dict(s);m.eval().requires_grad_(False);arr.append(m)
        group[world]=arr
    return group


def run(out,dry=False):
    torch.set_num_threads(1)
    out=Path(out);(out/'checkpoints').mkdir(parents=True,exist_ok=True);(out/'results').mkdir(parents=True,exist_ok=True)
    if not dry and not (out/'PROTOCOL.json').exists():raise RuntimeError('Must supply preregistered PROTOCOL.json')
    if not dry and (out/'results'/'SEALED_TRAIN_MANIFEST.json').exists():raise RuntimeError('No overwriting a sealed experiment')
    tm=time.perf_counter();records=[];teachers_hashes={};teacher_meta={}
    for seed in (SEEDS[:1] if dry else SEEDS):
        groups,dirs,costs,base=build_teachers(seed,dry)
        tf=out/'checkpoints'/f'teachers_seed{seed}.pt'
        torch.save({'base':base.state_dict(),
                    'correlated_independent':[t.state_dict() for t in groups['correlated_independent']],
                    'unrelated_independent':[t.state_dict() for t in groups['unrelated_independent']]},tf)
        teachers_hashes[str(seed)]=sha(tf)
        teacher_meta[str(seed)]={'teacher_training':costs,'teacher_family_diversity':{w:diversity(ts,seed) for w,ts in groups.items()}}
        for world in (WORLDS[:1] if dry else WORLDS):
            tr=student_data(groups[world],seed,world,'train',128 if dry else 1024)
            dv=student_data(groups[world],seed,world,'dev',128 if dry else 384)
            norm=tr[2].square().mean().clamp_min(1e-5).item()
            g=torch.Generator().manual_seed(21_000_000+seed*101+WORLDS.index(world)*3007)
            stream=torch.randint(len(tr[0]),(TOTAL_STEPS,BATCH),generator=g)
            init_seed=12_000_000+seed*7+WORLDS.index(world)*3007
            # Warm core uses same random initialization and first 250 batches as every cold arm.
            warm,pre_wall,pre_dev=warm_core(tr,dv,norm,stream,init_seed)
            for arm in (ARMS[:2] if dry else ARMS):
                kind=arm.split('_')[0]
                model=new_student(kind,init_seed)
                if arm.endswith('_warm'):
                    model.up.load_state_dict(warm.up.state_dict())
                    model.down.load_state_dict(warm.down.state_dict())
                    res=train_model(model,tr,dv,norm,stream,500,offset=250,dev_steps=(500,750))
                    warm_cost=pre_wall
                else:
                    res=train_model(model,tr,dv,norm,stream,750,offset=0,dev_steps=(500,750))
                    warm_cost=0.
                ident=f'{world}_seed{seed}_{arm}'
                cp=out/'checkpoints'/(ident+'.pt')
                torch.save({'state':model.state_dict(),'kind':kind,'seed':seed,'world':world,
                            'norm':norm,'selected_step':res['selected_step']},cp)
                rec={'ident':ident,'seed':seed,'world':world,'arm':arm,'kind':kind,'parameters':count(model),
                     'packet_bytes':packet_bytes(model),'norm':norm,'checkpoint':cp.name,'checkpoint_sha256':sha(cp),
                     'train_wall_s_conditional':res['train_wall_s'],'train_wall_s_pool_charge':warm_cost,
                     'train_wall_s_total':res['train_wall_s']+warm_cost,'dev_nmse':res['dev_nmse'],
                     'selected_step':res['selected_step'],'dev_curve':res['dev_curve'],
                     'total_student_updates':750,'examples_seen':TOTAL_STEPS*BATCH,
                     'core_linear_forward_MAC_per_sample':2*D*H if kind!='independent' else 2*D*7,
                     'pool_dev_nmse':pre_dev if arm.endswith('_warm') else None}
                records.append(rec)
            print('[train]',seed,world,'checkpoints=',len(records),flush=True)
    if dry:
        print('[dry] done. No seal or Fresh data.',flush=True);return
    manifest={'id':'MA-1179','status':'SEALED_BEFORE_FRESH','fresh_generated':False,
              'protocol_sha256':sha(out/'PROTOCOL.json'),
              'source_sha256':sha(Path(__file__)),
              'teacher_hashes':teachers_hashes,'teacher_meta':teacher_meta,'records':records,
              'student_count':len(records),'teacher_count':len(SEEDS)*7,
              'train_phase_wall_s':time.perf_counter()-tm,
              'hardware':{'torch':torch.__version__,'python':platform.python_version(),'cpu_threads':torch.get_num_threads(),
                          'cpu_info':next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown'),
                          'cpu_count':os.cpu_count(),'gpu':torch.cuda.is_available(),'dtype':'fp32'}}
    assert len(records)==120
    dump(out/'results'/'SEALED_TRAIN_MANIFEST.json',manifest)
    print('[seal] 120 student and 5 teacher checkpoint files SHA256 locked; Fresh unopened',flush=True)
    # Unseal evaluation: load ONLY previously sealed checkpoints, never update a model parameter.
    final=[]
    for seed in SEEDS:
        tf=out/'checkpoints'/f'teachers_seed{seed}.pt'
        assert sha(tf)==teachers_hashes[str(seed)]
        groups=reload_teachers(tf,seed)
        for world in WORLDS:
            fresh=student_data(groups[world],seed,world,'fresh',1024)
            ood=student_data(groups[world],seed,world,'ood',1024,scale=1.6)
            for arm in ARMS:
                rec=next(r for r in records if r['seed']==seed and r['world']==world and r['arm']==arm)
                cp=out/'checkpoints'/rec['checkpoint'];assert sha(cp)==rec['checkpoint_sha256']
                ck=torch.load(cp,weights_only=True,map_location='cpu')
                model=new_student(rec['kind'],777);model.load_state_dict(ck['state']);model.eval()
                q=metrics(model,fresh,rec['norm']);oq=metrics(model,ood,rec['norm'])
                lat=latency(model,seed)
                final.append({**rec,'fresh_nmse':q['nmse'],'fresh_worst_rule_nmse':q['max_rule_nmse'],
                              'fresh_rule_nmse':q['rule_nmse'],'ood_nmse':oq['nmse'],
                              'ood_worst_rule_nmse':oq['max_rule_nmse'],'ood_rule_nmse':oq['rule_nmse'],**lat})
        print('[fresh]',seed,'evaluated',len(final),flush=True)
    final.sort(key=lambda r:(WORLDS.index(r['world']),r['seed'],ARMS.index(r['arm'])))
    dump(out/'results'/'FULL_RESULTS.json',{'id':'MA-1179','rows':final,'teacher_meta':teacher_meta})
    flat=[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in final]
    fields=list(flat[0]);fields=[k for k in fields if k in set.intersection(*(set(row) for row in flat))]
    with (out/'results'/'RESULTS_CORE.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(flat)
    gates={};paired={}
    for world in WORLDS:
        vals=[]
        for seed in SEEDS:
            v={r['arm']:r for r in final if r['world']==world and r['seed']==seed}
            vals.append({'seed':seed,'mirror_warm':v['mirror_warm']['fresh_nmse'],
                         'mirror_cold':v['mirror_cold']['fresh_nmse'],
                         'film_warm':v['film_warm']['fresh_nmse'],
                         'rank1_warm':v['rank1_warm']['fresh_nmse'],
                         'independent_cold':v['independent_cold']['fresh_nmse'],
                         'wins_warm':v['mirror_warm']['fresh_nmse']<=0.9*v['mirror_cold']['fresh_nmse'],
                         'wins_both':v['mirror_warm']['fresh_nmse']<=0.9*min(v['film_warm']['fresh_nmse'],v['rank1_warm']['fresh_nmse'])})
        paired[world]=vals
    co=paired['correlated_independent'];gates['warm_optimization']={'wins':sum(r['wins_warm'] for r in co),'needed':4}
    gates['mirror_specific']={'wins':sum(r['wins_both'] for r in co),'needed':4}
    for entry in gates.values():entry['verdict']='PASS' if entry['wins']>=entry['needed'] else 'FAIL'
    medians={world:{arm:{field:statistics.median(r[field] for r in final if r['world']==world and r['arm']==arm)
                          for field in ('fresh_nmse','ood_nmse','packet_bytes','train_wall_s_total','batch1_us_median','batch128_us_median')}
                    for arm in ARMS} for world in WORLDS}
    verdict={'id':'MA-1179','gates':gates,'paired':paired,'medians':medians,
             'full_results_sha256':sha(out/'results'/'FULL_RESULTS.json'),
             'results_core_sha256':sha(out/'results'/'RESULTS_CORE.csv')}
    dump(out/'results'/'ANALYSIS.json',verdict)
    print('[decision]',json.dumps(gates),flush=True)
    # All 120 checkpoints reloaded to independent evaluation model and hashed again.
    replay_errors=[];hash_errors=[]
    for r in final:
        cp=out/'checkpoints'/r['checkpoint']
        if sha(cp)!=r['checkpoint_sha256']:hash_errors.append(r['ident']);continue
        ck=torch.load(cp,map_location='cpu',weights_only=True)
        model=new_student(r['kind'],999);model.load_state_dict(ck['state']);model.eval()
        groups=reload_teachers(out/'checkpoints'/f'teachers_seed{r["seed"]}.pt',r['seed'])
        fresh=student_data(groups[r['world']],r['seed'],r['world'],'fresh',1024)
        result=metrics(model,fresh,r['norm'])['nmse']
        replay_errors.append(abs(result-r['fresh_nmse']))
    report={'id':'MA-1179','verified_student_checkpoints':len(replay_errors),'teacher_checkpoint_hashes_match':all(sha(out/'checkpoints'/f'teachers_seed{s}.pt')==teachers_hashes[str(s)] for s in SEEDS),
            'student_checkpoint_hash_failures':hash_errors,'max_fresh_nmse_replay_abs_diff':max(replay_errors),
            'all_replays_equal':len(replay_errors)==120 and max(replay_errors)<1e-12 and not hash_errors,
            'seal_was_written_before_any_fresh_inputs':True,
            'protocol_sha256':manifest['protocol_sha256'],
            'source_sha256':manifest['source_sha256'],
            'csv_sha256':sha(out/'results'/'RESULTS_CORE.csv'),
            'torch':torch.__version__,'cpu_threads':torch.get_num_threads(),
            'limits':'synthetic 4D / 5 seeds / 35 teachers / one CPU thread; equal updates not equal FLOPs; no held-out natural LM task'}
    dump(out/'results'/'VERIFICATION.json',report)
    print('[verification]',len(replay_errors),'checkpoint replays; max error',max(replay_errors),'total wall',round(time.perf_counter()-tm,1),'s',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    run(args.out,args.dry_run)

if __name__=='__main__':main()