"""MA-1180: optimizer-geometry ablation for Householder Mirror codes.

Preregistered plan: ../PROTOCOL.json (commit before any student training).
Inputs: pre-sealed MA-1179 teacher bundles, SHA-verified, never trained student teacher weights.
Outputs: SHA-sealed 120 students -> unseen fresh/OOD -> verify and analyze.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,io,json,math,os,platform,statistics,time
from pathlib import Path
import torch
from torch.nn import functional as F

SEEDS=(101,211,307,401,503)
WORLDS=('correlated_independent','mirror_positive','unrelated_independent')
ARMS=('mirror_adamw','mirror_adamw_nodecay','mirror_adamw_low','mirror_adamw_high',
      'mirror_adamw_renorm','mirror_sphere_scalar_adam','mirror_sphere_sgd_momentum','film_adamw')
EUCLIDEAN=ARMS[:4]
GEOMETRIC=ARMS[4:7]
BATCH=128;STEPS=750;DEV_CHECKS=(250,500,750)


def load_parent(custom=None):
    candidates=[Path(custom)] if custom else [Path(__file__).resolve().parents[2]/'ma-1179-correlated-teacher-core-warmstart'/'source'/'run.py',Path('/mnt/data/MA1179_work/source/run.py')]
    path=next((p for p in candidates if p.is_file()),None)
    if path is None:raise FileNotFoundError(f'Need MA-1179 source, tried: {candidates}')
    spec=importlib.util.spec_from_file_location('ma1179_frozen',str(path));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod,path


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,sort_keys=True)+'\n',encoding='utf-8')


def tangent(v,g):
    return g-v*((v*g).sum(-1,keepdim=True)/v.square().sum(-1,keepdim=True).clamp_min(1e-12))


def renorm(v,initial_radii):
    with torch.no_grad():v.mul_(initial_radii/v.norm(dim=-1,keepdim=True).clamp_min(1e-12))


class SphereScalarAdam:
    """Experimental scalar-moment tangent adaptive step with sphere retraction.

    Not claimed equivalent to published Riemannian Adam. Row-wise mean-square gradient gives
    a coordinate-rotation insensitive scalar preconditioner, unlike coordinate-wise Adam.
    """
    def __init__(self,v,lr=.012,b1=.9,b2=.999,eps=1e-8):
        self.v=v;self.lr=lr;self.b1=b1;self.b2=b2;self.eps=eps
        self.r=v.detach().norm(dim=-1,keepdim=True).clone()
        self.m=torch.zeros_like(v);self.s=torch.zeros(v.shape[0],1,dtype=v.dtype,device=v.device)
        self.t=0
    @torch.no_grad()
    def step(self):
        if self.v.grad is None:raise RuntimeError('missing code gradients')
        g=tangent(self.v,self.v.grad)
        self.t+=1;self.m.mul_(self.b1).add_(g,alpha=1-self.b1)
        self.m.copy_(tangent(self.v,self.m))
        self.s.mul_(self.b2).add_(g.square().mean(dim=-1,keepdim=True),alpha=1-self.b2)
        m_hat=self.m/(1-self.b1**self.t)
        s_hat=self.s/(1-self.b2**self.t)
        direction=tangent(self.v,m_hat/(s_hat.sqrt()+self.eps))
        self.v.add_(direction,alpha=-self.lr)
        renorm(self.v,self.r)
        self.m.copy_(tangent(self.v,self.m))


class SphereMomentumSGD:
    def __init__(self,v,lr=.03,momentum=.9):
        self.v=v;self.lr=lr;self.momentum=momentum
        self.r=v.detach().norm(dim=-1,keepdim=True).clone();self.velocity=torch.zeros_like(v)
    @torch.no_grad()
    def step(self):
        if self.v.grad is None:raise RuntimeError('missing code gradients')
        g=tangent(self.v,self.v.grad)
        self.velocity.mul_(self.momentum).add_(g)
        self.velocity.copy_(tangent(self.v,self.velocity))
        self.v.add_(self.velocity,alpha=-self.lr)
        renorm(self.v,self.r)
        self.velocity.copy_(tangent(self.v,self.velocity))


def setup_optim(model,arm):
    if arm=='film_adamw':
        return torch.optim.AdamW(model.parameters(),lr=.012,weight_decay=1e-4),None,None
    core=torch.optim.AdamW([*model.up.parameters(),*model.down.parameters()],lr=.012,weight_decay=1e-4)
    radii=model.codes.detach().norm(dim=-1,keepdim=True).clone()
    if arm=='mirror_sphere_scalar_adam':code=SphereScalarAdam(model.codes)
    elif arm=='mirror_sphere_sgd_momentum':code=SphereMomentumSGD(model.codes)
    else:
        lr={'mirror_adamw_low':.004,'mirror_adamw_high':.036}.get(arm,.012)
        wd=0. if arm=='mirror_adamw_nodecay' else 1e-4
        code=torch.optim.AdamW([model.codes],lr=lr,weight_decay=wd)
    return core,code,radii


def new_data(parent,teachers,world,seed,split,n,scale=1.):
    key={'train':91_000_000,'dev':101_000_000,'fresh':111_000_000,'ood':121_000_000}[split]
    x,r=parent.draw(key+seed*337+WORLDS.index(world)*10003,n,scale)
    return x,r,parent.output(teachers,x,r)


def score(model,data,norm):
    x,r,y=data
    with torch.inference_mode():
        sq=(model(x,r)-y).square().mean(-1)
        by=[sq[r==i].mean().item()/norm for i in range(3)]
    return {'nmse':sq.mean().item()/norm,'per_rule':by,'worst_rule_nmse':max(by)}


def latency(model,seed):
    generator=torch.Generator().manual_seed(151_000_000+seed)
    results={}
    model.eval()
    with torch.inference_mode():
        for n in (1,128):
            x=torch.randn(n,4,generator=generator);r=torch.randint(3,(n,),generator=generator)
            for _ in range(5):model(x,r)
            blocks=[]
            for _ in range(5):
                start=time.perf_counter()
                for _ in range(20):model(x,r)
                blocks.append((time.perf_counter()-start)*1e6/20)
            results[f'batch{n}_us_median']=statistics.median(blocks)
    return results


def train(parent,model,arm,train_data,dev,norm,stream):
    core,code,initial=setup_optim(model,arm)
    best=float('inf');best_state=None;best_step=0;curve={}
    grad_rads=[];step_rads=[];norm_drifts=[]
    start=time.perf_counter()
    for i in range(STEPS):
        xx,rr,yy=train_data
        ix=stream[i]
        output=model(xx[ix],rr[ix]);loss=(output-yy[ix]).square().mean()/norm
        core.zero_grad(set_to_none=True)
        if code is not None and hasattr(code,'zero_grad'):code.zero_grad(set_to_none=True)
        elif code is not None and model.codes.grad is not None:model.codes.grad=None
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if i%50==0 and code is not None:
            with torch.no_grad():
                v=model.codes;g=v.grad
                grad_rads.append(float(((v*g).sum(-1).abs()/(v.norm(dim=-1)*g.norm(dim=-1)+1e-12)).mean()))
                old_v=v.detach().clone()
        core.step()
        if code is not None:code.step()
        if arm=='mirror_adamw_renorm':renorm(model.codes,initial)
        if i%50==0 and code is not None:
            with torch.no_grad():
                dv=model.codes-old_v
                step_rads.append(float((dv*old_v).sum(-1).abs().div(dv.norm(dim=-1)*old_v.norm(dim=-1)+1e-12).mean()))
                norm_drifts.append(float((model.codes.norm(dim=-1,keepdim=True)-initial).abs().div(initial).max()))
        step=i+1
        if step in DEV_CHECKS:
            value=score(model,dev,norm)['nmse'];curve[str(step)]=value
            if value<best:
                best=value;best_step=step
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    wall=time.perf_counter()-start
    model.load_state_dict(best_state)
    return {'dev_nmse':best,'selected_step':best_step,'dev_curve':curve,'train_wall_s':wall,
            'code_grad_radial_cosine_mean':statistics.mean(grad_rads) if grad_rads else None,
            'code_step_radial_cosine_mean':statistics.mean(step_rads) if step_rads else None,
            'code_max_rel_norm_drift_mean':statistics.mean(norm_drifts) if norm_drifts else None}


def audit_mechanism(parent):
    g=torch.Generator().manual_seed(1180);v=torch.randn(3,16,dtype=torch.float64,generator=g).requires_grad_(True)
    x=torch.randn(10,16,dtype=torch.float64,generator=g)
    h=parent.reflection(x,v[torch.arange(10)%3]);hn=parent.reflection(x,(-3.7*v)[torch.arange(10)%3])
    inv=(h-hn).abs().max().item()
    # Use a non-isometric target so the radial-gradient diagnostic has nonzero norm.
    target=torch.randn(h.shape,dtype=h.dtype,generator=g)
    grad=torch.autograd.grad((h-target).square().mean(),v)[0]
    cos=(grad*v).sum(-1).abs()/(grad.norm(dim=-1)*v.norm(dim=-1)+1e-30)
    return {'scale_sign_invariance_max_abs_error_fp64':inv,'radial_gradient_cosine_max_fp64':cos.max().item()}


def run(out,teacher_root,parent_source=None,dry=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(out);teacher_root=Path(teacher_root)
    (out/'checkpoints').mkdir(parents=True,exist_ok=True);(out/'results').mkdir(parents=True,exist_ok=True)
    if not (out/'PROTOCOL.json').exists():raise RuntimeError('Must stage preregistered PROTOCOL.json')
    if (out/'results'/'SEALED_TRAIN_MANIFEST.json').exists():raise RuntimeError('Refuse to overwrite sealed run')
    protocol=json.loads((out/'PROTOCOL.json').read_text())
    if protocol['id']!='MA-1180':raise RuntimeError('Bad protocol')
    parent,parent_path=load_parent(parent_source)
    assert len(ARMS)==8 and len(WORLDS)==3
    mechanism=audit_mechanism(parent)
    assert mechanism['scale_sign_invariance_max_abs_error_fp64']<1e-10
    assert mechanism['radial_gradient_cosine_max_fp64']<1e-8
    start=time.perf_counter();records=[];parent_teacher_hashes={}
    for seed in (SEEDS[:1] if dry else SEEDS):
        teacher_file=teacher_root/f'teachers_seed{seed}.pt'
        expected=protocol['control_teacher_provenance']['teacher_sha256'][str(seed)]
        actual=sha(teacher_file)
        if expected!=actual:raise RuntimeError(f'teacher SHA mismatch for seed={seed}: {actual} != {expected}')
        parent_teacher_hashes[str(seed)]=actual
        groups=parent.reload_teachers(teacher_file,seed)
        for world in (WORLDS[:1] if dry else WORLDS):
            train_set=new_data(parent,groups[world],world,seed,'train',256 if dry else 1024)
            dev=new_data(parent,groups[world],world,seed,'dev',256 if dry else 384)
            norm=train_set[2].square().mean().clamp_min(1e-5).item()
            gen=torch.Generator().manual_seed(131_000_000+seed*19+WORLDS.index(world)*733)
            stream=torch.randint(len(train_set[0]),(STEPS,128),generator=gen)
            init_seed=141_000_000+seed*27+WORLDS.index(world)*1117
            with torch.inference_mode():
                probe_x=train_set[0][:10];probe_r=train_set[1][:10]
                ref_pred=parent.new_student('mirror',init_seed)(probe_x,probe_r)
            for arm in (ARMS[:3] if dry else ARMS):
                typ='film' if arm=='film_adamw' else 'mirror'
                model=parent.new_student(typ,init_seed)
                if typ=='mirror':
                    with torch.inference_mode():
                        assert (model(probe_x,probe_r)-ref_pred).abs().max().item()==0.
                rec=train(parent,model,arm,train_set,dev,norm,stream)
                ident=f'{world}_seed{seed}_{arm}'
                cp=out/'checkpoints'/f'{ident}.pt'
                torch.save({'state':model.state_dict(),'seed':seed,'world':world,'arm':arm,'normalizer':norm,'selected_step':rec['selected_step']},cp)
                records.append({'ident':ident,'world':world,'seed':seed,'arm':arm,'kind':typ,'param_count':parent.count(model),
                                'packet_bytes':parent.packet_bytes(model),'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'norm':norm,
                                'examples_seen':STEPS*BATCH,'optimizer_updates':STEPS,**rec})
            print('[train]',seed,world,'models',len(records),flush=True)
    if dry:
        print('[dry] complete; no Fresh or manifest written',flush=True);return
    assert len(records)==120
    metadata={'id':'MA-1180','status':'SEALED_BEFORE_FRESH','fresh_generated':False,'student_count':len(records),
              'preregistered_protocol_sha256':sha(out/'PROTOCOL.json'),'source_sha256':sha(Path(__file__)),
              'parent_source_sha256':sha(parent_path),'teacher_hashes':parent_teacher_hashes,
              'mechanism':mechanism,'training_wall_s':time.perf_counter()-start,'records':records,
              'environment':{'python':platform.python_version(),'torch':torch.__version__,'cpu_threads':torch.get_num_threads(),
                             'dtype':'fp32','cpu':next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'unknown'),'cpu_count':os.cpu_count()}}
    write(out/'results'/'SEALED_TRAIN_MANIFEST.json',metadata)
    print('[seal] 120 checkpoints and five teacher hashes frozen; generating Fresh only now',flush=True)
    final=[]
    for seed in SEEDS:
        teacher_file=teacher_root/f'teachers_seed{seed}.pt'
        assert sha(teacher_file)==parent_teacher_hashes[str(seed)]
        teachers=parent.reload_teachers(teacher_file,seed)
        for world in WORLDS:
            fresh=new_data(parent,teachers[world],world,seed,'fresh',1024)
            ood=new_data(parent,teachers[world],world,seed,'ood',1024,scale=1.6)
            for arm in ARMS:
                rec=next(x for x in records if x['seed']==seed and x['world']==world and x['arm']==arm)
                cp=out/'checkpoints'/rec['checkpoint']
                assert sha(cp)==rec['checkpoint_sha256']
                state=torch.load(cp,map_location='cpu',weights_only=True)
                m=parent.new_student(rec['kind'],909);m.load_state_dict(state['state']);m.eval()
                qual=score(m,fresh,rec['norm']);outq=score(m,ood,rec['norm'])
                runtime=latency(m,seed)
                final.append({**rec,'fresh_nmse':qual['nmse'],'fresh_rule_nmse':qual['per_rule'],'fresh_worst_rule_nmse':qual['worst_rule_nmse'],
                              'ood_nmse':outq['nmse'],'ood_rule_nmse':outq['per_rule'],'ood_worst_rule_nmse':outq['worst_rule_nmse'],**runtime})
        print('[fresh]',seed,len(final),flush=True)
    final.sort(key=lambda x:(WORLDS.index(x['world']),x['seed'],ARMS.index(x['arm'])))
    write(out/'results'/'FULL_RESULTS.json',{'id':'MA-1180','rows':final,'mechanism':mechanism})
    flat=[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in final]
    with (out/'results'/'RESULTS_CORE.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(flat[0]));wr.writeheader();wr.writerows(flat)
    paired={};gates={}
    for world in WORLDS:
        sub=[]
        for seed in SEEDS:
            rr={v['arm']:v for v in final if v['seed']==seed and v['world']==world}
            e=min(EUCLIDEAN,key=lambda a:rr[a]['dev_nmse'])
            geom=min(GEOMETRIC,key=lambda a:rr[a]['dev_nmse'])
            sub.append({'seed':seed,'baseline_adamw_nmse':rr['mirror_adamw']['fresh_nmse'],
                        'sphere_scalar_adam_nmse':rr['mirror_sphere_scalar_adam']['fresh_nmse'],
                        'euclid_dev_best_arm':e,'geom_dev_best_arm':geom,
                        'euclid_selected_fresh_nmse':rr[e]['fresh_nmse'],
                        'geom_selected_fresh_nmse':rr[geom]['fresh_nmse'],
                        'film_fresh_nmse':rr['film_adamw']['fresh_nmse'],
                        'geometry_fixed_win':rr['mirror_sphere_scalar_adam']['fresh_nmse']<=.9*rr['mirror_adamw']['fresh_nmse'],
                        'geometry_tuned_win':rr[geom]['fresh_nmse']<=.9*rr[e]['fresh_nmse'],
                        'mirror_vs_film_win':rr[geom]['fresh_nmse']<=.9*rr['film_adamw']['fresh_nmse']})
        paired[world]=sub
    correlated=paired['correlated_independent']
    for name,field in [('geometry_fixed','geometry_fixed_win'),('geometry_vs_euclidean_tuned','geometry_tuned_win'),('mirror_specific_quality','mirror_vs_film_win')]:
        n=sum(int(q[field]) for q in correlated)
        gates[name]={'wins':n,'needed':4,'verdict':'PASS' if n>=4 else 'FAIL'}
    medians={world:{arm:{name:statistics.median(r[name] for r in final if r['arm']==arm and r['world']==world)
                         for name in ('fresh_nmse','ood_nmse','train_wall_s','packet_bytes','batch1_us_median','batch128_us_median','code_step_radial_cosine_mean') if all(r[name] is not None for r in final if r['arm']==arm and r['world']==world)}
                 for arm in ARMS} for world in WORLDS}
    write(out/'results'/'ANALYSIS.json',{'id':'MA-1180','paired':paired,'gates':gates,'medians':medians,
                                       'optimizer_search_note':'3 geometry and 4 Euclidean candidates chosen by DEV per paired seed; cost includes training all arms',
                                       'full_results_sha256':sha(out/'results'/'FULL_RESULTS.json'),'results_core_sha256':sha(out/'results'/'RESULTS_CORE.csv')})
    replay=[];bad=[]
    for r in final:
        cp=out/'checkpoints'/r['checkpoint']
        if sha(cp)!=r['checkpoint_sha256']:bad.append(r['ident']);continue
        m=parent.new_student(r['kind'],777)
        m.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['state']);m.eval()
        tf=parent.reload_teachers(teacher_root/f'teachers_seed{r["seed"]}.pt',r['seed'])
        fresh=new_data(parent,tf[r['world']],r['world'],r['seed'],'fresh',1024)
        replay.append(abs(score(m,fresh,r['norm'])['nmse']-r['fresh_nmse']))
    verification={'id':'MA-1180','verified_checkpoints':len(replay),'expected_checkpoints':120,
                  'max_fresh_nmse_replay_abs_diff':max(replay),'bad_student_hashes':bad,
                  'all_replays_equal':len(replay)==120 and max(replay)<1e-12 and not bad,
                  'all_teacher_hashes_match':all(sha(teacher_root/f'teachers_seed{s}.pt')==parent_teacher_hashes[str(s)] for s in SEEDS),
                  'mechanism':mechanism,'seal_before_fresh':True,
                  'protocol_sha256':sha(out/'PROTOCOL.json'),'source_sha256':sha(Path(__file__)),
                  'torch':torch.__version__,'cpu_threads':torch.get_num_threads()}
    write(out/'results'/'VERIFICATION.json',verification)
    print('[done]',gates,'verified',len(replay),'in',round(time.perf_counter()-start,1),'s',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--teacher-root',type=Path,required=True)
    ap.add_argument('--parent-source',type=Path,default=None)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args();run(args.out,args.teacher_root,args.parent_source,args.dry_run)
if __name__=='__main__':main()
