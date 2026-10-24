"""MA-1178 preregistered teacher-MLP -> compressed Mirror functional distillation study.

Usage: python source/run.py --out /mnt/data/MA1178_mlp_to_mirror_distillation
Train/dev/teacher-fitting independent of fresh/OOD; seal all checkpoints first.
"""
from __future__ import annotations
import argparse,csv,copy,hashlib,json,os,platform,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import (DenseMLP,SharedStudent,new_student,param_count,packet_bytes,teacher_mirror_family,teacher_film_family,teacher_outputs,TEACHER_WIDTH,WIDTHS,INDEPENDENT_WIDTHS)
from data import draw_split,analytic_target,student_sets,fresh_sets

SEEDS=[101,211,307,401,503]
WORLDS=['mirror_related','film_related','independent']
ARMS=['mirror','film','shared','independent']
STEPS=750
CHECKPOINTS=[250,500,750]
LR=0.012
TEACHER_STEPS=1100
TEACHER_LR=0.012
N_TRAIN=1024
N_DEV=384
BATCH=128


def checksum(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<19),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path:Path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def teacher_training(seed:int,teacher_id:int)->tuple[DenseMLP,dict]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed*3125+teacher_id*373+74)
        t=DenseMLP(TEACHER_WIDTH)
    x,_=draw_split(17_000_000+seed*100+teacher_id*1000,3072)
    y=analytic_target(x,teacher_id,seed)
    dx,_=draw_split(18_000_000+seed*100+teacher_id*1000,768)
    dy=analytic_target(dx,teacher_id,seed)
    g=torch.Generator().manual_seed(19_000_000+seed*100+teacher_id*1000)
    indices=torch.randint(x.shape[0],(TEACHER_STEPS,BATCH),generator=g)
    opt=torch.optim.AdamW(t.parameters(),lr=TEACHER_LR,weight_decay=1e-4)
    best=float('inf');best_state=None
    for step in range(1,TEACHER_STEPS+1):
        ind=indices[step-1];loss=F.mse_loss(t(x[ind]),y[ind]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(t.parameters(),1.0);opt.step()
        if step in [550,825,1100]:
            with torch.no_grad():score=F.mse_loss(t(dx),dy).item()
            if score<best:best=score;best_state={k:v.detach().clone() for k,v in t.state_dict().items()}
    t.load_state_dict(best_state);t.eval();t.requires_grad_(False)
    with torch.no_grad():
        yy=t(dx);nmse=(yy-dy).square().mean()/dy.square().mean().clamp_min(1e-5)
    return t,{'teacher_id':teacher_id,'teacher_fit_dev_mse':best,'teacher_fit_dev_nmse':nmse.item(),'teacher_params':param_count(t)}


def build_worlds(seed:int,base:DenseMLP,indep:list[DenseMLP]):
    g=torch.Generator().manual_seed(50_000+seed)
    dirs=torch.randn(3,TEACHER_WIDTH,generator=g)
    film_codes=torch.randn(3,TEACHER_WIDTH,generator=g)*1.4
    teachers={
      'mirror_related':teacher_mirror_family(base,dirs),
      'film_related':teacher_film_family(base,film_codes),
      'independent':indep,
    }
    return teachers,dirs,film_codes


def teacher_diversity(teachers:list[DenseMLP],seed:int):
    x,_=draw_split(99900+seed,500)
    with torch.no_grad():
        yy=torch.stack([t(x) for t in teachers],dim=0)
        denom=yy.square().mean().clamp_min(1e-5)
        pair=[(yy[a]-yy[b]).square().mean()/denom for a,b in [(0,1),(0,2),(1,2)]]
    return dict(pair_nmse=[x.item() for x in pair],pair_mean_nmse=torch.stack(pair).mean().item())


def train_student(model,train,dev,norm,seed:int,world:str,arm:str,width:int):
    x,r,y=train;dx,dr,dy=dev
    g=torch.Generator().manual_seed(2_000_000+seed*100+WORLDS.index(world)*1000)
    indices=torch.randint(x.shape[0],(STEPS,BATCH),generator=g)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    best=float('inf');best_at=None;best_state=None;dev_curve={};start=time.perf_counter()
    for step in range(1,STEPS+1):
        idx=indices[step-1]
        yhat=model(x[idx],r[idx]);loss=(yhat-y[idx]).square().mean()/norm
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if step in CHECKPOINTS:
            with torch.inference_mode():val=((model(dx,dr)-dy).square().mean()/norm).item()
            dev_curve[str(step)]=val
            if val<best:
                best=val;best_at=step;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    wall=time.perf_counter()-start
    model.load_state_dict(best_state)
    return {'selected_dev_nmse':best,'selected_checkpoint_step':best_at,'dev_curve':dev_curve,'fit_wall_s':wall}


def evaluate_model(model,example,norm:float):
    x,r,y=example
    with torch.inference_mode():
        yhat=model(x,r)
        sq=(yhat-y).square().mean(dim=-1)
        means=[sq[r==i].mean().item()/norm for i in range(3)]
        max_abs=(yhat-y).abs().max().item()
    return {'nmse':sq.mean().item()/norm,'worst_rule_nmse':max(means),'rule_nmse':means,'max_abs_error':max_abs}


def latency_us(model,width,kind,seed):
    g=torch.Generator().manual_seed(909_000+seed)
    result={}
    model.eval()
    with torch.inference_mode():
        for n in [1,128]:
            x=torch.randn(n,4,generator=g);r=torch.randint(3,(n,),generator=g)
            for _ in range(5):model(x,r)
            trials=25
            tic=time.perf_counter()
            for _ in range(trials):model(x,r)
            result[f'latency_{n}_us']=(time.perf_counter()-tic)*1e6/trials
    return result


def main():
    torch.set_num_threads(1)
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--dry-run',action='store_true');args=ap.parse_args()
    out=args.out;out.mkdir(exist_ok=True,parents=True)
    (out/'checkpoints').mkdir(exist_ok=True)
    (out/'results').mkdir(exist_ok=True)
    records=[];world_metadata={};teacher_hashes={};start=time.perf_counter()
    for seed in (SEEDS[:1] if args.dry_run else SEEDS):
        base,meta=teacher_training(seed,0)
        independent=[];teacher_fit=[meta]
        for i in range(3):
            t,tm=teacher_training(seed,i+1);independent.append(t);teacher_fit.append(tm)
        groups,dirs,fcodes=build_worlds(seed,base,independent)
        teacher_data={'base':base.state_dict(),'independent':[x.state_dict() for x in independent],'mirror_dirs':dirs,'film_codes':fcodes}
        tf=out/'checkpoints'/f'teacher_{seed}.pt';torch.save(teacher_data,tf);teacher_hashes[str(seed)]=checksum(tf)
        world_metadata[str(seed)]={'teacher_fit':teacher_fit,'diversity':{w:teacher_diversity(gr,seed) for w,gr in groups.items()}}
        # Algebraic oracle: the exact pretrained hidden weights and original Mirror code make a 196-float student equal a 444-float 3-teacher family.
        oracle=SharedStudent(16,'mirror')
        oracle.up.load_state_dict(base.up.state_dict());oracle.down.load_state_dict(base.down.state_dict())
        with torch.no_grad():oracle.codes.copy_(dirs)
        for world in (WORLDS[:1] if args.dry_run else WORLDS):
            teachers=groups[world]
            tr,dv,norm=student_sets(world,seed,teachers,n_train=128 if args.dry_run else N_TRAIN,n_dev=N_DEV)
            if world=='mirror_related':
                world_metadata[str(seed)]['mirror_oracle_dev_nmse']=evaluate_model(oracle,dv,norm)['nmse']
            for width in (WIDTHS[:1] if args.dry_run else WIDTHS):
                for arm in (ARMS[:2] if args.dry_run else ARMS):
                    init_seed=720_000+seed*13+WORLDS.index(world)*733+width*179
                    model=new_student(arm,width,init_seed)
                    res=train_student(model,tr,dv,norm,seed,world,arm,width)
                    ident=f'{world}_seed{seed}_H{width}_{arm}'
                    f=out/'checkpoints'/f'{ident}.pt'
                    obj={'model_state':model.state_dict(),'model_kind':arm,'world':world,'seed':seed,'width':width,'normalizer_train':norm,'dev_nmse':res['selected_dev_nmse'],'teacher_hash':teacher_hashes[str(seed)],'step':res['selected_checkpoint_step']}
                    torch.save(obj,f)
                    row={'ident':ident,'world':world,'seed':seed,'width':width,'arm':arm,'student_params':param_count(model),'teacher_sum_params':444,'checkpoint_file':f.name,'checkpoint_sha256':checksum(f),'packet_bytes':packet_bytes(model),'normalizer':norm,**res}
                    records.append(row)
                print(f'[train] seed={seed} world={world} h={width} done {len(records)}',flush=True)
    if args.dry_run:
        print('[dry-run] complete',flush=True)
        return
    # No fresh or OOD generation before all teacher + student checkpoints have been saved and hashed.
    train_manifest={'id':'MA-1178','status':'SEALED_BEFORE_FRESH','fresh_generated':False,'teacher_hashes':teacher_hashes,'records':records,'world_metadata':world_metadata,'training_wall_s':time.perf_counter()-start,'hardware':{'python':sys.version,'torch':torch.__version__,'cpu_threads':torch.get_num_threads(),'cpu':platform.processor()},'protocol_file_sha256':checksum(out/'PROTOCOL.json')}
    write_json(out/'results'/'SEALED_TRAIN_MANIFEST.json',train_manifest)
    print('[seal] all',len(records),'student checkpoint hashes locked',flush=True)
    # Now open Fresh/OOD once and replay only previously selected dev checkpoints.
    all_results=[]
    for seed in SEEDS:
        td=torch.load(out/'checkpoints'/f'teacher_{seed}.pt',weights_only=True,map_location='cpu')
        base=DenseMLP(16);base.load_state_dict(td['base']);base.eval().requires_grad_(False)
        indep=[]
        for sd in td['independent']:
            t=DenseMLP(16);t.load_state_dict(sd);t.eval().requires_grad_(False);indep.append(t)
        groups={'mirror_related':teacher_mirror_family(base,td['mirror_dirs']),'film_related':teacher_film_family(base,td['film_codes']),'independent':indep}
        for world in WORLDS:
            fresh,ood=fresh_sets(world,seed,groups[world])
            for width in WIDTHS:
                for arm in ARMS:
                    ident=f'{world}_seed{seed}_H{width}_{arm}'
                    rec=next(x for x in records if x['ident']==ident)
                    path=out/'checkpoints'/rec['checkpoint_file']
                    if checksum(path)!=rec['checkpoint_sha256']:raise RuntimeError('checkpoint hash mismatch '+ident)
                    ck=torch.load(path,weights_only=True,map_location='cpu')
                    model=new_student(arm,width,72_000)
                    model.load_state_dict(ck['model_state']);model.eval()
                    id_metrics=evaluate_model(model,fresh,rec['normalizer'])
                    ood_metrics=evaluate_model(model,ood,rec['normalizer'])
                    runtimes=latency_us(model,width,arm,seed)
                    all_results.append({**rec,**{f'fresh_{k}':v for k,v in id_metrics.items()},**{f'ood_{k}':v for k,v in ood_metrics.items()},**runtimes})
        print(f'[fresh] seed={seed} all 3 worlds scored',flush=True)
    all_results.sort(key=lambda r:(WORLDS.index(r['world']),r['seed'],r['width'],ARMS.index(r['arm'])))
    write_json(out/'results'/'FULL_RESULTS.json',{'id':'MA-1178','rows':all_results,'world_metadata':world_metadata})
    scalar_fields=[p for p,v in all_results[0].items() if not isinstance(v,(list,dict))]
    with (out/'results'/'RESULTS_CORE.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=scalar_fields,extrasaction='ignore');wr.writeheader();wr.writerows(all_results)
    write_json(out/'results'/'EVAL_COMPLETE.json',{'id':'MA-1178','model_count':len(all_results),'fresh_scored_after_seal':True,'checkpoint_count':len(records),'total_elapsed_s':time.perf_counter()-start,'hash_of_results':checksum(out/'results'/'RESULTS_CORE.csv')})
    print('[done] scored',len(all_results),'total elapsed',round(time.perf_counter()-start,2), 's',flush=True)

if __name__=='__main__':main()
