"""MA-1182 preregistered experimental runner. Train all models, seal, then score fresh/OOD once."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,os,time,platform,statistics,shutil,traceback
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from core import Model,make_data,P,PROTOCOL,POSITIONS,ARMS,TASKS,bytes_inference,count_params

BASE=Path(__file__).resolve().parents[1]; CONFIG=PROTOCOL['design']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jwrite(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n')

def score(model,x,y):
    model.eval()
    with torch.no_grad():
        z=model(x);return {'ce':F.cross_entropy(z,y).item(),'acc':(z.argmax(-1)==y).float().mean().item()}

def train_one(out,seed,task,pos,arm):
    # Don't use the no-heldout Fresh/OOD prior to seal. Each seed/task has paired data and minibatches.
    model=Model(arm,pos,seed)
    x,y=make_data(task,seed,CONFIG['n_train'],1)
    dx,dy=make_data(task,seed,CONFIG['n_dev'],2)
    init=model.view_difference(x[:96])
    bs=CONFIG['batch_size']
    rng=torch.Generator().manual_seed(884120+seed*701+TASKS.index(task)*2173)
    indexes=torch.randint(x.shape[0],(CONFIG['steps'],bs),generator=rng)
    optim=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    tic=time.perf_counter();best_ce=float('inf');best_state=None;best_step=None;curve={}
    model.train()
    for step in range(1,CONFIG['steps']+1):
        ix=indexes[step-1]
        loss=F.cross_entropy(model(x[ix]),y[ix]);optim.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['grad_clip']);optim.step()
        if step in CONFIG['checkpoints']:
            d=score(model,dx,dy);curve[str(step)]=d
            if d['ce']<best_ce:
                best_ce=d['ce'];best_state={k:v.detach().clone() for k,v in model.state_dict().items()};best_step=step
            model.train()
    training_s=time.perf_counter()-tic
    assert best_state is not None and best_step is not None
    model.load_state_dict(best_state);model.eval()
    final_diag=model.view_difference(dx[:96])
    key=f'{task}_{pos}_{arm}_seed{seed}'
    cp=out/'checkpoints'/f'{key}.pt'
    torch.save({'state':best_state,'seed':seed,'arm':arm,'pos':pos,'task':task,'step':best_step},cp)
    return {'key':key,'task':task,'seed':seed,'pos':pos,'arm':arm,'dev_ce':best_ce,'dev_accuracy':curve[str(best_step)]['acc'],'selected_step':best_step,'dev_curve':curve,'train_wall_s':training_s,'params':count_params(model),'bytes':bytes_inference(model),'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'init_diag':init,'dev_diag':final_diag}

def measured_latency(model,seed,task,pos,repeat=40):
    # 64 batch, fixed input from independent diagnostic stream but not Fresh. Warmup.
    x,_=make_data(task,seed,64,2)
    model.eval()
    with torch.no_grad():
        for _ in range(10):model(x)
        times=[]
        for _ in range(5):
            t=time.perf_counter()
            for _ in range(repeat):model(x)
            times.append((time.perf_counter()-t)*1e6/repeat)
    return {'batch64_forward_median_us':statistics.median(times),'block_samples_us':times}

def verify(out):
    rows=json.loads((out/'FULL_RESULTS.json').read_text())
    sealed=json.loads((out/'SEALED_MANIFEST.json').read_text())
    assert sealed['fresh_generated']==False
    max_diff=0.0;hash_count=0
    for r in rows:
        cp=out/'checkpoints'/r['checkpoint'];assert sha(cp)==r['checkpoint_sha256'];hash_count+=1
        sd=torch.load(cp,weights_only=True)
        m=Model(r['arm'],r['pos'],r['seed']);m.load_state_dict(sd['state']);m.eval()
        for split,label,clength in [(3,'fresh',P['train_sequence_content_length']),(4,'ood',P['ood_content_length'])]:
            x,y=make_data(r['task'],r['seed'],CONFIG['n_'+label],split,content_length=clength)
            sc=score(m,x,y);max_diff=max(max_diff,abs(sc['ce']-r[f'{label}_ce']),abs(sc['acc']-r[f'{label}_acc']))
        assert bytes_inference(m)==r['bytes']
    return {'checkpoint_hash_ok':hash_count,'fresh_ood_replays':len(rows),'max_abs_ce_or_acc_diff':max_diff,'seal_had_fresh_generated_false':sealed['fresh_generated']==False}

def analyze(rows):
    from collections import defaultdict
    bykey={(r['task'],r['seed'],r['pos'],r['arm']):r for r in rows}
    med=[];seeds=CONFIG['seeds']
    for task in TASKS:
        for pos in POSITIONS:
            for arm in ARMS:
                a=[bykey[(task,s,pos,arm)] for s in seeds]
                med.append({'task':task,'pos':pos,'arm':arm,'fresh_ce_median':statistics.median(x['fresh_ce'] for x in a),'fresh_acc_median':statistics.median(x['fresh_acc'] for x in a),'ood_ce_median':statistics.median(x['ood_ce'] for x in a),'lat_us_median':statistics.median(x['latency']['batch64_forward_median_us'] for x in a),'train_wall_s_median':statistics.median(x['train_wall_s'] for x in a),'params':a[0]['params'],'bytes_median':statistics.median(x['bytes'] for x in a)})
    assess=[]
    for s in seeds:
        def row(task,pos,arm):return bykey[(task,s,pos,arm)]
        m=row('last_matching_key','hybrid','mirror')['fresh_ce']
        f=row('last_matching_key','hybrid','film')['fresh_ce']
        r=row('last_matching_key','hybrid','coupled_rank1')['fresh_ce']
        pm=row('last_value','hybrid','mirror')['fresh_ce']
        pf=row('last_value','hybrid','film')['fresh_ce']
        rm=row('last_matching_key','rope','mirror')['fresh_ce']
        rf=row('last_matching_key','rope','film')['fresh_ce']
        assess.append({'seed':s,'hybrid_mirror_ce':m,'hybrid_film_ce':f,'hybrid_coupled_rank1_ce':r,'mirror_minus_film_pos_ce':pm-pf,'mirror_film_delta_hybrid':f-m,'mirror_film_delta_rope':rf-rm,'interaction_diff':(f-m)-(rf-rm),'main_paired_success':(m <= min(f,r)-.02 and pm-pf<=.02),'mirror_beats_both_by_0p02':m<=min(f,r)-.02,'positional_regression':pm-pf>.02})
    return {'condition_medians':med,'paired_main':assess,'primary_passes':sum(x['main_paired_success'] for x in assess),'primary_required':2,'verdict':'PASS' if sum(x['main_paired_success'] for x in assess)>=2 else 'FAIL'}

def main(out):
    torch.set_num_threads(1)
    if out.exists(): raise FileExistsError(f'Never overwrite: {out}')
    out.mkdir(parents=True);(out/'checkpoints').mkdir()
    shutil.copy2(BASE/'PROTOCOL.json',out/'PROTOCOL.json')
    env={'torch':torch.__version__,'python':platform.python_version(),'cpu':platform.processor(),'platform':platform.platform(),'torch_threads':torch.get_num_threads(),'protocol_sha256':sha(BASE/'PROTOCOL.json')};jwrite(out/'ENVIRONMENT.json',env)
    total=CONFIG['total_models'];rows=[];start=time.perf_counter()
    for task in TASKS:
        for seed in CONFIG['seeds']:
            for pos in POSITIONS:
                for arm in ARMS:
                    r=train_one(out,seed,task,pos,arm)
                    rows.append(r)
                    print(f'TRAIN {len(rows):3d}/{total} {r["key"]} DEV={r["dev_ce"]:.5f} train_s={r["train_wall_s"]:.2f}',flush=True)
    assert len(rows)==total
    # Seal raw checkpoints and protocol BEFORE any Fresh/OOD generation.
    jwrite(out/'SEALED_MANIFEST.json',{'id':'MA-1182','fresh_generated':False,'ood_generated':False,'protocol_sha256':sha(BASE/'PROTOCOL.json'),'rows':rows,'elapsed_preseal_s':time.perf_counter()-start})
    jwrite(out/'RUN_STATE.json',{'sealed_sha256':sha(out/'SEALED_MANIFEST.json'),'postseal_started':True})
    scored=[]
    for r in rows:
        cp=out/'checkpoints'/r['checkpoint'];assert sha(cp)==r['checkpoint_sha256']
        m=Model(r['arm'],r['pos'],r['seed']);m.load_state_dict(torch.load(cp,weights_only=True)['state']);m.eval()
        outrow=dict(r)
        for split,label,clength in [(3,'fresh',P['train_sequence_content_length']),(4,'ood',P['ood_content_length'])]:
            x,y=make_data(r['task'],r['seed'],CONFIG['n_'+label],split,content_length=clength)
            sc=score(m,x,y)
            outrow.update({f'{label}_ce':sc['ce'],f'{label}_acc':sc['acc']})
        outrow['latency']=measured_latency(m,r['seed'],r['task'],r['pos'])
        scored.append(outrow)
    jwrite(out/'FULL_RESULTS.json',scored)
    with (out/'RESULTS_CORE.csv').open('w',newline='') as h:
        keys=['key','task','seed','pos','arm','selected_step','dev_ce','dev_accuracy','fresh_ce','fresh_acc','ood_ce','ood_acc','params','bytes','train_wall_s','batch64_forward_us','checkpoint','checkpoint_sha256']
        w=csv.DictWriter(h,fieldnames=keys);w.writeheader()
        for r in scored:
            d={k:r.get(k) for k in keys};d['batch64_forward_us']=r['latency']['batch64_forward_median_us'];w.writerow(d)
    jwrite(out/'ANALYSIS.json',analyze(scored))
    jwrite(out/'VERIFICATION.json',{**verify(out),'source_protocol_sha256':sha(BASE/'PROTOCOL.json'),'results_csv_sha256':sha(out/'RESULTS_CORE.csv'),'sealed_manifest_sha256':sha(out/'SEALED_MANIFEST.json'),'total_models':len(scored),'total_wall_s':time.perf_counter()-start})
    print('FINISHED',time.perf_counter()-start,'verdict',json.loads((out/'ANALYSIS.json').read_text())['verdict'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);args=p.parse_args();main(args.out)
