"""MA-1176 pre-registered tiny scaling-axis research runner.

`train` generates train and development splits only, freezes checkpoint manifest.
`eval` verifies ALL checkpoint hashes and then opens fresh/OOD splits.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, sys, time
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from models import FeatureModel, LoopModel, LoopOnePassControl, model_from_spec, packet_bytes
from data import generate, TEACHER_WORLDS

ROOT = Path(__file__).resolve().parents[1]
PROTO = json.loads((ROOT / 'PROTOCOL.json').read_text(encoding='utf-8'))
SEEDS = PROTO['splits']['student_seeds']
THREADS = int(PROTO['optimizer']['torch_threads'])
torch.set_num_threads(THREADS)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def file_write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')


def specs():
    cases = {}
    def add(lane, world, seed, model, width=8, n=384, regime='clean_balanced', steps=240, init=0, loop_steps=1, distill=False):
        key = (world,seed,model,width,n,regime,steps,init,loop_steps,distill)
        if key not in cases:
            cid='{:03d}'.format(len(cases))
            cases[key]=dict(case_id=cid,world=world,seed=seed,model=model,width=width,train_size=n,
                            regime=regime,steps=steps,init=init,loop_steps=loop_steps,distill=distill,lanes=[])
        if lane not in cases[key]['lanes']:
            cases[key]['lanes'].append(lane)
    for world in TEACHER_WORLDS:
        for seed in SEEDS:
            # Representation comparisons (same width for Mirror, FiLM; independent h4 ~ similar bytes)
            for model,width in [('shared',8),('film',8),('mirror',8),('independent',4)]:
                add('representation',world,seed,model,width)
            for model in ('film','mirror'):
                for width in (4,8,12):
                    add('parameters',world,seed,model,width)
                for n in (96,384,1536):
                    add('unique_data',world,seed,model,8,n=n)
                for regime in ('clean_balanced','skew','corrupted','oracle_filtered'):
                    add('data_quality',world,seed,model,8,regime=regime)
            candidates=[('film',8),('mirror',8),('shared',8),('independent',4)]*2
            for i,(model,width) in enumerate(candidates):
                add('search_candidate',world,seed,model,width,steps=120,init=i+1)
            for updates in (120,480,960):
                add('search_no_search_baseline',world,seed,'film',8,steps=updates,init=100)
    for seed in SEEDS:
        for i in (1,2,3): add('loop','loop_teacher',seed,'loop',4,steps=320,loop_steps=i)
        add('loop','loop_teacher',seed,'onepass',4,steps=320)
        # All loop teachers have trained by this point in execution: per seed T3 is above
        add('loop_distillation','loop_teacher',seed,'loop',4,steps=320,loop_steps=1,distill=True,init=88)
    return list(cases.values())


def choose_teacher_checkpoint(case, records):
    for rec in records.values():
        s=rec['spec']
        if s['world']=='loop_teacher' and s['seed']==case['seed'] and s['model']=='loop' and s['loop_steps']==3 and not s['distill']:
            return ROOT/'checkpoints'/f"{s['case_id']}.pt", rec['sha256']
    raise RuntimeError('Loop T3 must train before student distillation')


def fit(spec, previous):
    seed=int(spec['seed'])
    world=spec['world']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99210+seed*197+spec['init']*1559+sum(ord(c) for c in spec['model'])+spec['width']*173)
        m=model_from_spec(spec)
    x,r,y=generate(world,seed,'train',spec['train_size'],spec['regime'])
    xd,rd,yd=generate(world,seed,'dev',PROTO['splits']['development_size'])
    sampler=torch.Generator().manual_seed(77312+seed*71+int(spec['train_size'])*13 + (0 if world==TEACHER_WORLDS[0] else 2000 if world==TEACHER_WORLDS[1] else 9000))
    teacher=None
    if spec['distill']:
        teacher_file,teacher_hash = choose_teacher_checkpoint(spec,previous)
        if sha(teacher_file)!=teacher_hash:
            raise RuntimeError('Teacher checkpoint was changed')
        state=torch.load(teacher_file,weights_only=True,map_location='cpu')
        teacher=LoopModel(3).eval()
        teacher.load_state_dict(state['state'])
        for p in teacher.parameters(): p.requires_grad_(False)
    opt=torch.optim.AdamW(m.parameters(),lr=PROTO['optimizer']['lr'],weight_decay=PROTO['optimizer']['weight_decay'])
    t0=time.perf_counter()
    m.train()
    for step in range(spec['steps']):
        idx=torch.randint(0,len(x),(PROTO['optimizer']['batch_size'],),generator=sampler)
        xp,rp,yp=x[idx],r[idx],y[idx]
        pred=m(xp,rp)
        loss=F.mse_loss(pred,yp)
        if teacher is not None:
            with torch.no_grad():
                target=teacher(xp,rp)
            loss = 0.65*loss + 0.35*F.mse_loss(pred,target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        opt.step()
    train_s=time.perf_counter()-t0
    m.eval()
    with torch.no_grad():
        out=m(xd,rd)
        dnmse=(out-yd).square().mean().item() / (yd.square().mean().item()+1e-12)
        outtrain=m(x,r)
        tnmse=(outtrain-y).square().mean().item()/(y.square().mean().item()+1e-12)
    cp=ROOT/'checkpoints'/f"{spec['case_id']}.pt"
    torch.save(dict(spec=spec,state=m.state_dict(),dev_nmse=dnmse,train_nmse=tnmse),cp)
    return dict(spec=spec,sha256=sha(cp),dev_nmse=dnmse,train_nmse=tnmse,
        checkpoint_bytes=cp.stat().st_size,parameter_count=m.num_parameters(),packet_bytes=packet_bytes(m),
        training_seconds=train_s,unique_train=len(x),examples_seen=spec['steps']*PROTO['optimizer']['batch_size'],
        forward_MAC_per_example=m.active_linear_macs(), extra_vector_ops_per_example=m.extra_vector_ops(),
        train_forward_MAC_proxy=m.active_linear_macs()*spec['steps']*PROTO['optimizer']['batch_size'])


def run_train():
    manifest=ROOT/'results'/'SEALED_TRAIN_MANIFEST.json'
    if manifest.exists():
        raise RuntimeError('Sealed manifest already exists; refusing to train or overwrite checkpoint evidence')
    cases=specs()
    print('MA-1176 pre-registered',len(cases),'distinct tiny model trainings', flush=True)
    print('PRE-REGISTERED PROTOCOL SHA256',sha(ROOT/'PROTOCOL.json'),flush=True)
    records={}
    started=time.perf_counter()
    for i,s in enumerate(cases):
        r=fit(s,records)
        records[s['case_id']]=r
        if (i+1)%20==0 or i+1==len(cases):
            print('TRAIN',i+1,'/',len(cases),f"elapsed={time.perf_counter()-started:.1f}s",'latest_dev=',round(r['dev_nmse'],5),flush=True)
    assert len(records)==len(cases)
    config={
        'id':'MA-1176','protocol_sha256':sha(ROOT/'PROTOCOL.json'),
        'source_sha256':{p.name:sha(p) for p in (ROOT/'source').glob('*.py')},
        'train_sealed_before_fresh':True,'fresh_or_ood_generated':False,
        'records':records,
        'platform':platform.platform(),'python':sys.version,'torch':torch.__version__,'threads':THREADS,
        'training_elapsed_s':time.perf_counter()-started
    }
    file_write_json(manifest,config)
    print('SEALED',manifest,len(records),'sha256',sha(manifest),flush=True)


def normalized(pred,target,rule):
    mse=(pred-target).square().mean().item()
    nmse=mse/(target.square().mean().item()+1e-12)
    per=[]
    for k in range(3):
        mask=rule==k
        if not mask.any(): per.append(None)
        else: per.append(((pred[mask]-target[mask]).square().mean()/(target[mask].square().mean()+1e-12)).item())
    return nmse,per


def bench(m):
    # Tiny CPU timings, not hardware-independent estimates.
    m.eval()
    vals=[]
    with torch.no_grad():
        for n in (1,32):
            x=torch.randn(n,4);r=torch.arange(n)%3
            for _ in range(4): m(x,r)
            t0=time.perf_counter()
            for _ in range(32): m(x,r)
            vals.append((time.perf_counter()-t0)*1e6/32)
    return vals


def run_eval():
    manifest=ROOT/'results'/'SEALED_TRAIN_MANIFEST.json'
    if not manifest.exists(): raise RuntimeError('Must run train first and seal all hashes')
    doc=json.loads(manifest.read_text(encoding='utf-8'))
    if doc['protocol_sha256']!=sha(ROOT/'PROTOCOL.json'):
        raise RuntimeError('Protocol changed since preregistration')
    if doc['source_sha256']!={p.name:sha(p) for p in (ROOT/'source').glob('*.py')}:
        raise RuntimeError('Source has changed since checkpoint sealing')
    cases=specs()
    if len(doc['records'])!=len(cases): raise RuntimeError('Manifest not complete')
    if (ROOT/'results'/'RESULTS_CORE.csv').exists(): raise RuntimeError('Fresh results already exist; cannot silently replace')
    for s in cases:
        rec=doc['records'][s['case_id']]
        path=ROOT/'checkpoints'/f"{s['case_id']}.pt"
        if not path.is_file() or sha(path)!=rec['sha256']:
            raise RuntimeError('HASH_FAIL '+s['case_id'])
    print('PASS: all checkpoint SHA256 verified; now opening fresh and OOD',len(cases),flush=True)
    rows=[]
    for i,s in enumerate(cases):
        checkpoint=torch.load(ROOT/'checkpoints'/f"{s['case_id']}.pt",map_location='cpu',weights_only=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1)
            m=model_from_spec(s)
        m.load_state_dict(checkpoint['state']);m.eval()
        metrics={}
        for part,size in [('dev',PROTO['splits']['development_size']),('fresh',PROTO['splits']['fresh_size']),('ood',PROTO['splits']['ood_size'])]:
            x,r,y=generate(s['world'],s['seed'],part,size)
            with torch.no_grad():p=m(x,r)
            nmse,per=normalized(p,y,r)
            metrics[part+'_nmse']=nmse
            for k,v in enumerate(per):metrics[f'{part}_rule{k}_nmse']=v
        if abs(metrics['dev_nmse']-doc['records'][s['case_id']]['dev_nmse'])>3e-6:
            raise RuntimeError('DEV_CHECKPOINT_REPLAY_FAIL '+s['case_id'])
        lats=bench(m)
        rec=doc['records'][s['case_id']]
        row={**s,**metrics,**{k:v for k,v in rec.items() if k!='spec'},'latency_batch1_us':lats[0],'latency_batch32_us':lats[1]}
        row['lanes']='|'.join(s['lanes'])
        rows.append(row)
        if (i+1)%40==0 or i+1==len(cases):print('EVAL',i+1,'/',len(cases),flush=True)
    fields=list(rows[0].keys())
    p=ROOT/'results'/'RESULTS_CORE.csv'
    with p.open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');wr.writeheader();wr.writerows(rows)
    file_write_json(ROOT/'results'/'EVAL_COMPLETE.json',dict(rows=len(rows),checkpoint_manifest_sha256=sha(manifest),results_sha256=sha(p),dev_checkpoint_replay='ALL_PASS',fresh_evaluated_after_manifest_seal=True))
    print('RESULTS',p,'rows',len(rows),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,choices=['train','eval']);args=parser.parse_args()
    if args.phase=='train':run_train()
    else:run_eval()

if __name__=='__main__':main()
