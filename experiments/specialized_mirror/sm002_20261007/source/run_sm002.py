"""SM002 locked development/main and measured-loop-time runners."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
import sm002 as s
ROOT=Path(__file__).resolve().parents[1]
MASTER_STEPS=20000
DEV_WORLD=92001
MAIN_WORLDS=[22001,22002,22003]
MAIN_SEEDS=[201,202,203]
LR_GRID=[.001,.003,.01]

def dump(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_bytes(s.canonical(obj)+b'\n');temp.replace(path)

def configure():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

def check_stage_world(stage,world,frozen):
    if stage=='dev' and world!=DEV_WORLD:raise ValueError('development may not inspect audit worlds')
    if stage in ('main','time') and (world not in MAIN_WORLDS or not frozen):raise ValueError('audit world locked until protocol freeze')

def master_stream(world,n_private,batch):
    # Always generate the SAME maximum-length stream, then take prefixes.
    # Calling old.stream with different total lengths would change its RNG layout.
    return s.stream(world,n_private,MASTER_STEPS,batch,world.seed+310000)

def optimize(model,world,n_private,data,targets,lr,steps,folder:Path,curve_every=500,time_budget=None):
    folder.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01,foreach=False)
    trainwall=0.;traincpu=0.;curve=[];losses=[];last=None;stop='updates'
    wall0=time.perf_counter();cpu0=time.process_time();model.train()
    for step in range(1,min(steps,len(data))+1):
        opt.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(data[step-1]),targets[step-1])
        if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss step{step}')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%100==0:last=float(loss.detach());losses.append({'step':step,'minibatch_nll':last})
        reached=(time_budget is not None and trainwall+time.perf_counter()-wall0>=time_budget)
        final=(step==min(steps,len(data)) or reached)
        if step%curve_every==0 or final:
            trainwall+=time.perf_counter()-wall0;traincpu+=time.process_time()-cpu0
            payload=s.encode(model);name=f'step{step:05d}.sm02';(folder/name).write_bytes(payload)
            quality=s.evaluate(model,world,n_private,wrong_views=False)
            curve.append({'step':step,'examples':step*data.shape[1],
                          'training_wall_seconds':trainwall,'process_cpu_seconds':traincpu,
                          'checkpoint':name,'payload_sha256':s.sha(payload),'quality':quality})
            if final:
                if reached:stop='wall_budget'
                break
            wall0=time.perf_counter();cpu0=time.process_time()
    return {'steps':step,'batch':data.shape[1],'lr':lr,'weight_decay':.01,'clip_grad_norm':1.,
            'training_wall_seconds':trainwall,'process_cpu_seconds':traincpu,'curve':curve,
            'losses':losses,'stop_reason':stop,'time_budget':time_budget,
            'wall_budget_overshoot':None if time_budget is None else trainwall-time_budget}

def parent(world,seed,width,base):
    path=base/'parents'/f'w{world.seed}_s{seed}_h{width}.sm02'
    if path.exists():
        meta=json.loads(path.with_suffix('.json').read_text());raw=path.read_bytes()
        assert s.sha(raw)==meta['payload_sha256'];return s.decode(raw),meta
    torch.manual_seed(seed);model=s.Model(s.Config(width=width))
    data=s.stream(world,0,1000,128,world.seed+210000)
    targets=world.targets(data.reshape(-1,3)).reshape(data.shape[:2])
    train=optimize(model,world,0,data,targets,.003,1000,base/'parent_snapshots'/path.stem,curve_every=1000)
    raw=s.encode(model);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    meta={'world':world.seed,'seed':seed,'width':width,'steps':1000,'batch':128,'lr':.003,
          'payload_sha256':s.sha(raw),'world_sha256':world.digest,'data_sha256':s.sha(data.numpy().tobytes()),
          'quality':s.evaluate(model,world,0),'train':train}
    dump(path.with_suffix('.json'),meta);return model,meta

def run_one(stage,world_id,seed,n_private,cfg,lr,steps=4000,time_budget=None):
    check_stage_world(stage,world_id,(ROOT/'FROZEN_PROTOCOL.json').exists())
    world=s.World(world_id);base=ROOT/('dev' if stage=='dev' else 'main')
    par,pinfo=parent(world,seed,cfg.width,base)
    key=f'w{world_id}_s{seed}_n{n_private:02d}_{s.METHODS[cfg.method]}_h{cfg.width}_lr{lr:g}'
    if stage=='time':key+='_wall'
    folder=ROOT/stage/'runs'/key
    if (folder/'result.json').exists():
        r=json.loads((folder/'result.json').read_text());assert s.sha((folder/'model.sm02').read_bytes())==r['payload_sha256'];return r
    folder.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed+600000);model=s.Model(cfg);model.copy_parent(par)
    initial=s.encode(model);(folder/'initial.sm02').write_bytes(initial)
    initial_quality=s.evaluate(model,world,n_private)
    data=master_stream(world,n_private,128)
    targets=world.targets(data.reshape(-1,3)).reshape(data.shape[:2])
    worldpath=base/'worlds'/f'w{world_id}.json'
    if not worldpath.exists():dump(worldpath,{'world':world_id,'labels':world.labels.tolist(),'sha256':world.digest})
    train=optimize(model,world,n_private,data,targets,lr,steps,folder/'snapshots',time_budget=time_budget)
    raw=s.encode(model);(folder/'model.sm02').write_bytes(raw)
    audit=s.evaluate(s.decode(raw),world,n_private,wrong_views=True)
    r={'id':key,'stage':stage,'world':world_id,'init_seed':seed,'n_private':n_private,
       'method':s.METHODS[cfg.method],'config':asdict(cfg),'parent_payload_sha256':pinfo['payload_sha256'],
       'parent_quality':pinfo['quality'],'parent_train_cpu_seconds':pinfo['train']['process_cpu_seconds'],
       'parent_train_wall_seconds':pinfo['train']['training_wall_seconds'],
       'initial_payload_sha256':s.sha(initial),'initial_quality':initial_quality,
       'world_sha256':world.digest,'master_stream_sha256':s.sha(data.numpy().tobytes()),
       'used_stream_sha256':s.sha(data[:train['steps']].numpy().tobytes()),
       'exposure':s.coverage(data[:train['steps']],n_private),'train':train,
       'serialized_bytes':len(raw),'learned_parameters':sum(p.numel() for p in model.parameters()),
       'linear_forward_macs_per_example':s.active_macs(cfg),'payload_sha256':s.sha(raw),'quality':audit}
    dump(folder/'result.json',r)
    print(s.canonical({'id':key,'private':audit['private_accuracy'],'common':audit['common_accuracy'],
                      'steps':train['steps'],'bytes':len(raw),'train_cpu':train['process_cpu_seconds'],
                      'train_wall':train['training_wall_seconds']}).decode(),flush=True)
    return r

def prepare_dev():
    for w in sorted(set(s.fit_config(m).width for m in s.PRIMARY)):
        parent(s.World(DEV_WORLD),171,w,ROOT/'dev')
    dump(ROOT/'ARCHITECTURES.json',{s.METHODS[m]:{'config':asdict(s.fit_config(m)),
         'bytes':len(s.encode(s.Model(s.fit_config(m))))} for m in s.PRIMARY})

def run_dev(bucket):
    jobs=[(m,lr) for m in s.PRIMARY for lr in LR_GRID]
    for j,(m,lr) in enumerate(jobs):
        if j%3==bucket:run_one('dev',DEV_WORLD,171,32,s.fit_config(m),lr)

def freeze():
    files=sorted((ROOT/'dev'/'runs').glob('*/result.json'))
    runs=[json.loads(p.read_text()) for p in files]
    assert len(runs)==24
    chosen={}
    for method in s.PRIMARY:
        cells=[r for r in runs if r['config']['method']==method];assert len(cells)==3
        best=min(cells,key=lambda r:(r['quality']['common_accuracy']<.99,-r['quality']['private_accuracy'],r['quality']['private_nll'],r['train']['lr']))
        chosen[str(method)]={'method':s.METHODS[method],'config':best['config'],'lr':best['train']['lr'],
                             'selected_development_id':best['id'],'development_quality':best['quality']}
    obj={'experiment':'SM002','created_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
         'base_commit':'0c3e5d78c9587ffb1ae7e8bf62b2ca4f9309fb61','cap_bytes':s.CAP,
         'selected':chosen,'main_worlds':MAIN_WORLDS,'main_seeds':MAIN_SEEDS,'private_counts':[4,8,16,32],
         'steps':4000,'batch':128,'master_steps':MASTER_STEPS,'primary_methods':s.PRIMARY,
         'controls':{'7':[16,32],'8':[16,32],'9':[32]},'development_lr_grid':LR_GRID,
         'source_sha256':{p.name:s.sha(p.read_bytes()) for p in (ROOT/'source').glob('*.py')},
         'plan_sha256':s.sha((ROOT/'PLAN.md').read_bytes()),'wall_budget_seconds':None}
    if (ROOT/'FROZEN_PROTOCOL.json').exists():raise ValueError('already frozen; never overwrite')
    dump(ROOT/'FROZEN_PROTOCOL.json',obj);print(json.dumps({v['method']:v['lr'] for v in chosen.values()},indent=2))

def run_main(index):
    p=json.loads((ROOT/'FROZEN_PROTOCOL.json').read_text());world=MAIN_WORLDS[index];seed=MAIN_SEEDS[index]
    jobs=[(n,m) for n in p['private_counts'] for m in s.PRIMARY]
    jobs.extend((n,m) for m,ns in [(7,[16,32]),(8,[16,32]),(9,[32])] for n in ns)
    # Fixed shuffle independent of quality to avoid systematic method/time ordering.
    import random;random.Random(750+index).shuffle(jobs)
    for n,m in jobs:
        if m in s.PRIMARY:
            chosen=p['selected'][str(m)];cfg=s.Config(**chosen['config']);lr=chosen['lr']
        else:
            cfg=s.Config(width=64,method=m);lr=p['selected']['1']['lr'] if m in (7,8) else .003
        run_one('main',world,seed,n,cfg,lr)

def run_time():
    p=json.loads((ROOT/'FROZEN_PROTOCOL.json').read_text());budget=json.loads((ROOT/'TIME_PROTOCOL.json').read_text())['budget_seconds']
    jobs=[(i,m) for i in range(3) for m in s.PRIMARY]
    import random;random.Random(861).shuffle(jobs)
    for i,m in jobs:
        choice=p['selected'][str(m)]
        run_one('time',MAIN_WORLDS[i],MAIN_SEEDS[i],32,s.Config(**choice['config']),choice['lr'],MASTER_STEPS,budget)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare-dev','dev','freeze','main','time']);ap.add_argument('--index',type=int,default=0)
    a=ap.parse_args();configure()
    if a.action=='prepare-dev':prepare_dev()
    elif a.action=='dev':run_dev(a.index)
    elif a.action=='freeze':freeze()
    elif a.action=='main':run_main(a.index)
    else:run_time()
if __name__=='__main__':main()
