"""Isolated development-only calibration; freeze equal wall-loop budget pre-main."""
from pathlib import Path
import json,math,time,platform,os,subprocess,statistics
import torch
import sm002 as s
from run_sm002 import ROOT,configure,dump

def step(model,opt,rows,y):
    opt.zero_grad(set_to_none=True)
    loss=torch.nn.functional.cross_entropy(model(rows),y)
    if not torch.isfinite(loss):raise FloatingPointError('loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()

def main():
    configure();p=json.loads((ROOT/'FROZEN_PROTOCOL.json').read_text());result={}
    world=s.World(92001);data=s.stream(world,32,800,128,92001+841)
    target=world.targets(data.reshape(-1,3)).reshape(800,128)
    # Deterministic rotated order; no audit-world data or quality queried.
    for m in [4,0,10,2,6,1,5,3]:
        c=p['selected'][str(m)];path=ROOT/'dev'/'runs'/c['selected_development_id']/'model.sm02'
        model=s.decode(path.read_bytes());opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=.01,foreach=False)
        for i in range(30):step(model,opt,data[i],target[i])
        groups=[]
        for g in range(7):
            t0=time.perf_counter();c0=time.process_time()
            for k in range(100):step(model,opt,data[30+g*100+k],target[30+g*100+k])
            groups.append({'wall_seconds':time.perf_counter()-t0,'cpu_seconds':time.process_time()-c0,'updates':100})
        per=[g['wall_seconds']/100 for g in groups]
        result[s.METHODS[m]]={'groups':groups,'median_seconds_per_update':statistics.median(per),
                              'min_seconds_per_update':min(per),'max_seconds_per_update':max(per),
                              'config':c['config'],'lr':c['lr'],'bytes':path.stat().st_size,
                              'linear_forward_macs_per_example':s.active_macs(model.config)}
    t=statistics.median([r['median_seconds_per_update'] for r in result.values()])
    budget=math.ceil(4000*t*4)/4
    env={'python':platform.python_version(),'torch':torch.__version__,'numpy':__import__('numpy').__version__,
         'platform':platform.platform(),'cpu_count':os.cpu_count(),'threads':torch.get_num_threads(),
         'cuda':torch.cuda.is_available(),'dtype':'FP32','backend':'eager CPU','batch':128,
         'lscpu':subprocess.check_output(['lscpu'],text=True),
         'cpuinfo':Path('/proc/cpuinfo').read_text(),'cpu_quota':Path('/sys/fs/cgroup/cpu.max').read_text(),
         'clock_fixed':False,'benchmark_warmup':30,'benchmark_repeats':'7 groups x 100 updates',
         'main_concurrent_workers':3,'time_followup_concurrent_workers':1}
    dump(ROOT/'ENVIRONMENT.json',env)
    dump(ROOT/'CALIBRATION.json',{'methods':result,'environment':env,'budget_seconds':budget,
                                 'protocol_sha256':s.sha((ROOT/'FROZEN_PROTOCOL.json').read_bytes())})
    dump(ROOT/'TIME_PROTOCOL.json',{'created_before_main_worlds':True,'budget_seconds':budget,
          'selection':'ceil_to_0.25s(4000*median_across_eight_method_median_update_seconds)',
          'timed_segment':'optimizer loop only; includes indexing, forward/backward, gradient clipping, AdamW and budget-check overhead; excludes data preparation, pretraining, evaluation and serialization',
          'private_counts':[32],'worlds':[22001,22002,22003],'primary_methods':s.PRIMARY,
          'stop':'first full optimizer update reaching elapsed budget; report overshoot',
          'max_steps':20000,'batch':128,'protocol_sha256':s.sha((ROOT/'FROZEN_PROTOCOL.json').read_bytes())})
    for name,r in result.items():print(name,round(r['median_seconds_per_update']*1000,3),'ms/update')
    print('FROZEN WALL BUDGET',budget,'s')
if __name__=='__main__':main()
