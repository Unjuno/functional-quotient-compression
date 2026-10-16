"""Prospectively locked, common-parent mid-training expansion control."""
import json,hashlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from engine import ROOT,CFG,Bank,fit,world,tensors,load,save,evaluate,sha
SPEC=json.loads((ROOT/'WIDEN_PROTOCOL.json').read_text())

def branch(parent,arm):
    q=1 if arm=='continue1' else 8
    m=Bank('mirror',q,parent.seeds)
    with torch.no_grad():
        for k,p in m.named_parameters():
            if k!='codes':p.copy_(getattr(parent,k))
        if arm!='expand8noise':m.codes.zero_()
        m.codes[:,:,:1].copy_(parent.codes)
        if arm=='expand8warm':m.router.zero_();m.rb.zero_()
    return m

def task(job):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    w,fam=job;n=SPEC['ntrain'];seeds=SPEC['inits'];lr=json.loads((ROOT/'FRESH_LOCK.json').read_text())['learning_rate']
    out=ROOT/'results'/'widen'/f'w{w}_{fam}'
    if (out/'records.json').exists():return json.loads((out/'records.json').read_text())
    r=fit(w,fam,n,'mirror1_free',seeds,lr,out/'prefix',steps=125)
    parent=Bank('mirror',1,seeds)
    with torch.no_grad():
        for i,row in enumerate(r['records']):
            old=load(ROOT/row['file'])
            for k,p in parent.named_parameters():p[i].copy_(getattr(old,k)[0])
    d=tensors(world(w,fam));stream=np.random.default_rng(w+400000).integers(0,n,size=(500,64));records=[]
    for arm in SPEC['arms']:
        m=branch(parent,arm);original=parent(d['val_x'],d['val_c'])[0].detach();at_split=m(d['val_x'],d['val_c'])[0].detach()
        insertion_error=(original-at_split).abs().max(1).values.tolist()
        optimizer=torch.optim.Adam(m.parameters(),lr=lr,foreach=False);start=time.perf_counter()
        for step in range(125,500):
            ids=stream[step];optimizer.zero_grad(set_to_none=True)
            pred=m(d['train_x'][ids],d['train_c'][ids],uniform=arm=='expand8warm' and step<250)[0]
            loss=(pred-d['train_y'][ids]).square().mean(1).sum();loss.backward()
            if step==375:
                for g in optimizer.param_groups:g['lr']=lr*.1
            optimizer.step()
        elapsed=time.perf_counter()-start;ev=evaluate(m,d,n)
        for i,seed in enumerate(seeds):
            path=out/f'{arm}_s{seed}.bin';h=save(m,i,path)
            row={'world':w,'family':fam,'ntrain':n,'method':arm,'init':seed,'test_mse':ev['test_mse'][i],'train_mse':ev['train_mse'][i],
                 'worst_mse':ev['worst_mse'][i],'usage':ev['usage'][i],'route_match':ev['route_match'][i],'expert_difference':ev['expert_difference'][i],
                 'best_test_mse':ev['test_mse'][i],'val_noisy_mse':ev['val_noisy_mse'][i],'bytes':path.stat().st_size,'file':str(path.relative_to(ROOT)),'sha256':h,
                 'insertion_error':insertion_error[i],'group_seconds':elapsed,'prefix_steps':125,'continuation_steps':375}
            records.append(row)
    (out/'records.json').write_text(json.dumps(records,indent=2));return records

def main():
    lock=json.loads((ROOT/'WIDEN_LOCK.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in lock['files'].items())
    jobs=[(w,f) for w in SPEC['worlds'] for f in SPEC['families']];records=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,res in enumerate(pool.map(task,jobs),1):records.extend(res);print('widen',i,len(jobs),flush=True)
    (ROOT/'WIDEN_RECORDS.json').write_text(json.dumps(records,indent=2));print('COMPLETE',len(records),flush=True)
if __name__=='__main__':main()
