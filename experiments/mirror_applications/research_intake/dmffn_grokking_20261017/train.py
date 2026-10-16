"""Modular-arithmetic grokking probe with exact train/test split and checkpoint history."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import Config,MirrorSwiGLU,mod_dataset,checkpoint_bytes

ROOT=Path(__file__).resolve().parent
METHODS=['mirror','gain','pair_mirror','tied','independent']

def source_sha():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'model.py',ROOT/'train.py',ROOT/'tests/test_model.py']}

def split_data(p,world,frac):
    x,y=mod_dataset(p)
    rng=np.random.default_rng(world)
    order=rng.permutation(len(x));ntrain=int(round(len(x)*frac));sel=order[:ntrain];audit=order[ntrain:]
    return (x[sel],y[sel]),(x[audit],y[audit]),sel,audit

@torch.no_grad()
def metrics(model,x,y):
    logits=model(x);prob=F.cross_entropy(logits.reshape(-1,y.max().item()+1) if False else logits.reshape(-1,model.cfg.p),y.reshape(-1))
    acc=(logits.argmax(-1)==y).float().mean().item()
    per_pair_acc=(logits.argmax(-1)==y).float().mean(0).tolist()
    return {'ce':float(prob),'accuracy':float(acc),'min_pair_accuracy':float(min(per_pair_acc))}

def train_one(world,seed,method,cfg,lr,wd,frac,steps,interval=200,store_payload=False):
    train,audit,sel,sel_test=split_data(cfg.p,world,frac)
    torch.manual_seed(seed)
    model=MirrorSwiGLU(cfg,method)
    initial_bytes=checkpoint_bytes(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    tstart=time.perf_counter()
    records=[]
    for step in range(steps+1):
        if step==0 or step%interval==0 or step==steps:
            model.eval()
            tr=metrics(model,*train);te=metrics(model,*audit)
            records.append({'world':world,'seed':seed,'method':method,'step':step,
               'train_ce':tr['ce'],'test_ce':te['ce'],'train_accuracy':tr['accuracy'],
               'test_accuracy':te['accuracy'],'test_min_pair_accuracy':te['min_pair_accuracy']})
            model.train()
        if step==steps:break
        logits=model(train[0])
        loss=F.cross_entropy(logits.reshape(-1,cfg.p),train[1].reshape(-1))
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    seconds=time.perf_counter()-tstart
    blob=io.BytesIO();torch.save({'model':model.state_dict(),'method':method,'cfg':cfg.__dict__,'world':world,'seed':seed,'trainfrac':frac},blob)
    raw=blob.getvalue()
    return {'history':records,'world':world,'seed':seed,'method':method,'elapsed_seconds':seconds,
       'parameters':model.param_count(),'serialized_inference_bytes':initial_bytes,'final_train_ce':records[-1]['train_ce'],
       'final_test_ce':records[-1]['test_ce'],'final_train_accuracy':records[-1]['train_accuracy'],
       'final_test_accuracy':records[-1]['test_accuracy'],'train_count':len(sel),'test_count':len(sel_test),
       'payload_sha256':hashlib.sha256(raw).hexdigest(),'raw':raw if store_payload else None}

def do_run(phase,worlds,seeds,methods,steps,lr,wd,frac,p,h,q,out,interval=200,store_payload=True):
    os.makedirs(out,exist_ok=True)
    cfg=Config(p=p,h=h,q=q)
    setup={'phase':phase,'worlds':worlds,'seeds':seeds,'methods':methods,'steps':steps,'lr':lr,'wd':wd,'frac':frac,'p':p,'h':h,'q':q,'interval':interval,'source_hashes':source_sha(),'environment':{'python':__import__('platform').python_version(),'torch':torch.__version__,'num_threads':torch.get_num_threads()}}
    (Path(out)/'RUN_SETUP.json').write_text(json.dumps(setup,indent=2,default=str)+'\n')
    result_path=Path(out)/'results.jsonl';hist_path=Path(out)/'history.csv';mode='a' if hist_path.exists() else 'w'
    hist_fields=list(train_one(worlds[0],seeds[0],methods[0],cfg,lr,wd,frac,0)['history'][0])
    with hist_path.open(mode,newline='') as f:
      wr=csv.DictWriter(f,fieldnames=hist_fields)
      if mode=='w':wr.writeheader()
    for w in worlds:
      for seed in seeds:
        for meth in methods:
          r=train_one(w,seed,meth,cfg,lr,wd,frac,steps,interval,store_payload)
          if store_payload:
            (Path(out)/f'model_{w}_{seed}_{meth}.pt').write_bytes(r.pop('raw'))
          else:r.pop('raw')
          with hist_path.open('a',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=hist_fields);wr.writerows(r.pop('history'))
          with result_path.open('a') as f:f.write(json.dumps(r)+'\n')
          print(f"{phase} w{w} s{seed} {meth} step {steps} train {r['final_train_accuracy']:.3f} test {r['final_test_accuracy']:.3f} ce {r['final_test_ce']:.4f} t={r['elapsed_seconds']:.2f}s",flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',required=True,choices=['dev','fresh','followup'])
    ap.add_argument('--worlds',type=int,nargs='+',required=True)
    ap.add_argument('--seeds',type=int,nargs='+',required=True)
    ap.add_argument('--methods',nargs='+',default=METHODS)
    ap.add_argument('--steps',type=int,default=12000)
    ap.add_argument('--interval',type=int,default=200)
    ap.add_argument('--lr',type=float,default=0.005)
    ap.add_argument('--wd',type=float,default=0.5)
    ap.add_argument('--frac',type=float,default=0.6)
    ap.add_argument('--p',type=int,default=31)
    ap.add_argument('--h',type=int,default=64)
    ap.add_argument('--q',type=int,default=16)
    ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(1)
    do_run(a.phase,a.worlds,a.seeds,a.methods,a.steps,a.lr,a.wd,a.frac,a.p,a.h,a.q,a.out,a.interval)

if __name__=='__main__':main()
