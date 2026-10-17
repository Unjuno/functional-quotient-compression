"""MA-1164 matched-native extension, fresh partitions distinct from prior MA-1164.

Two phases: development LR selection, then independently fixed fresh evaluation.
Parent module performs training of full NN and label-free Mirror weight compilation.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / 'source'))
from core import CoeffDecoder, N, state_bytes, logits
from importlib.util import spec_from_file_location, module_from_spec
_parent_spec = spec_from_file_location('ma1164_parent_runner', BASE/'source'/'run.py')
_parent = module_from_spec(_parent_spec)
_parent_spec.loader.exec_module(_parent)
world_data, native_train, native_flat, fit_teacher_weights, eval_model, median_forward_ms = (
    getattr(_parent, name) for name in ('world_data','native_train','native_flat',
    'fit_teacher_weights','eval_model','median_forward_ms'))

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
DEV_WORLDS = (618440, 618441)
FRESH_WORLDS = (618450, 618451, 618452)
DEV_SEEDS=(40,)
FRESH_SEEDS=(51,52)
HIDDEN=(6,7,10,11,13,16)
LRS=(.004,.008,.016)
MS=(256,512,768,1024)
KR={256:(2,4),512:(2,4),768:(8,4),1024:(2,4)}
STEPS=900
BATCH=128


class NativeWidth(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h=int(h)
        self.l1=nn.Linear(64,h,dtype=torch.float64)
        self.l2=nn.Linear(h,10,dtype=torch.float64)

    def forward(self,x):return self.l2(torch.tanh(self.l1(x)))

    @property
    def trainable(self):return sum(p.numel() for p in self.parameters())


def payload(net, dtype=torch.float64):
    return {'kind':'native_width', 'h':net.h,'stored_dtype':str(dtype),'state':{
        n:p.detach().cpu().to(dtype).contiguous().clone() for n,p in net.named_parameters()
    }}


def replay_payload(p, x):
    w={k:v.to(torch.float64) for k,v in p['state'].items()}
    return F.linear(torch.tanh(F.linear(x,w['l1.weight'],w['l1.bias'])),w['l2.weight'],w['l2.bias'])


def convert_full_teacher_to_width(teacher):
    """Lossless NativeMLP -> NativeWidth(16) structural adapter; no fitting/data."""
    model = NativeWidth(16)
    with torch.no_grad():
        model.l1.load_state_dict(teacher.l1.state_dict())
        model.l2.load_state_dict(teacher.l2.state_dict())
    return model


def train_small(h,lr,seed,world,xx,yy,splits):
    torch.manual_seed(1000+seed)
    g=torch.Generator().manual_seed(world*17+seed*103+37)
    model=NativeWidth(h)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
    idx=splits['train']
    start=time.perf_counter()
    for step in range(STEPS):
        take=idx[torch.randint(len(idx),(BATCH,),generator=g)]
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(xx[take]),yy[take])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
    return model,time.perf_counter()-start


def add_native_row(rows,model,world,seed,lr,xx,yy,split,part,train_s,dtype,phase):
    pay=payload(model,dtype)
    fun=lambda x:replay_payload(pay,x)
    acc,nll=eval_model(fun,xx,yy,split,part)
    rows.append({'phase':phase,'world':world,'seed':seed,'family':'native','kind':f'native_h{model.h}_{str(dtype).split(".")[-1]}',
                 'hidden':model.h,'m':model.trainable,'k':0,'r':0,'trainable_params':model.trainable,
                 'teacher_train_s':0.,'fit_s':0.,'train_s':train_s,'lr':lr,'updates':STEPS,
                 'accuracy':acc,'nll':nll,'serialized_bytes':state_bytes(pay),
                 'infer_batch64_ms':median_forward_ms(fun,xx[split[part][:64]]),
                 'train_forward_mac_proxy':64*model.h+model.h*10})


def mirror_row(world,seed,xx,yy,splits,target,teacher_s,m,kind):
    k,r=KR[m]
    model,fit=fit_teacher_weights(target,kind,m,k=k,r=r)
    acc,nll=eval_model(model,xx,yy,splits,'test')
    return {'phase':'fresh','world':world,'seed':seed,'family':'compiled','kind':kind,
            'hidden':16,'m':m,'k':k if kind=='mirror' else 0,'r':r if kind=='mirror' else 0,
            'trainable_params':m,'teacher_train_s':teacher_s,'fit_s':fit['fit_seconds'],
            'train_s':0.,'lr':0.,'updates':STEPS,'accuracy':acc,'nll':nll,
            'serialized_bytes':state_bytes(model.payload()),
            'infer_batch64_ms':median_forward_ms(model,xx[splits['test'][:64]]),
            'train_forward_mac_proxy':64*16+16*10,
            'weight_l2':fit['weight_l2'],
            'teacher_assisted':True}


def write_rows(rows,path):
    if path.exists():raise FileExistsError(path)
    fields=list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def select_dev():
    out=RES/'development.csv'
    if out.exists():raise FileExistsError(out)
    rows=[]
    for w in DEV_WORLDS:
        xx,yy,splits=world_data(w)
        for seed in DEV_SEEDS:
            for h in HIDDEN:
                for lr in LRS:
                    model,sec=train_small(h,lr,seed,w,xx,yy,splits)
                    add_native_row(rows,model,w,seed,lr,xx,yy,splits,'dev',sec,torch.float64,'development')
                    print(f'DEV {w} h={h} lr={lr} dev={rows[-1]["accuracy"]:.4f}',flush=True)
    write_rows(rows,out)
    selected={}
    for h in HIDDEN:
        opts=[]
        for lr in LRS:
            a=[x for x in rows if x['hidden']==h and x['lr']==lr]
            assert len(a)==2
            opts.append({'lr':lr,'accuracy':sum(x['accuracy'] for x in a)/2,'nll':sum(x['nll'] for x in a)/2})
        best=sorted(opts,key=lambda x:(-x['accuracy'],x['nll'],x['lr']))[0]
        selected[str(h)]=best
    path=RES/'selection.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps({'source':'development.csv','selected':selected,'worlds':DEV_WORLDS,
                                 'do_not_modify_after_fresh':True},indent=2))
    print('SELECTION',selected,flush=True)


def fresh_run():
    selpath=RES/'selection.json'
    if not selpath.exists():raise RuntimeError('Development-first required')
    selection=json.loads(selpath.read_text())['selected']
    assert set(selection)==set(map(str,HIDDEN))
    out=RES/'fresh.csv'
    if out.exists():raise FileExistsError(out)
    rows=[]
    for w in FRESH_WORLDS:
        xx,yy,splits=world_data(w)
        for seed in FRESH_SEEDS:
            teacher,meta=native_train(seed,w,xx,yy,splits)
            target=native_flat(teacher)
            for h in HIDDEN:
                lr=selection[str(h)]['lr']
                if h==16 and lr==.008:
                    model,sec=convert_full_teacher_to_width(teacher),meta['native_train_s']
                else:
                    model,sec=train_small(h,lr,seed,w,xx,yy,splits)
                for dtype in (torch.float64,torch.float16):
                    add_native_row(rows,model,w,seed,lr,xx,yy,splits,'test',sec,dtype,'fresh')
            for m in MS:
                for kind in ('plain','mirror'):
                    rows.append(mirror_row(w,seed,xx,yy,splits,target,meta['native_train_s'],m,kind))
            print(f'FRESH {w} seed={seed} rows_total={len(rows)}',flush=True)
    write_rows(rows,out)
    environment={'torch':torch.__version__,'cpu_threads':torch.get_num_threads(),'cuda':torch.cuda.is_available(),
        'platform':platform.platform(),'cpu_name':open('/proc/cpuinfo').read().split('model name')[1].split(':',1)[1].split('\n')[0].strip(),
        'description':'CPU eager FP64 run; FP16 serialized native weights replayed in FP64; 64-image forward medians'}
    (RES/'environment.json').write_text(json.dumps(environment,indent=2))
    print('FRESH DONE',len(rows),flush=True)


def audit():
    import pandas as pd
    dev=pd.read_csv(RES/'development.csv')
    fresh=pd.read_csv(RES/'fresh.csv')
    assert len(dev)==len(DEV_WORLDS)*len(DEV_SEEDS)*len(HIDDEN)*len(LRS)
    assert len(fresh)==len(FRESH_WORLDS)*len(FRESH_SEEDS)*(len(HIDDEN)*2+len(MS)*2)
    assert sorted(fresh.world.unique().tolist())==list(FRESH_WORLDS)
    assert sorted(fresh.seed.unique().tolist())==list(FRESH_SEEDS)
    assert not fresh.duplicated(['world','seed','kind','m']).any()
    summary=fresh.groupby(['kind','m'],dropna=False).agg(accuracy=('accuracy','mean'),nll=('nll','mean'),
                bytes=('serialized_bytes','mean'),infer_ms=('infer_batch64_ms','median'),count=('accuracy','count')).reset_index()
    summary.to_csv(RES/'summary.csv',index=False)
    ctrl=fresh[fresh.kind.str.startswith('native_h') & fresh.kind.str.endswith('float64')]
    for m,hlo,hhi in ((512,6,7),(768,10,11)):
        mir=fresh[(fresh.kind=='mirror')&(fresh.m==m)]
        for h in (hlo,hhi):
            base=ctrl[ctrl.hidden==h]
            delta=mir.accuracy.to_numpy()-base.accuracy.to_numpy()
            print(f'mirror m={m} vs native h{h} params={int(base.m.iloc[0])}: mirror={mir.accuracy.mean():.5f} native={base.accuracy.mean():.5f} delta_pp={100*delta.mean():+.4f}',flush=True)
    print(summary[['kind','m','accuracy','bytes','infer_ms']].to_string(index=False))
    manifest={'status':'SCREENING','dev_rows':len(dev),'fresh_rows':len(fresh),
              'worlds':FRESH_WORLDS,'seeds':FRESH_SEEDS,'dataset':'sklearn digits (same original dataset reused)',
              'error_check':'No test splits used for LR selection; source/program hashes frozen separately',
              'unfairness':'Mirror compressed from a separately trained full-size teacher while native narrow MLP was directly trained; teacher compute charged separately.',
              'result_scope':'Matched scalars and serialized bytes exploratory follow-up; no automatic Mirror advantage claim',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),
              'selection_sha256':hashlib.sha256((RES/'selection.json').read_bytes()).hexdigest()}
    (RES/'verification.json').write_text(json.dumps(manifest,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True,choices=('dev','fresh','audit'))
    a=ap.parse_args();torch.set_num_threads(1);RES.mkdir(exist_ok=True)
    if a.phase=='dev':select_dev()
    elif a.phase=='fresh':fresh_run()
    else:audit()

if __name__=='__main__':main()
