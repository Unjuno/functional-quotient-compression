"""Frozen calibration and development runner; no fresh run without explicit unlock."""
import argparse,json,itertools,time,hashlib
from pathlib import Path
import torch
from .experiment import Model,make_world,fit,metrics,save_model,METHODS,FAMILIES

def configs(method:str)->list[dict]:
    if method=='learned':
        return [dict(lr=.001,axis_lr=lr,warmup=w,sigma=s,steps=800) for lr,w,s in itertools.product([.0003,.001],[0,100],[.03,.12])]
    sigmas=[.03] if method in ('shared','multihead','private_ffn') else [.03,.12]
    return [dict(lr=lr,axis_lr=lr,warmup=0,sigma=s,steps=800) for lr,s in itertools.product([.0003,.001,.003,.01],sigmas)]

def read_rows(root:Path)->list[dict]:
    return [json.loads(p.read_text()) for p in sorted(root.rglob('metric.json'))]

def select(root:Path)->dict:
    selected={}
    for family in ('aligned','digits8'):
        rows=read_rows(root/'calibration'/family)
        chosen={}
        for method in METHODS:
            group=[r for r in rows if r['label']==method]
            if len(group)!=len(configs(method)):raise ValueError('incomplete calibration: '+family+' '+method)
            best=min(group,key=lambda r:(r['val']['loss'],r['cfg']['lr'],r['cfg']['sigma']))
            chosen[method]=best['cfg']
        good=[r for r in rows if r['label'] in ('native4','film','lora2','householder','multihead')]
        chosen['_long_native']=min(good,key=lambda r:r['val']['loss'])['label']
        selected[family]=chosen
    out=root/'SELECTED.json';out.write_text(json.dumps(selected,indent=2,sort_keys=True))
    return selected

def run_one(root:Path,stage:str,family:str,seed:int,label:str,cfg:dict,index:int,full_steps:int):
    method='fixed' if label=='fixed_matched' else label
    dest=root/stage/family/f'seed{seed}'/f'{label}_{index}'
    if (dest/'DONE').exists():return
    dest.mkdir(parents=True,exist_ok=True)
    w=make_world(seed,family)
    model=Model(method,w['di'],w['o'],w['k'],seed,cfg['sigma'],w['digits'])
    # No test-set access during calibration.
    run_cfg=dict(cfg,steps=full_steps)
    trained=fit(model,w,run_cfg,seed)
    (dest/'trace.json').write_text(json.dumps(trained['trace']))
    snapshot_steps=sorted(trained['snapshots']) or [full_steps]
    for step in snapshot_steps:
        if step in trained['snapshots']:model.load_state_dict(trained['snapshots'][step])
        save_at=dest if step==800 else dest/f'step{step}'
        payload=save_model(model,save_at/'model.npz')
        row=dict(stage=stage,family=family,seed=seed,label=label,method=method,cfg=cfg,steps=step,
                 val=metrics(model,w,'val'),test=None if stage=='calibration' else metrics(model,w,'test'),
                 payload=payload,train_seconds=trained['snapshot_times'].get(step,trained['train_seconds']),
                 learned_scalars=sum(p.numel() for p in model.parameters() if p.requires_grad),
                 input_examples_per_step=128 if w['digits'] else 64,task_count=w['k'])
        (save_at/'metric.json').write_text(json.dumps(row,indent=2,sort_keys=True))
    (dest/'DONE').write_text('complete\n')
    last=row['val' if stage=='calibration' else 'test']
    print(stage,family,seed,label,index,full_steps,'loss',round(last['loss'],6),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['calibration','select','dev'])
    ap.add_argument('--family',choices=FAMILIES);ap.add_argument('--seed',type=int,default=17100)
    ap.add_argument('--out',type=Path,default=Path('runs'))
    args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.stage=='select':print(json.dumps(select(args.out),indent=2));return
    if args.family is None:raise ValueError('family required')
    if args.stage=='calibration':
        if args.seed!=17100 or args.family not in ('aligned','digits8'):raise ValueError('calibration firewall')
        for method in METHODS:
            for index,cfg in enumerate(configs(method)):
                run_one(args.out,args.stage,args.family,args.seed,method,cfg,index,800)
    else:
        if args.seed not in (17101,17102,17103):raise ValueError('development firewall')
        selected=json.loads((args.out/'SELECTED.json').read_text())['digits8' if args.family.startswith('digits') else 'aligned']
        native=selected['_long_native']
        for method in METHODS+['fixed_matched']:
            cfg=selected['learned' if method=='fixed_matched' else method]
            long=method in ('learned','fixed_matched',native) and args.family in ('aligned','digits8')
            run_one(args.out,'dev',args.family,args.seed,method,cfg,0,2400 if long else 800)
if __name__=='__main__':main()
