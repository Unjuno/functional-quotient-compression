"""MA-1177: preregistered controlled Softmax Fix × Mirror Attention experiment.

Usage: python source/run.py train       (hard + matched-student distillation, seals checkpoints)
       python source/run.py evaluate    (requires complete seal, creates fresh/OOD)

No tuning/selection on fresh/OOD. This source itself is checksum-recorded in seal.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import torch
from torch.nn import functional as F
from core import *

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
CHECKPOINTS = ROOT / 'checkpoints'
HARD_UPDATES, DISTILL_UPDATES, BATCH = 550,150,64


def file_sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def model_outputs(model, data):
    with torch.no_grad():
        return model(data['q'],data['k'],data['rule']).detach()


def choose_batch_order(seed:int,steps:int,lane:str):
    sg=seed*900+(8 if lane=='hard' else 12018)
    return torch.randint(0,384,(steps,BATCH),generator=torch.Generator().manual_seed(sg))


def margin_stats(z,y):
    margins=hard_target_margin(z,y)
    return dict(margin_max=float(margins.max()),margin_mean=float(margins.mean()),
                margin_gt_16p6=float((margins>16.6).float().mean()),
                margin_gt_5p5=float((margins>5.5).float().mean()))


def gradient_balance(logits, targets, loss_type):
    test_logits=logits.detach().clone().requires_grad_(True)
    criterion=standard_ce if loss_type=='standard_ce' else fixed_ce
    l=criterion(test_logits, targets)
    g=torch.autograd.grad(l,test_logits)[0]
    return float(g.sum(-1).abs().max())


def train_hard(world,seed,kind,beta,loss_type,tr,dev,indices):
    model=TinyAttention(kind,seed,beta)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.016,eps=1e-8,weight_decay=.05)
    criterion=standard_ce if loss_type=='standard_ce' else fixed_ce
    history=[];t0=time.perf_counter()
    for step,ix in enumerate(indices,1):
        logits=model(tr['q'][ix],tr['k'][ix],tr['rule'][ix]);loss=criterion(logits,tr['target'][ix])
        optimizer.zero_grad(set_to_none=True)
        loss.backward(); optimizer.step()
        if step%50==0 or step==1:
            dev_logits=model_outputs(model,dev)
            tr_logits=model_outputs(model,tr)
            history.append(dict(step=step,dev_ce=fp64_cross_entropy(dev_logits,dev['target']),
                                train_ce=fp64_cross_entropy(tr_logits,tr['target']),
                                dev_top1=float((dev_logits.argmax(-1)==dev['target']).float().mean()),
                                **margin_stats(tr_logits,tr['target'])))
    wall=time.perf_counter()-t0
    dev_logits=model_outputs(model,dev)
    train_logits=model_outputs(model,tr)
    hchanges=[h['dev_ce']-g['dev_ce'] for g,h in zip(history,history[1:])]
    margin=margin_stats(train_logits,tr['target'])
    meta={"lane":"hard","world":world,"seed":seed,"kind":kind,"beta":beta,"loss":loss_type,
          "steps":HARD_UPDATES,"wall_s":wall,"parameter_count":sum(p.numel() for p in model.parameters()),
          "dev_ce_fp64":fp64_cross_entropy(dev_logits,dev['target']),
          "dev_accuracy":float((dev_logits.argmax(-1)==dev['target']).float().mean()),
          "train_ce_fp64":fp64_cross_entropy(train_logits,tr['target']),
          "worst_dev_ce_step_change":max(hchanges) if hchanges else 0.0,
          "gradient_balance_max":gradient_balance(train_logits[:64],tr['target'][:64],loss_type),
          **margin,"history":history}
    return model,meta


def train_distill(world,seed,kind,eps,loss_kind,tr,indices,teacher_state):
    teacher=TinyAttention(kind,seed,12.0)
    teacher.load_state_dict(teacher_state)
    teacher.eval()
    student=TinyAttention(kind,seed,12.0)
    student.load_state_dict(teacher_state)
    optim=torch.optim.AdamW(student.parameters(),lr=.02,eps=eps,weight_decay=0.0)
    soft_loss=standard_soft_ce if loss_kind=='standard_soft_ce' else fixed_soft_ce
    pre_st=model_outputs(teacher,tr)
    pre_student=model_outputs(student,tr)
    assert torch.equal(pre_st,pre_student)
    z0=pre_student[:64].detach().requires_grad_(True)
    grad0=torch.autograd.grad(soft_loss(z0,pre_st[:64]),z0)[0]
    init_grad_norm=float(grad0.norm())
    t0=time.perf_counter()
    for step,ix in enumerate(indices,1):
        with torch.no_grad():t=teacher(tr['q'][ix],tr['k'][ix],tr['rule'][ix])
        s=student(tr['q'][ix],tr['k'][ix],tr['rule'][ix])
        loss=soft_loss(s,t)
        optim.zero_grad(set_to_none=True);loss.backward();optim.step()
    wall=time.perf_counter()-t0
    displacements=[(student.state_dict()[k]-teacher_state[k]).abs().max().item() for k in teacher_state]
    meta={"lane":"distill","world":world,"seed":seed,"kind":kind,"beta":12.0,
          "loss":loss_kind,"eps":eps,"steps":DISTILL_UPDATES,"wall_s":wall,
          "parameter_count":sum(p.numel() for p in student.parameters()),
          "initial_equal_logits_grad_norm":init_grad_norm,
          "max_param_abs_displacement":max(displacements),
          "train_teacher_student_fp64_kl":fp64_distill_kl(model_outputs(student,tr),pre_st)}
    return student,meta


def save_model(model,meta):
    key=f"{len(list(CHECKPOINTS.glob('*.pt'))):03d}_{meta['lane']}_{meta['world']}_{meta['seed']}_{meta['kind']}_{meta['beta']}_{meta['loss']}_{meta.get('eps','na')}"
    path=CHECKPOINTS/(key+'.pt')
    assert not path.exists()
    payload={"meta":meta,"state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
    torch.save(payload,path)
    return {"file":path.name,"sha256":file_sha(path),"meta":meta,"file_bytes":path.stat().st_size}


def load_record(record):
    path=CHECKPOINTS/record['file']
    if file_sha(path)!=record['sha256']:raise RuntimeError('checkpoint checksum mismatch '+str(path))
    try: obj=torch.load(path,map_location='cpu',weights_only=True)
    except Exception:obj=torch.load(path,map_location='cpu',weights_only=False)
    assert obj['meta']==record['meta']
    meta=obj['meta']
    model=TinyAttention(meta['kind'],meta['seed'],meta['beta'])
    model.load_state_dict(obj['state_dict']);model.eval()
    return model


def train():
    torch.set_num_threads(1)
    CHECKPOINTS.mkdir(exist_ok=True,parents=True);RESULTS.mkdir(exist_ok=True,parents=True)
    if (RESULTS/'SEALED_TRAIN.json').exists() or list(CHECKPOINTS.glob('*.pt')):
        raise RuntimeError('Already has checkpoints. Never silently overwrite a sealed training experiment.')
    all_records=[]
    print('START HARD CE 72 preregistered models',flush=True)
    for world in WORLDS:
      for seed in SEEDS:
        tr=dataset(world,seed,'train',384);dev=dataset(world,seed,'dev',256)
        indices=choose_batch_order(seed,HARD_UPDATES,'hard')
        for kind in KINDS:
          for beta in (2.5,12.0):
            for loss_kind in ('standard_ce','fixed_ce'):
                model,meta=train_hard(world,seed,kind,beta,loss_kind,tr,dev,indices)
                all_records.append(save_model(model,meta))
                print('HARD',len(all_records),'of 72',world,seed,kind,beta,loss_kind,'devCE',round(meta['dev_ce_fp64'],5),'sat',round(meta['margin_gt_16p6'],3),flush=True)
    print('START DISTILL 48 exact matched-student models; NO Fresh/OOD generated',flush=True)
    hard_map={(r['meta']['world'],r['meta']['seed'],r['meta']['kind'],r['meta']['beta'],r['meta']['loss']):r for r in all_records}
    for world in WORLDS:
      for seed in SEEDS:
        tr=dataset(world,seed,'train',384)
        indices=choose_batch_order(seed,DISTILL_UPDATES,'distill')
        for kind in ('mirror','film'):
            teacher_rec=hard_map[(world,seed,kind,12.0,'fixed_ce')]
            teacher_state=load_record(teacher_rec).state_dict()
            for eps in (1e-8,1e-12):
                for loss_kind in ('standard_soft_ce','fixed_soft_ce'):
                    model,meta=train_distill(world,seed,kind,eps,loss_kind,tr,indices,teacher_state)
                    all_records.append(save_model(model,meta))
                    print('DISTILL',len(all_records)-72,'of 48',world,seed,kind,eps,loss_kind,'drift',round(meta['max_param_abs_displacement'],6),flush=True)
    assert len(all_records)==120
    manifest={"experiment":"MA-1177","train_complete":True,"fresh_or_ood_generated":False,"model_count":len(all_records),
      "prereg_sha256":file_sha(ROOT/'PROTOCOL.json'),
      "source_sha256":{name:file_sha(ROOT/name) for name in ('source/core.py','source/run.py','tests/test_numeric.py')},
      "environment":{"torch":torch.__version__,"python":sys.version,"platform":platform.platform(),"threads":torch.get_num_threads(),"cuda":torch.cuda.is_available()},
      "records":all_records}
    temp=RESULTS/'SEALED_TRAIN.json.tmp'
    temp.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    temp.replace(RESULTS/'SEALED_TRAIN.json')
    print('SEALED',len(all_records),'manifest sha256',file_sha(RESULTS/'SEALED_TRAIN.json'),flush=True)


def eval_data(model,teacher,dat,lane):
    logits=model_outputs(model,dat)
    if lane=='hard':
        return {"ce_fp64":fp64_cross_entropy(logits,dat['target']),
                "accuracy":float((logits.argmax(-1)==dat['target']).float().mean()),
                **margin_stats(logits,dat['target'])}
    ref=model_outputs(teacher,dat)
    return {"teacher_student_fp64_kl":fp64_distill_kl(logits,ref),
            "rms_logit_delta":float(torch.linalg.vector_norm((logits-ref).float()).item()/math.sqrt(logits.numel())),
            "top1_mismatch":float((logits.argmax(-1)!=ref.argmax(-1)).float().mean())}


def evaluate():
    torch.set_num_threads(1)
    sealpath=RESULTS/'SEALED_TRAIN.json'
    if not sealpath.exists():raise RuntimeError('no frozen training manifest')
    manifest=json.loads(sealpath.read_text())
    if not manifest.get('train_complete') or manifest.get('model_count')!=120 or manifest.get('fresh_or_ood_generated'):
        raise RuntimeError('not correctly sealed')
    for name,hashval in manifest['source_sha256'].items():
        if file_sha(ROOT/name)!=hashval:raise RuntimeError('source modified after checkpoint seal: '+name)
    for rec in manifest['records']:
        if file_sha(CHECKPOINTS/rec['file'])!=rec['sha256']:raise RuntimeError('checkpoint tampered '+rec['file'])
    if (RESULTS/'RESULTS_CORE.csv').exists():raise RuntimeError('fresh already evaluated')
    print('FRESH OPENED after verified seal',file_sha(sealpath),flush=True)
    rows=[]
    for record in manifest['records']:
        m=record['meta'];model=load_record(record)
        fresh=dataset(m['world'],m['seed'],'fresh',768)
        ood=dataset(m['world'],m['seed'],'ood',768)
        teacher=None
        if m['lane']=='distill':
            k=(m['world'],m['seed'],m['kind'],12.0,'fixed_ce')
            source=next(r for r in manifest['records'] if r['meta']['lane']=='hard' and (r['meta']['world'],r['meta']['seed'],r['meta']['kind'],r['meta']['beta'],r['meta']['loss'])==k)
            teacher=load_record(source)
        x=eval_data(model,teacher,fresh,m['lane']);y=eval_data(model,teacher,ood,m['lane'])
        rows.append({"lane":m['lane'],"world":m['world'],"seed":m['seed'],"kind":m['kind'],"beta":m['beta'],"loss":m['loss'],"eps":m.get('eps',''),"n_params":m['parameter_count'],"updates":m['steps'],"wall_s":m['wall_s'],
            "dev_ce_fp64":m.get('dev_ce_fp64',''),"train_ce_fp64":m.get('train_ce_fp64',''),"train_saturated_gt16p6":m.get('margin_gt_16p6',''),"train_max_margin":m.get('margin_max',''),"initial_equal_logit_grad_norm":m.get('initial_equal_logits_grad_norm',''),"max_param_abs_displacement":m.get('max_param_abs_displacement',''),"gradient_balance_max":m.get('gradient_balance_max',''),"worst_dev_ce_step_change":m.get('worst_dev_ce_step_change',''),"fresh_ce_fp64":x.get('ce_fp64',''),"fresh_top1":x.get('accuracy',''),"fresh_margin_gt16p6":x.get('margin_gt_16p6',''),"fresh_teacher_student_fp64_kl":x.get('teacher_student_fp64_kl',''),"fresh_rms_logit_delta":x.get('rms_logit_delta',''),"fresh_top1_mismatch":x.get('top1_mismatch',''),"ood_ce_fp64":y.get('ce_fp64',''),"ood_top1":y.get('accuracy',''),"ood_teacher_student_fp64_kl":y.get('teacher_student_fp64_kl',''),"ood_rms_logit_delta":y.get('rms_logit_delta',''),"checkpoint_file":record['file'],"sha256":record['sha256'],"checkpoint_bytes":record['file_bytes']})
    out=RESULTS/'RESULTS_CORE.csv'
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    complete={"id":"MA-1177","rows":len(rows),"seal_sha256":file_sha(sealpath),"evaluation_csv_sha256":file_sha(out),"checkpoint_replay_verified":120,"fresh_evaluated_after_seal":True}
    (RESULTS/'EVAL_COMPLETE.json').write_text(json.dumps(complete,indent=2)+'\n')
    print('EVAL COMPLETE',json.dumps(complete),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['train','evaluate']);args=parser.parse_args()
    train() if args.command=='train' else evaluate()
