#!/usr/bin/env python3
"""Frozen real-digit shifted-task Mirror-code research intake; NOT an MA status gate.
Protocol: experiments/mirror_applications/research_intake/natural_digit_function_20261008/PROTOCOL.md
"""
import argparse
import csv
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, rotate, shift
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SOURCE = ['rot_p20', 'rot_m20', 'shift_xp1', 'contrast_07', 'blur_08']
TARGET = ['rot_p33', 'rot_m33', 'shift_yp1', 'blur_12']
MODES = ['base', 'scalar', 'diagonal', 'mirror', 'dense', 'lora4']
RANK, K, BATCH = 4, 6, 256
PRETRAIN_STEPS, ADAPT_STEPS = 500, 220


def corrupt(flat, kind):
    ims = flat.reshape(-1, 8, 8)
    if kind.startswith('rot_'):
        angle = int(kind.split('p')[-1]) if '_p' in kind else -int(kind.split('m')[-1])
        v = np.stack([rotate(i, angle, reshape=False, order=1, mode='constant', cval=0) for i in ims])
    elif kind == 'shift_xp1':
        v = np.stack([shift(i, (0, 1), order=1, mode='constant', cval=0) for i in ims])
    elif kind == 'shift_yp1':
        v = np.stack([shift(i, (1, 0), order=1, mode='constant', cval=0) for i in ims])
    elif kind == 'blur_08':
        v = np.stack([gaussian_filter(i, sigma=0.8) for i in ims])
    elif kind == 'blur_12':
        v = np.stack([gaussian_filter(i, sigma=1.2) for i in ims])
    elif kind == 'contrast_07':
        v = ims * 0.7
    else:
        raise ValueError(kind)
    return torch.from_numpy(np.clip(v.reshape(-1, 64), 0, 1).astype(np.float32))


def build_base(seed, xx, yy):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU(),
                                torch.nn.Linear(64, 64), torch.nn.ReLU(),
                                torch.nn.Linear(64, 10))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    gen = torch.Generator().manual_seed(seed * 31 + 1)
    t0 = time.perf_counter()
    for _ in range(PRETRAIN_STEPS):
        idx = torch.randint(len(xx), (BATCH,), generator=gen)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xx[idx]), yy[idx]); loss.backward(); opt.step()
    wall = time.perf_counter() - t0
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model, wall


def forward(model, x, delta):
    x = F.relu(model[0](x))
    x = F.relu(F.linear(x, model[2].weight + delta, model[2].bias))
    return model[4](x)


def eval_metric(model, x, y, delta):
    with torch.no_grad():
        pred = forward(model, x, delta)
        return {'loss':float(F.cross_entropy(pred,y)),
                'accuracy':float((pred.argmax(dim=-1)==y).float().mean())}


def givens(k, i, j, a):
    eye = torch.eye(k)
    ei, ej = eye[i], eye[j]
    ca, sa = torch.cos(a), torch.sin(a)
    return eye + (ca - 1) * (torch.outer(ei,ei)+torch.outer(ej,ej)) + sa * (torch.outer(ej,ei)-torch.outer(ei,ej))


def code_delta(mode, params, U=None, V=None):
    if mode == 'base': return torch.zeros((64,64))
    if mode == 'lora4': return params['B'] @ params['A']
    if mode == 'scalar': return params['c'][0] * torch.outer(U[:,0], V[:,0])
    if mode == 'diagonal': return (U * params['c'][None,:]) @ V.T
    if mode == 'dense': return U @ params['c'].reshape(K,K) @ V.T
    if mode == 'mirror':
        L=givens(K,0,1,params['ang'][0]); Rt=givens(K,2,3,params['ang'][1])
        return U @ L @ torch.diag(params['c']) @ Rt.T @ V.T
    raise ValueError(mode)


def fit_delta(model, x, y, mode, seed, task_index, U=None, V=None, initial_core=None):
    torch.manual_seed(seed*10000+task_index+17)
    if mode=='lora4':
        params={'A':torch.nn.Parameter(torch.randn(RANK,64)*0.05),
                'B':torch.nn.Parameter(torch.zeros(64,RANK))}
    elif mode in ('scalar','diagonal','mirror','dense'):
        if mode=='scalar': initial = torch.zeros(1)
        elif mode=='dense': initial = initial_core.clone().reshape(-1)
        else: initial = torch.diag(initial_core).clone()
        params={'c':torch.nn.Parameter(initial)}
        if mode=='mirror': params['ang']=torch.nn.Parameter(torch.zeros(2))
    else: raise ValueError(mode)
    opt=torch.optim.Adam(list(params.values()),lr=0.025 if mode=='lora4' else 0.06)
    gen=torch.Generator().manual_seed(seed * 211 + task_index * 13 + 8)
    batch_indices=torch.randint(len(x),(ADAPT_STEPS,BATCH),generator=gen)
    start=time.perf_counter()
    for idx in batch_indices:
        opt.zero_grad(set_to_none=True)
        delta=code_delta(mode,params,U,V)
        loss=F.cross_entropy(forward(model,x[idx],delta),y[idx])
        loss.backward();opt.step()
    wall=time.perf_counter()-start
    return {k:v.detach().clone() for k,v in params.items()},wall


def source_bases(deltas):
    DD=torch.stack(deltas)
    left=torch.einsum('nij,nkj->ik',DD,DD)
    right=torch.einsum('nji,njk->ik',DD,DD)
    U=torch.linalg.eigh(left).eigenvectors[:,-K:].flip(1)
    V=torch.linalg.eigh(right).eigenvectors[:,-K:].flip(1)
    C=U.T @ DD.mean(0) @ V
    return U.contiguous(),V.contiguous(),C.contiguous()


def full_state(mode,U,V,pmap):
    if mode=='base': return {}
    if mode=='lora4':
        return {f'{name}_{key}': val.numpy() for name,p in pmap.items() for key,val in p.items()}
    if mode=='scalar':
        out={'u0':U[:,0].numpy(),'v0':V[:,0].numpy()}
    else:
        out={'U':U.numpy(),'V':V.numpy()}
    for name,p in pmap.items():
        out[f'{name}_m']=torch.cat([p['c'],p['ang']]).numpy() if mode=='mirror' else p['c'].numpy()
    return out


def serialized_nbytes(arrays):
    h=io.BytesIO()
    np.savez(h,**arrays)
    assert len(h.getvalue()) > 20
    with np.load(io.BytesIO(h.getvalue())) as check:
        assert set(check.files)==set(arrays)
        for name,v in arrays.items(): assert np.array_equal(check[name],v)
    return len(h.getvalue())


def benchmark_forward(model,x,mode,params,U,V):
    with torch.no_grad():
        for _ in range(10): _=forward(model,x,code_delta(mode,params,U,V))
        t=time.perf_counter()
        for _ in range(50): _=forward(model,x,code_delta(mode,params,U,V))
        return (time.perf_counter()-t)/50


def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    data=load_digits(); X=(data.data.astype(np.float32)/16.0); Y=data.target.astype(np.int64)
    indices=np.arange(len(X))
    tr,hold=train_test_split(indices,test_size=0.4,stratify=Y,random_state=seed)
    dev,test=train_test_split(hold,test_size=0.5,stratify=Y[hold],random_state=seed+100)
    labels=torch.from_numpy(Y)
    clean=torch.from_numpy(X)
    model,base_wall=build_base(seed,clean[tr],labels[tr])
    with torch.no_grad(): base_clean=eval_metric(model,clean[test],labels[test],torch.zeros(64,64))
    xforms={name:corrupt(X,name) for name in SOURCE+TARGET}
    sources=[]
    for tid,task in enumerate(SOURCE):
        pars,_=fit_delta(model,xforms[task][tr],labels[tr],'lora4',seed,101+tid)
        sources.append(code_delta('lora4',pars).detach())
    U,V,C0=source_bases(sources)
    fitted={mode:{} for mode in MODES if mode!='base'}
    rows=[]
    for tid,task in enumerate(TARGET):
        xtr=xforms[task][tr];xte=xforms[task][test];xdev=xforms[task][dev]
        per_task={}
        per_task['base']=(None,0.0)
        # Fit identical deterministic minibatch sequences for modes at the same task index.
        for mode in MODES:
            if mode=='base': continue
            params,wall=fit_delta(model,xtr,labels[tr],mode,seed,201+tid,U,V,C0)
            per_task[mode]=(params,wall);fitted[mode][task]=params
        target_lora=code_delta('lora4',per_task['lora4'][0]).detach()
        with torch.no_grad(): reference=forward(model,xte,target_lora)
        for mode in MODES:
            params,wall=per_task[mode]
            D=code_delta(mode,params,U,V).detach()
            scores=eval_metric(model,xte,labels[test],D)
            devscores=eval_metric(model,xdev,labels[dev],D)
            with torch.no_grad():
                logits=forward(model,xte,D)
                y_err=float(torch.sqrt(torch.mean((logits-reference)**2)))
                w_err=float((torch.linalg.norm(D-target_lora)/(torch.linalg.norm(target_lora)+1e-12)))
            infer_ms=benchmark_forward(model,xte,mode,params,U,V)*1000
            rows.append({'seed':seed,'task':task,'mode':mode,**scores,
                         'dev_loss':devscores['loss'],'dev_accuracy':devscores['accuracy'],
                         'train_wall_s':wall,'forward_ms':infer_ms,
                         'relative_weight_error_vs_target_LoRA':w_err,'logit_rmse_vs_target_LoRA':y_err,
                         'train_examples':len(tr),'dev_examples':len(dev),'test_examples':len(test),
                         'task_updates':0 if mode=='base' else ADAPT_STEPS})
    sizes={mode:serialized_nbytes(full_state(mode,U,V,fitted.get(mode,{}))) for mode in MODES}
    for row in rows:row['four_task_bank_npz_bytes']=sizes[row['mode']]
    return rows,{'seed':seed,'base_clean':base_clean,'base_pretrain_wall_s':base_wall,
                 'source_tasks':SOURCE,'audit_tasks':TARGET,'shared_rank':K,'source_lora_rank':RANK,
                 'source_shared_basis_spectrum_left':torch.linalg.svdvals(torch.cat(sources,dim=1)).tolist(),
                 'bank_npz_bytes':sizes}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--seeds',default='41,42')
    args=ap.parse_args();torch.set_num_threads(1)
    seeds=[int(z) for z in args.seeds.split(',')]
    if seeds!=[41,42]: raise ValueError('Frozen protocol requires both seeds 41 and 42.')
    rows=[];meta=[]
    for seed in seeds:
        rr,mm=run_seed(seed);rows.extend(rr);meta.append(mm)
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    path=dest/'RESULTS_CORE.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'protocol':'frozen preregistered real digits shift pilot','statuses_unchanged':True,
            'seeds':seeds,'tasks':TARGET,'method_means':{},'per_seed_structured_vs_diagonal':{},
            'metadata':meta,'row_count':len(rows),'source_dataset':'scikit-learn load_digits'}
    for mode in MODES:
        cur=[x for x in rows if x['mode']==mode]
        result['method_means'][mode]={k:float(np.mean([r[k] for r in cur])) for k in ['accuracy','loss','dev_loss','train_wall_s','forward_ms','four_task_bank_npz_bytes']}
    for seed in seeds:
        better=[]
        for task in TARGET:
            m=next(r for r in rows if r['seed']==seed and r['task']==task and r['mode']=='mirror')
            d=next(r for r in rows if r['seed']==seed and r['task']==task and r['mode']=='diagonal')
            better.append({'task':task,'mirror_accuracy_minus_diagonal':m['accuracy']-d['accuracy'],
                           'mirror_loss_minus_diagonal':m['loss']-d['loss'],
                           'mirror_forward_ms_over_diagonal':m['forward_ms']/d['forward_ms']})
        result['per_seed_structured_vs_diagonal'][str(seed)]=better
    with (dest/'SUMMARY.json').open('w') as f:json.dump(result,f,indent=2)
    print(json.dumps({'row_count':len(rows),'means':result['method_means'],
                      'comparisons':result['per_seed_structured_vs_diagonal']},indent=2))

if __name__=='__main__':main()