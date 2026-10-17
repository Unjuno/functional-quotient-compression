"""MA-1164 reproducible world-first native -> no-label weight-compilation evaluation.

Two separated studies: dev K/R selection, then fresh paired worlds. Frozen protocol.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, os, platform, statistics, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from core import (N,NativeMLP,native_flat,logits,invert_orbit,seed_transform,
    CoeffDecoder,state_bytes,native_payload,best_svd_under_byte_cap,from_svd)

ROOT=Path(__file__).resolve().parent.parent
RESULTS=ROOT/'results'
DEV_WORLDS=(618410,618411)
FRESH_WORLDS=(618420,618421,618422)
DEV_SEEDS=(21,)
FRESH_SEEDS=(31,32)
DEV_MS=(64,256,512,768)
FRESH_MS=(512,768)
KS=(1,2,4,8)
RS=(1,2,4)
BASE_UPDATES=900
BATCH=128


def world_data(world):
    raw=load_digits()
    xx=torch.as_tensor(raw.images.reshape(-1,64).astype(np.float64)/16.0,dtype=torch.float64)
    yy=torch.as_tensor(raw.target,dtype=torch.long)
    ids=np.arange(len(yy)); y=yy.numpy()
    work,test=train_test_split(ids,test_size=.2,stratify=y,random_state=world)
    train,dev=train_test_split(work,test_size=.2,stratify=y[work],random_state=world+1000)
    return xx,yy,{'train':torch.as_tensor(train),'dev':torch.as_tensor(dev),'test':torch.as_tensor(test)}


def eval_model(model,xx,yy,split,part='dev'):
    idx=split[part]
    with torch.no_grad():
        pred=model(xx[idx]);true=yy[idx]
        return float((pred.argmax(-1)==true).double().mean()),float(F.cross_entropy(pred,true))


def native_train(seed,world,xx,yy,split):
    torch.manual_seed(1000+seed)
    g=torch.Generator().manual_seed(world*17+seed*103+37)
    net=NativeMLP()
    opt=torch.optim.AdamW(net.parameters(),lr=.008,weight_decay=0.)
    idx=split['train']
    t0=time.perf_counter()
    for update in range(BASE_UPDATES):
        take=idx[torch.randint(len(idx),(BATCH,),generator=g)]
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(net(xx[take]),yy[take])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
        opt.step()
    dev_acc,dev_nll=eval_model(net,xx,yy,split,'dev')
    if dev_acc<.90:raise RuntimeError(f'native viability failure world={world} seed={seed} dev_acc={dev_acc}')
    return net, {'native_train_s':time.perf_counter()-t0,'native_dev_accuracy':dev_acc,
                 'native_dev_nll':dev_nll}


def fit_teacher_weights(target,kind,m,k=0,r=0):
    """Teacher weights only: no label/logit tensors are accessed or passed."""
    with torch.no_grad():
        if kind=='mirror':inv=invert_orbit(target,k,r)
        elif kind=='plain':inv=seed_transform(target,True)
        elif kind=='raw':inv=target.clone()
        else:raise ValueError(kind)
        indices=torch.topk(inv.abs(),m).indices.sort().values
        model=CoeffDecoder(indices,kind,k,r)
        model.coords.copy_(inv[indices])
    opt=torch.optim.LBFGS([model.coords],lr=.8,max_iter=75,
                          history_size=15,line_search_fn='strong_wolfe',
                          tolerance_grad=1e-12,tolerance_change=1e-15)
    calls=[0];start=time.perf_counter()
    def closure():
        opt.zero_grad(set_to_none=True)
        error=model.weights()-target
        loss=error.square().mean();loss.backward()
        calls[0]+=1
        return loss
    opt.step(closure)
    with torch.no_grad():
        diff=model.weights()-target
        l2=float(torch.linalg.vector_norm(diff))
    return model,{'fit_seconds':time.perf_counter()-start,'fit_calls':calls[0],'weight_l2':l2}


def median_forward_ms(model,x):
    with torch.no_grad():
        for _ in range(3):model(x)
        values=[]
        for _ in range(11):
            start=time.perf_counter();model(x)
            values.append((time.perf_counter()-start)*1000)
    return float(statistics.median(values))


def add_row(rows,**kwargs):
    rows.append(kwargs)


def write_csv(rows,path):
    if path.exists():raise FileExistsError(path)
    if not rows:raise RuntimeError('Empty experiment: '+str(path))
    fields=list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def native_variants(target,xx,yy,split,part,world,seed,phase):
    rows=[]
    for dtype in (torch.float64,torch.float32,torch.float16):
        pay=native_payload(target,dtype)
        repr_target=torch.cat((pay['state']['l1_w'].flatten().double(),pay['state']['l1_b'].double(),
                               pay['state']['l2_w'].flatten().double(),pay['state']['l2_b'].double()))
        fn=lambda x, w=repr_target:logits(w,x)
        acc,nll=eval_model(fn,xx,yy,split,part)
        rows.append({'phase':phase,'world':world,'seed':seed,'kind':'native_'+str(dtype).split('.')[-1],
                     'm':N,'k':0,'rounds':0,'accuracy':acc,'nll':nll,
                     'weight_l2':float(torch.linalg.vector_norm(repr_target-target)),
                     'serialized_bytes':state_bytes(pay),'trainable_params':N,
                     'virtual_params':N,'infer_batch64_ms':median_forward_ms(fn,xx[split[part][:64]])})
    return rows


def model_to_row(model,target,xx,yy,split,part,phase,world,seed,k,r,m,fit):
    acc,nll=eval_model(model,xx,yy,split,part)
    bench=median_forward_ms(model,xx[split[part][:64]])
    with torch.no_grad():
        pred_err=logits(model.weights(),xx[split[part]])-logits(target,xx[split[part]])
        rms=float(pred_err.square().mean().sqrt())
    return {'phase':phase,'world':world,'seed':seed,'kind':model.kind,'m':m,'k':k if model.kind=='mirror' else 0,
            'rounds':r if model.kind=='mirror' else 0,'accuracy':acc,'nll':nll,
            'weight_l2':fit['weight_l2'],'logit_rmse':rms,
            'serialized_bytes':state_bytes(model.payload()),'trainable_params':m,'virtual_params':N,
            'fit_seconds':fit['fit_seconds'],'fit_calls':fit['fit_calls'],
            'infer_batch64_ms':bench,'mirror_blocks':k*r if model.kind=='mirror' else 0,
            'mlp_mac_proxy':64*16+16*10}


def rank_row(target,cap,xx,yy,split,part,phase,world,seed,m):
    svd=best_svd_under_byte_cap(target,cap)
    if svd is None:
        return {'phase':phase,'world':world,'seed':seed,'kind':'svd_none_under_cap','m':m,
                'accuracy':None,'serialized_bytes':0,'svd_rank':0}
    new=from_svd(svd)
    fn=lambda x:logits(new,x)
    acc,nll=eval_model(fn,xx,yy,split,part)
    return {'phase':phase,'world':world,'seed':seed,'kind':'svd_matched','m':m,
            'accuracy':acc,'nll':nll,'svd_rank':svd['rank'],'trainable_params':80*svd['rank']+186,
            'weight_l2':float(torch.linalg.vector_norm(new-target)),'serialized_bytes':state_bytes(svd),
            'infer_batch64_ms':median_forward_ms(fn,xx[split[part][:64]])}


def prepare_world(world,seed):
    xx,yy,split=world_data(world)
    net,diag=native_train(seed,world,xx,yy,split)
    target=native_flat(net)
    return xx,yy,split,target,diag


def run_development():
    path=RESULTS/'development_sweep.csv'
    if path.exists() or (RESULTS/'development_selection.json').exists():raise FileExistsError('development evidence already present')
    rows=[]
    for world in DEV_WORLDS:
        for seed in DEV_SEEDS:
            xx,yy,split,target,diag=prepare_world(world,seed)
            rows+=native_variants(target,xx,yy,split,'dev',world,seed,'development')
            print('DEVELOPMENT native',world,seed,diag,flush=True)
            for m in DEV_MS:
                for kind in ('raw','plain'):
                    mod,fit=fit_teacher_weights(target,kind,m)
                    rows.append(model_to_row(mod,target,xx,yy,split,'dev','development',world,seed,0,0,m,fit))
                for k in KS:
                    for r in RS:
                        mod,fit=fit_teacher_weights(target,'mirror',m,k,r)
                        row=model_to_row(mod,target,xx,yy,split,'dev','development',world,seed,k,r,m,fit)
                        rows.append(row)
                        print(f'DEV world={world} m={m} K={k} R={r} acc={row["accuracy"]:.4f} bytes={row["serialized_bytes"]} fit={row["fit_seconds"]:.3f}s',flush=True)
            write_csv(rows,path) if False else None
    write_csv(rows,path)
    # Freeze one K/R per m on development only, cannot be retuned after seeing fresh labels.
    selected={}
    for m in FRESH_MS:
        candidates=[]
        for k in KS:
            for r in RS:
                group=[x for x in rows if x['kind']=='mirror' and x['m']==m and x['k']==k and x['rounds']==r]
                assert len(group)==len(DEV_WORLDS)*len(DEV_SEEDS)
                candidates.append({'k':k,'r':r,'mean_dev_accuracy':sum(x['accuracy'] for x in group)/len(group),
                                   'mean_dev_nll':sum(x['nll'] for x in group)/len(group)})
        best=sorted(candidates,key=lambda a:(-a['mean_dev_accuracy'],a['mean_dev_nll'],a['k']*a['r']))[0]
        selected[str(m)]=best
    result={'selection_source':str(path),'selected':selected,'seed_worlds':list(DEV_WORLDS),
            'note':'No fresh world has been opened in this process. Do not alter after opening fresh.'}
    (RESULTS/'development_selection.json').write_text(json.dumps(result,indent=2))
    print('DEV SELECTED',selected,flush=True)


def fine_tune(model,world,seed,xx,yy,split):
    before={n:p.clone().detach() for n,p in model.state_dict().items()}
    g=torch.Generator().manual_seed(world*43+seed*101+884)
    idx=split['train'];opt=torch.optim.AdamW([model.coords],lr=.001,weight_decay=0.)
    start=time.perf_counter()
    for step in range(200):
        samples=idx[torch.randint(len(idx),(BATCH,),generator=g)]
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(xx[samples]),yy[samples])
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
    return {'fine_tune_s':time.perf_counter()-start,'fine_tune_updates':200}


def run_fresh():
    path=RESULTS/'fresh_results.csv'
    if path.exists():raise FileExistsError(path)
    selection_path=RESULTS/'development_selection.json'
    if not selection_path.exists():raise RuntimeError('Must finish development before fresh')
    selection=json.loads(selection_path.read_text())['selected']
    assert set(selection.keys())=={'512','768'}
    rows=[]
    for world in FRESH_WORLDS:
        for seed in FRESH_SEEDS:
            xx,yy,split,target,diag=prepare_world(world,seed)
            rows+=native_variants(target,xx,yy,split,'test',world,seed,'fresh')
            print(f'FRESH baseline world={world} seed={seed} dev_acc={diag["native_dev_accuracy"]:.4f}',flush=True)
            for m in FRESH_MS:
                selected=(selection[str(m)]['k'],selection[str(m)]['r'])
                variants=sorted(set([selected,(1,1)]))
                # Strong plain, raw and byte-matched SVD controls (each refit from teacher weights only).
                for kind in ('plain','raw'):
                    mod,fit=fit_teacher_weights(target,kind,m)
                    row=model_to_row(mod,target,xx,yy,split,'test','fresh',world,seed,0,0,m,fit)
                    rows.append(row)
                    if kind=='plain':
                        finfo=fine_tune(mod,world,seed,xx,yy,split)
                        acc,nll=eval_model(mod,xx,yy,split,'test')
                        rows.append({**row,'kind':'plain_finetune','accuracy':acc,'nll':nll,**finfo})
                for k,r in variants:
                    mod,fit=fit_teacher_weights(target,'mirror',m,k,r)
                    row=model_to_row(mod,target,xx,yy,split,'test','fresh',world,seed,k,r,m,fit)
                    rows.append(row)
                    if (k,r)==selected:
                        svd=rank_row(target,row['serialized_bytes'],xx,yy,split,'test','fresh',world,seed,m)
                        rows.append(svd)
                        # Explicit label-assisted adaptation: separate from mechanical compilation.
                        finfo=fine_tune(mod,world,seed,xx,yy,split)
                        acc,nll=eval_model(mod,xx,yy,split,'test')
                        rows.append({**row,'kind':'mirror_finetune','accuracy':acc,'nll':nll,**finfo})
                print(f'FRESH world={world} seed={seed} m={m} selected={selected} full_rows={len(rows)}',flush=True)
            # partial file written only once per world/seed, so interrupted work is explicitly incomplete
            temp=RESULTS/'fresh_results.partial.csv'
            if temp.exists():temp.unlink()
            write_csv(rows,temp)
    write_csv(rows,path)
    temp=RESULTS/'fresh_results.partial.csv'
    if temp.exists():temp.unlink()
    (RESULTS/'runtime_environment.json').write_text(json.dumps({
        'platform':platform.platform(),'torch':torch.__version__, 'cpu_name':open('/proc/cpuinfo').read().split('model name')[1].split(':',1)[1].split('\n')[0].strip(),
        'threads':torch.get_num_threads(),'cpu_visible':os.cpu_count(),'cuda':torch.cuda.is_available(),
        'float64_inference':True,'float16_models_replayed_in_float64':True},indent=2))
    print('FRESH DONE rows',len(rows),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('dev','fresh'),required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    RESULTS.mkdir(parents=True,exist_ok=True)
    if args.phase=='dev':run_development()
    else:run_fresh()

if __name__=='__main__':main()
