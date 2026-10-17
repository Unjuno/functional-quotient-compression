"""Frozen native training -> direct label-free weight compilation -> fresh inference.

Use: cd experiment-root; OMP_NUM_THREADS=1 ... python source/run.py --phase baseline
then --phase compile, then --phase direct. No test labels are observed in fit.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time,os,io,platform,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from model import N, K, R, NativeMLP, MirrorMLP, compile_exact, weight_to_flat, seed_transform, invert_nonlinear_orbit, payload_bytes

ROOT=Path(__file__).resolve().parent.parent
RESULTS=ROOT/'results'; RESULTS.mkdir(parents=True,exist_ok=True)
BATCH=128
TRAIN_STEPS=900
DIRECT_STEPS=450
SEEDS=(11,12,13)
MS=(1,16,64,256,N)


def dataset():
    d=load_digits()
    x=d.images.reshape(-1,64).astype(np.float64)/16.0
    y=d.target
    ids=np.arange(len(y))
    ids_work,ids_test=train_test_split(ids,test_size=.20,stratify=y,random_state=618201)
    ids_train,ids_dev=train_test_split(ids_work,test_size=.20,stratify=y[ids_work],random_state=618200)
    xs=torch.tensor(x,dtype=torch.float64)
    ys=torch.tensor(y,dtype=torch.int64)
    indices={'train':ids_train.tolist(),'dev':ids_dev.tolist(),'test':ids_test.tolist()}
    return xs,ys,indices


def evaluate(model,xx,yy,indices,split):
    model.eval()
    idx=torch.tensor(indices[split],dtype=torch.int64)
    with torch.no_grad():
        out=model(xx[idx]);loss=float(F.cross_entropy(out,yy[idx]))
        accuracy=float((out.argmax(-1)==yy[idx]).double().mean())
    return accuracy,loss


def baseline(seed,xx,yy,indices):
    torch.manual_seed(1000+seed)
    rng=torch.Generator().manual_seed(56789+seed)
    model=NativeMLP()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=0.)
    idx=torch.tensor(indices['train'],dtype=torch.int64)
    started=time.perf_counter()
    for step in range(TRAIN_STEPS):
        take=idx[torch.randint(len(idx),(BATCH,),generator=rng)]
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(xx[take]),yy[take]);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
    seconds=time.perf_counter()-started
    dev, devloss=evaluate(model,xx,yy,indices,'dev')
    return model,{'train_seconds':seconds,'dev_accuracy':dev,'dev_nll':devloss}


def fit_weights(model,target,max_iter=150):
    """Optimize ONLY training-network's fitted weights; no images or logits."""
    if model.m==N and model.nonlinear:
        with torch.no_grad():model.coords.copy_(invert_nonlinear_orbit(target))
        return {'method':'constructive_inverse','optimizer_iterations':0,'fit_seconds':0.}
    with torch.no_grad():
        transformed=invert_nonlinear_orbit(target) if model.nonlinear else seed_transform(target,inverse=True)
        model.coords.copy_(transformed[:model.m])
    opt=torch.optim.LBFGS([model.coords],lr=.8,max_iter=max_iter,
                         history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-12,
                         tolerance_change=1e-16)
    count=[0];t0=time.perf_counter()
    def closure():
        opt.zero_grad(set_to_none=True)
        residual=model.weights()-target
        l=(residual*residual).mean()
        l.backward();count[0]+=1
        return l
    opt.step(closure)
    return {'method':'LBFGS_weight_fit','optimizer_iterations':count[0],'fit_seconds':time.perf_counter()-t0}


def infer_median_ms(model,xx,indices):
    idx=torch.tensor(indices['test'][:64],dtype=torch.long)
    x=xx[idx]
    import statistics
    with torch.no_grad():
        for _ in range(4):model(x)
        durations=[]
        for _ in range(21):
            t0=time.perf_counter();model(x);durations.append((time.perf_counter()-t0)*1000)
    return float(statistics.median(durations))


def write_csv(path,rows):
    if path.exists():raise FileExistsError(path)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)


def run_baseline(xx,yy,indices):
    rows=[]
    for seed in SEEDS:
        model,metrics=baseline(seed,xx,yy,indices)
        if metrics['dev_accuracy']<.9:
            raise RuntimeError('Native model failed viability gate: '+str((seed,metrics)))
        path=RESULTS/f'native_{seed}.pt'
        torch.save({'seed':seed,'state':model.state_dict()},path)
        row={'seed':seed,'native_params':N,'serialized_bytes':payload_bytes('native',model),**metrics}
        rows.append(row)
        print('NATIVE',seed,row,flush=True)
    write_csv(RESULTS/'baseline_development.csv',rows)


def run_compile(xx,yy,indices):
    rows=[]
    source_bytes=Path(__file__).with_name('model.py').stat().st_size
    for seed in SEEDS:
        native=NativeMLP();native.load_state_dict(torch.load(RESULTS/f'native_{seed}.pt',weights_only=True)['state'])
        native.eval();target=weight_to_flat(native)
        ta,tl=evaluate(native,xx,yy,indices,'test')
        raw=RESULTS/f'native_{seed}.pt'
        print('TEACHER TEST',seed,ta,flush=True)
        for kind in ('plain','mirror'):
            for m in MS:
                model=MirrorMLP(m,nonlinear=(kind=='mirror'))
                fit=fit_weights(model,target)
                model.eval()
                ca,cl=evaluate(model,xx,yy,indices,'test')
                with torch.no_grad():
                    delta=model.weights()-target
                    l2=float(torch.linalg.vector_norm(delta))
                    maxweight=float(delta.abs().max())
                    testidx=torch.tensor(indices['test'],dtype=torch.long)
                    logits_delta=model(xx[testidx])-native(xx[testidx])
                    logits_max=float(logits_delta.abs().max())
                    logits_rmse=float(torch.sqrt(torch.mean(logits_delta.square())))
                row={'seed':seed,'kind':kind,'m':m,'K':K if kind=='mirror' else 0,
                     'rounds':R if kind=='mirror' else 0,'physical_trainable_scalars':m,
                     'virtual_weight_scalars':N,'teacher_test_acc':ta,'compiled_test_acc':ca,
                     'teacher_test_nll':tl,'compiled_test_nll':cl,
                     'test_accuracy_delta':ca-ta,'weight_l2':l2,'weight_max_abs':maxweight,
                     'test_logit_max_abs':logits_max,'test_logit_rmse':logits_rmse,
                     'payload_bytes':payload_bytes('mirror_'+kind,model),
                     'teacher_payload_bytes':payload_bytes('native',native),
                     'mirror_code_source_bytes':source_bytes,
                     'mirror_payload_plus_source_bytes':payload_bytes('mirror_'+kind,model)+source_bytes,
                     'inference_batch64_ms':infer_median_ms(model,xx,indices),
                     'teacher_inference_batch64_ms':infer_median_ms(native,xx,indices),**fit}
                rows.append(row)
                print('COMPILED',seed,kind,m,'acc',round(ca,5),'weight_L2',round(l2,5),
                      'byte',row['payload_bytes'],'fit_s',round(fit['fit_seconds'],2),flush=True)
                if m==N and kind=='mirror':
                    assert maxweight<1e-9 and logits_max<1e-8
                    torch.save({'m':m,'kind':kind,'state':model.state_dict()},RESULTS/f'compiled_exact_{seed}.pt')
    write_csv(RESULTS/'compiled_fresh.csv',rows)


def run_direct(xx,yy,indices):
    rows=[]
    idx=torch.tensor(indices['train'],dtype=torch.int64)
    for seed in SEEDS:
        torch.manual_seed(1000+seed)
        reference=NativeMLP()
        reference_weights=weight_to_flat(reference)
        for m in (16,256,N):
            model=MirrorMLP(m,True)
            with torch.no_grad():
                model.coords.copy_(invert_nonlinear_orbit(reference_weights)[:m])
            opt=torch.optim.AdamW([model.coords],lr=.008,weight_decay=0.)
            rng=torch.Generator().manual_seed(56789+seed)
            t0=time.perf_counter()
            for step in range(DIRECT_STEPS):
                take=idx[torch.randint(len(idx),(BATCH,),generator=rng)]
                opt.zero_grad(set_to_none=True)
                l=F.cross_entropy(model(xx[take]),yy[take]);l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()
            seconds=time.perf_counter()-t0
            va,vl=evaluate(model,xx,yy,indices,'dev')
            ta,tl=evaluate(model,xx,yy,indices,'test')
            row={'seed':seed,'m':m,'updates':DIRECT_STEPS,'batch':BATCH,'dev_accuracy':va,'test_accuracy':ta,
                 'dev_nll':vl,'test_nll':tl,'train_seconds':seconds,
                 'serialized_bytes':payload_bytes('mirror',model),
                 'inference_batch64_ms':infer_median_ms(model,xx,indices)}
            rows.append(row)
            print('DIRECT',seed,m,'dev',round(va,5),'test',round(ta,5),'sec',round(seconds,2),flush=True)
    write_csv(RESULTS/'direct_training_fresh.csv',rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('baseline','compile','direct'),required=True)
    a=parser.parse_args()
    torch.set_num_threads(1)
    xx,yy,indices=dataset()
    print('ENV',a.phase,len(indices['train']),len(indices['dev']),len(indices['test']),
          'torch',torch.__version__,'threads',torch.get_num_threads(),flush=True)
    if a.phase=='baseline':run_baseline(xx,yy,indices)
    elif a.phase=='compile':run_compile(xx,yy,indices)
    else:run_direct(xx,yy,indices)

if __name__=='__main__':main()
