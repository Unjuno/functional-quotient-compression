import argparse, csv, json, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
D,K,Q=32,4,64
TASKS=(8,32); RESIDUALS=(0.,.05,.2); INTERVAL=8
SCHEMA=np.frombuffer(b"FAST-SLOW-MIRROR-STATE-V1",dtype=np.uint8)

def world(seed,n,rho):
    rng=np.random.default_rng(seed*1000+n)
    flat,_=np.linalg.qr(rng.normal(size=(D*D,K)))
    basis=flat.T.reshape(K,D,D)
    coeff=rng.normal(size=(n,K))
    target0=coeff@flat.T
    residual=rng.normal(size=(n,D*D)); residual-= (residual@flat)@flat.T
    scale=np.sqrt(np.mean(target0**2,axis=1,keepdims=True))
    residual*=scale/np.maximum(np.sqrt(np.mean(residual**2,axis=1,keepdims=True)),1e-12)
    targets=(target0+rho*residual).reshape(n,D,D).astype(np.float32)
    x=rng.normal(size=(n,Q,D)).astype(np.float32)
    return basis.astype(np.float32),targets,x

def nrmse(pred,target,x):
    y=np.einsum('nqd,ndh->nqh',x,target); p=np.einsum('nqd,ndh->nqh',x,pred)
    e=np.mean((p-y)**2,axis=(1,2)); v=np.mean(y*y,axis=(1,2))
    return np.sqrt(e/np.maximum(v,1e-12))

def compress(payload,name):
    path=ROOT/'source'/name; start=time.perf_counter(); np.savez_compressed(path,**payload)
    return path.stat().st_size,time.perf_counter()-start

def run_condition(seed,n,rho):
    _,targets,x=world(seed,n,rho); rows=[]; previous={}
    for end in range(INTERVAL,n+1,INTERVAL):
        seen=targets[:end]; flat=seen.reshape(end,-1)
        _,_,vh=np.linalg.svd(flat,full_matrices=False); pca=vh[:K].astype(np.float32)
        codes=(flat@pca.T).astype(np.float32); decoded=(codes@pca).reshape(end,D,D)
        ids=np.arange(end,dtype=np.uint16)
        payloads={
          'mirror':{'schema':SCHEMA,'base':np.zeros((D,D),np.float32),'task_ids':ids,'basis':pca.reshape(K,D,D),'view_codes':codes},
          'native':{'schema':SCHEMA.copy(),'base':np.zeros((D,D),np.float32),'task_ids':ids.copy(),'basis':pca.reshape(K,D,D).copy(),'view_codes':codes.copy()},
          'independent':{'schema':SCHEMA.copy(),'task_ids':ids.copy(),'task_matrices':seen.copy()},
        }
        # Ordinary private rank-2 adapter bank is an inexpensive per-task control.
        factors=[]
        for m in seen:
            u,s,vh_m=np.linalg.svd(m,full_matrices=False); factors.append((u[:,:2]*s[:2])@vh_m[:2])
        factors=np.asarray(factors,dtype=np.float32)
        payloads['private_rank2']={'schema':SCHEMA.copy(),'task_ids':ids.copy(),'left':np.stack([np.linalg.svd(m,full_matrices=False)[0][:,:2] for m in seen]).astype(np.float32),'right_scaled':np.stack([np.linalg.svd(m,full_matrices=False)[1][:2,None]*np.linalg.svd(m,full_matrices=False)[2][:2] for m in seen]).astype(np.float32)}
        predictions={'mirror':decoded,'native':decoded.copy(),'independent':seen,'private_rank2':factors}
        for method,payload in payloads.items():
            fname=f'{seed}_n{n}_rho{rho:g}_t{end}_{method}.npz'
            size,wall=compress(payload,fname)
            errs=nrmse(predictions[method],seen,x[:end])
            old=previous.get(method)
            drift=float(np.mean(np.sqrt(np.mean((predictions[method][:len(old)]-old)**2,axis=(1,2))))) if old is not None else 0.
            rows.append({'seed':seed,'task_count':n,'residual_rms':rho,'tasks_written':end,'method':method,'serialized_bytes':size,'mean_task_nrmse':float(np.mean(errs)),'worst_task_nrmse':float(np.max(errs)),'prior_task_transition_drift':drift,'query_examples':end*Q,'optimizer_updates':0,'update_compute_proxy':int(end*K*D*D),'decode_compute_proxy':int(end*Q*2*D*D),'wall_time_s':wall})
        # Mirror/native states and reconstructed functions are expected exact aliases by construction.
        assert np.array_equal(payloads['mirror']['basis'],payloads['native']['basis'])
        assert np.array_equal(payloads['mirror']['view_codes'],payloads['native']['view_codes'])
        previous=predictions
    return rows

def run(seeds):
    rows=[r for seed in seeds for n in TASKS for rho in RESIDUALS for r in run_condition(seed,n,rho)]
    (ROOT/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
        for r in rows:w.writerow([f"n{r['task_count']}_rho{r['residual_rms']:g}_t{r['tasks_written']}",r['seed'],r['method'],r['serialized_bytes'],r['query_examples'],r['optimizer_updates'],r['update_compute_proxy']+r['decode_compute_proxy'],r['wall_time_s'],'mean_task_nrmse',r['mean_task_nrmse'],'worst_task_nrmse',r['worst_task_nrmse'],'development; fresh sealed'])
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);args=ap.parse_args();print(json.dumps(run(args.seeds),indent=2))
