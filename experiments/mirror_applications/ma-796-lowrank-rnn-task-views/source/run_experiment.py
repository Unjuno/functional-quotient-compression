"""Synthetic shared linear recurrent dynamics with task-specific coordinates."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[1]; SEED=79601; D=8; U=4; T=32; N=256
TASKS=np.array([(a,b) for a in (-1.,0.,1.) for b in (-1.,0.,1.)],np.float32)
HELD=np.array([0,8])  # (-1,-1), (1,1)
TRAIN=np.array([i for i in range(9) if i not in HELD])

def pack(**x):
    b=io.BytesIO();np.savez_compressed(b,**x);return b.getvalue()

def sim(A,G,C,h0,us):
    n=len(h0); hs=np.empty((n,T+1,D),np.float32);ys=np.empty((n,T+1,D),np.float32);hs[:,0]=h0;ys[:,0]=h0@C.T
    for t in range(T): hs[:,t+1]=hs[:,t]@A.T+us[:,t]@G.T;ys[:,t+1]=hs[:,t+1]@C.T
    return hs,ys

def nrmse(p,y): return float(np.sqrt(np.mean((p-y)**2)/(np.mean(y*y)+1e-12)))

def run():
    rng=np.random.default_rng(SEED)
    A0=(.43*np.eye(D)+rng.normal(0,.018,(D,D))).astype(np.float32)
    B1=rng.normal(0,.018,(D,D)).astype(np.float32);B2=rng.normal(0,.018,(D,D)).astype(np.float32)
    teacher=np.stack([A0+a*B1+b*B2 for a,b in TASKS])
    G=rng.normal(0,.12,(D,U)).astype(np.float32);C=rng.normal(0,.3,(D,D)).astype(np.float32)
    rho=np.array([max(abs(np.linalg.eigvals(A))) for A in teacher]);
    train_h=[];train_u=[];train_next=[]
    # Generate separate training sequences only for the seven observed task IDs.
    for k in TRAIN:
        h0=rng.normal(0,.5,(N,D)).astype(np.float32);us=rng.normal(0,.3,(N,T,U)).astype(np.float32)
        h,_=sim(teacher[k],G,C,h0,us);train_h.append(h[:,:-1].reshape(-1,D));train_u.append(us.reshape(-1,U));train_next.append(h[:,1:].reshape(-1,D))
    fitted=[]
    for j,k in enumerate(TRAIN):
        x=np.concatenate([train_h[j],train_u[j]],axis=1)
        beta=np.linalg.lstsq(x,train_next[j],rcond=None)[0]
        fitted.append(beta[:D].T.astype(np.float32))
    fitted=np.stack(fitted)
    # Fit recurrent core and two shared dynamical components from observed task matrices only.
    design=np.concatenate([np.ones((len(TRAIN),1),np.float32),TASKS[TRAIN]],axis=1)
    coef=np.linalg.lstsq(design,fitted.reshape(len(TRAIN),-1),rcond=None)[0].astype(np.float32)
    core=coef[0].reshape(D,D);bases=coef[1:].reshape(2,D,D)
    task_codes=TASKS.copy()
    pred=np.stack([core+a*bases[0]+b*bases[1] for a,b in task_codes])
    shared=np.broadcast_to(core,(9,D,D)).copy()
    # Heldout test trajectories for all tasks use independent inputs/initial states.
    h0=rng.normal(0,.5,(9,N,D)).astype(np.float32);us=rng.normal(0,.3,(9,N,T,U)).astype(np.float32)
    target=[]
    for k in range(9): target.append(sim(teacher[k],G,C,h0[k],us[k])[1])
    target=np.stack(target)
    methods=[]
    payloads=[
      ('independent_full_task_matrices',teacher,pack(recurrent=teacher,input_matrix=G,readout=C,task_ids=np.arange(9,dtype=np.int16)),0),
      ('shared_core_no_task_code',shared,pack(recurrent_core=core,input_matrix=G,readout=C,task_ids=np.arange(9,dtype=np.int16)),0),
      ('mirror_recurrent_task_views',pred,pack(shared_core=core,shared_bases=bases,task_codes=task_codes,task_ids=np.arange(9,dtype=np.int16),input_matrix=G,readout=C),9*2*D*D),
      ('native_lowrank_task_conditioning',pred,pack(shared_core=core,shared_bases=bases,task_codes=task_codes,task_ids=np.arange(9,dtype=np.int16),input_matrix=G,readout=C),9*2*D*D)]
    for name,As,payload,decode in payloads:
        saved=np.load(io.BytesIO(payload))
        if name=='independent_full_task_matrices': restored=saved['recurrent']
        elif name=='shared_core_no_task_code':restored=np.broadcast_to(saved['recurrent_core'],(9,D,D))
        else:restored=np.stack([saved['shared_core']+a*saved['shared_bases'][0]+b*saved['shared_bases'][1] for a,b in saved['task_codes']])
        rt=float(np.max(np.abs(restored-As)))
        if rt>1e-5:raise AssertionError(f'{name} payload roundtrip error {rt}')
        start=time.perf_counter();outputs=[]
        for k in range(9):outputs.append(sim(As[k],G,C,h0[k],us[k])[1])
        wall=time.perf_counter()-start;outputs=np.stack(outputs)
        # Ablate one physical dynamical component at a time; report task-family nRMSE.
        if name in ('mirror_recurrent_task_views','native_lowrank_task_conditioning'):
            ab1=[];ab2=[]
            for k,(a,b) in enumerate(TASKS):
                y1=sim(core+b*bases[1],G,C,h0[k],us[k])[1]
                y2=sim(core+a*bases[0],G,C,h0[k],us[k])[1]
                ab1.append(nrmse(y1,target[k]));ab2.append(nrmse(y2,target[k]))
            b1=float(np.mean([ab1[k] for k in range(9) if TASKS[k,0]!=0]));b2=float(np.mean([ab2[k] for k in range(9) if TASKS[k,1]!=0]))
        else:b1=b2=float('nan')
        method_rho=float(max(abs(np.linalg.eigvals(A)).max() for A in As))
        methods.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),
          heldout_one_step_nrmse=nrmse(np.einsum('kij,knj->kni',As,h0),np.einsum('kij,knj->kni',teacher,h0)),
          heldout_rollout_nrmse=nrmse(outputs[HELD],target[HELD]),all_task_rollout_nrmse=nrmse(outputs,target),max_spectral_radius=method_rho,
          b1_ablation_mean_nrmse=b1,b2_ablation_mean_nrmse=b2,active_macs_per_step=D*D+D*U+D*D,reconstruction_macs_total=decode,
          train_sequences=len(TRAIN)*N,test_sequences=9*N,sequence_length=T,optimizer_updates=0,wall_time_s=wall,payload_roundtrip_max_abs=rt,
          status_note='exact native low-rank alias' if name=='mirror_recurrent_task_views' else ''))
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=methods[0].keys(),lineterminator='\n');w.writeheader();w.writerows(methods)
    return methods

if __name__=='__main__':
    for r in run():print(r)
