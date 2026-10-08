#!/usr/bin/env python3
"""Deterministic CPU mechanism screen for MA-255."""
import argparse, io, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s],[s,c]])


def world(seed, d=24, tasks=6):
    rng=np.random.default_rng(seed)
    vals=np.linspace(.35,1.65,d)
    # Task family has exactly one functional degree of variation: an orthogonal
    # two-coordinate chart acting on a common physical operator.
    base=np.diag(vals)
    angles=np.linspace(-.8,.8,tasks)
    mats=[]
    for a in angles:
        q=np.eye(d); q[:2,:2]=rot(a)
        mats.append(q@base@q.T)
    xs=[]; ys=[]; xt=[]; yt=[]
    for w in mats:
        x=rng.normal(size=(384,d)); z=rng.normal(size=(512,d))
        xs.append(x); ys.append(x@w.T); xt.append(z); yt.append(z@w.T)
    return np.array(mats),np.array(xs),np.array(ys),np.array(xt),np.array(yt)


def fit_task_mats(xs,ys):
    return np.array([np.linalg.lstsq(x,y,rcond=None)[0].T for x,y in zip(xs,ys)])


def mse(mats,xt,yt):
    return float(np.mean([(x@m.T-y)**2 for m,x,y in zip(mats,xt,yt)]))


def payload_bytes(arrays):
    b=io.BytesIO(); np.savez(b,**{f'a{i}':np.asarray(a) for i,a in enumerate(arrays)})
    return len(b.getvalue())


def psp(target,seed):
    # Native diagonal PSP: bind each full task map with a fixed random sign
    # context, superpose in one physical tensor, then unbind by its context.
    rng=np.random.default_rng(seed+77); row=rng.choice([-1.,1.],size=(len(target),target.shape[1])).astype(np.float32); col=rng.choice([-1.,1.],size=(len(target),target.shape[2])).astype(np.float32)
    contexts=row[:,:,None]*col[:,None,:]
    physical=np.sum(target*contexts,axis=0)
    decoded=np.array([physical*c for c in contexts])
    return decoded,[physical,row,col]


def lowrank(target,rank):
    mean=target.mean(axis=0); delta=(target-mean).reshape(len(target),-1)
    u,s,v=np.linalg.svd(delta,full_matrices=False)
    basis=v[:rank].reshape(rank,*mean.shape)
    code=u[:,:rank]*s[:rank]
    decoded=mean+np.einsum('tr,rij->tij',code,basis)
    return decoded,[mean,basis,code]


def mirror(target):
    # Fit shared diagonal spectrum and one task angle per target using training
    # maps only. Optimize all physical values/coordinates by least squares.
    d=target.shape[-1]; n=len(target)
    def build(p):
        vals=p[:d]; angles=p[d:d+n]
        out=[]
        for a in angles:
            q=np.eye(d); q[:2,:2]=rot(a)
            out.append(q@np.diag(vals)@q.T)
        return np.array(out)
    # deterministic initialization from average diagonal and broad angle grid
    p0=np.r_[np.mean(np.diagonal(target,axis1=1,axis2=2),axis=0),np.linspace(-.5,.5,n)]
    res=minimize(lambda p: np.mean((build(p)-target)**2),p0,method='BFGS',options={'maxiter':400,'gtol':1e-10})
    return build(res.x),[res.x[:d],res.x[d:]],int(res.nit)


def run(seed,rank):
    target,xs,ys,xt,yt=world(seed)
    fitted=fit_task_mats(xs,ys)
    psp_m,psp_state=psp(fitted,seed)
    low_m,low_state=lowrank(fitted,rank)
    mir_m,mir_state,steps=mirror(fitted)
    methods=[('independent',fitted,[fitted]),('psp',psp_m,psp_state),('task_code_lowrank',low_m,low_state),('mirror_rotation',mir_m,mir_state)]
    rows=[]
    for name,pred,state in methods:
        start=time.perf_counter(); value=mse(pred,xt,yt); elapsed=time.perf_counter()-start
        rows.append({'method':name,'mse':value,'serialized_bytes':payload_bytes(state),'wall_time_s':elapsed,'updates':0,'train_examples':len(xs)*len(xs[0]),'inference_mac_proxy':len(xt)*xt.shape[1]*pred.shape[1]*pred.shape[2]})
    return rows,steps


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',nargs='+',type=int,required=True); ap.add_argument('--rank',type=int,default=4); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    records=[]
    for seed in args.seeds:
        rows,steps=run(seed,args.rank)
        for row in rows: row.update(seed=seed,mirror_optimizer_steps=steps)
        records.extend(rows)
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps({'rows':len(records),'out':str(args.out)}))
if __name__=='__main__': main()
