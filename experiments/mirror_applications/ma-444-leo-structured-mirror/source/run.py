#!/usr/bin/env python3
"""Frozen MA-444 matched-latent decoder experiment."""
from __future__ import annotations
import argparse, hashlib, io, json, time
from pathlib import Path
import numpy as np
import torch

D, Z, HIDDEN = 9, 3, 24
SUPPORT, QUERY, INNER, OUTER, TASKS, LR = 12, 40, 4, 160, 8, 0.06
TEMPLATE = torch.tensor([1.0, .4, .6, -.8, .7, .25, .2, -.4, .5], dtype=torch.float32)

def rotate(base: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    out = base.clone()
    for idx, (i,j) in enumerate(((0,1),(2,3),(4,5))):
        c,s=torch.cos(z[idx]),torch.sin(z[idx])
        out[i]=c*base[i]-s*base[j]
        out[j]=s*base[i]+c*base[j]
    return out

def decode(method: str, base: torch.Tensor, z: torch.Tensor, state: list[torch.Tensor]) -> torch.Tensor:
    if method in ("mirror","native_givens"):
        return rotate(base,z)
    if method=="leo":
        w1,b1,w2,b2=state
        return base + w2 @ torch.tanh(w1 @ z + b1) + b2
    if method=="rank3":
        (basis,)=state
        return base+basis@z
    raise ValueError(method)

def rng_for(seed:int,offset:int=0)->np.random.Generator:
    return np.random.default_rng(seed+offset)

def sample_task(rng:np.random.Generator,n:int):
    angles=torch.tensor(rng.uniform(-.9,.9,size=Z),dtype=torch.float32)
    target=rotate(TEMPLATE,angles)
    x=torch.tensor(rng.normal(size=(n,D)),dtype=torch.float32)
    noise=torch.tensor(rng.normal(scale=.02,size=n),dtype=torch.float32)
    return angles,x,x@target+noise

def loss(x,y,w): return torch.mean((x@w-y)**2)

def init_model(method:str,seed:int):
    torch.manual_seed(seed+{"full":11,"mirror":22,"native_givens":22,"leo":33,"rank3":44}[method])
    base=TEMPLATE.clone()
    if method=="leo": state=[torch.randn(HIDDEN,Z)*.12,torch.zeros(HIDDEN),torch.randn(D,HIDDEN)*.04,torch.zeros(D)]
    elif method=="rank3": state=[torch.randn(D,Z)*.15]
    else: state=[]
    return base,state

def adapt(method,base,state,x,y):
    if method=="full":
        w=base.detach().clone().requires_grad_(True)
        for _ in range(INNER):
            g,=torch.autograd.grad(loss(x,y,w),w)
            w=(w-LR*g).detach().requires_grad_(True)
        return w.detach()
    z=torch.zeros(Z,dtype=base.dtype,requires_grad=True)
    for _ in range(INNER):
        w=decode(method,base.detach(),z,[p.detach() for p in state])
        g,=torch.autograd.grad(loss(x,y,w),z)
        z=(z-LR*g).detach().requires_grad_(True)
    return z.detach()

def train(method:str,seed:int):
    base,state=init_model(method,seed)
    rng=rng_for(seed+71)
    outer_lr={"full":.08,"mirror":.04,"native_givens":.04,"leo":.006,"rank3":.025}[method]
    t0=time.perf_counter()
    for _ in range(OUTER):
        grads=[torch.zeros_like(base)]+[torch.zeros_like(p) for p in state]
        for _ in range(TASKS):
            _,x,y=sample_task(rng,SUPPORT+QUERY)
            xs,xq=x[:SUPPORT],x[SUPPORT:]; ys,yq=y[:SUPPORT],y[SUPPORT:]
            if method=="full":
                adapted=adapt(method,base,[],xs,ys).detach().requires_grad_(True)
                g,=torch.autograd.grad(loss(xq,yq,adapted),adapted); grads[0]+=g.detach()
            else:
                z=adapt(method,base,state,xs,ys)
                req=[base.detach().clone().requires_grad_(True)]+[p.detach().clone().requires_grad_(True) for p in state]
                w=decode(method,req[0],z,req[1:])
                gs=torch.autograd.grad(loss(xq,yq,w),req)
                for j,g in enumerate(gs): grads[j]+=g.detach()
        denom=float(TASKS)
        base=(base-outer_lr*grads[0]/denom).detach()
        state=[(p-outer_lr*g/denom).detach() for p,g in zip(state,grads[1:])]
    elapsed=time.perf_counter()-t0
    factor=2 if method in ("mirror","native_givens","rank3") else (3 if method=="leo" else 1)
    per_episode=SUPPORT*INNER*D*factor+QUERY*D
    return base,state,{"outer_updates":OUTER,"inner_updates":OUTER*TASKS*INNER,
        "train_examples":OUTER*TASKS*(SUPPORT+QUERY),"train_wall_s":elapsed,
        "training_active_ops_proxy":OUTER*TASKS*per_episode}

def fit_full(xs,ys): return torch.linalg.lstsq(xs,ys).solution

def pack(arrays,metadata):
    b=io.BytesIO(); payload={k:np.asarray(v,dtype=np.float32) for k,v in arrays.items()}
    payload["__schema_json__"]=np.frombuffer(json.dumps(metadata,sort_keys=True).encode(),dtype=np.uint8)
    np.savez(b,**payload); return b.getvalue()

def evaluate(method,base,state,seed):
    rng=rng_for(seed+901); scores=[]; vecs=[]; codes=[]; adapt_t=[]; query_t=[]
    for _ in range(16):
        _,x,y=sample_task(rng,SUPPORT+QUERY); xs,xq=x[:SUPPORT],x[SUPPORT:]; ys,yq=y[:SUPPORT],y[SUPPORT:]
        t=time.perf_counter()
        if method in ("mirror","native_givens","leo","rank3"):
            z=adapt(method,base,state,xs,ys); w=decode(method,base,z,state); codes.append(z.detach().numpy())
        elif method=="full": w=adapt(method,base,[],xs,ys); codes.append(w.detach().numpy())
        elif method=="independent": w=fit_full(xs,ys); codes.append(w.detach().numpy())
        elif method=="no_adapt": w=base; adapt_t.append(0.0)
        else: raise ValueError(method)
        if method!="no_adapt": adapt_t.append(time.perf_counter()-t)
        qt=time.perf_counter(); scores.append(float(torch.sqrt(loss(xq,yq,w)))); query_t.append(time.perf_counter()-qt); vecs.append(w.detach().numpy())
    arrays={}
    if method in ("mirror","native_givens","leo","rank3","full","no_adapt"): arrays["shared_base"]=base.numpy()[None,:]
    if method in ("mirror","native_givens","leo","rank3"): arrays["task_codes"]=np.stack(codes)
    if method=="leo":
        for name,p in zip(("w1","b1","w2","b2"),state): arrays[f"decoder_{name}"]=p.detach().numpy()
    if method=="rank3": arrays["basis"]=state[0].detach().numpy()
    if method=="full": arrays["task_vectors"]=np.stack(codes)
    if method=="independent": arrays["task_vectors"]=np.stack(codes)
    family="three-angle-conditioned-linear-v1" if method in ("mirror","native_givens") else method
    metadata={"format":"MA444-inference-npz-v1","method_family":family,"task_count":16,"dtype":"float32","dimensions":D,"latent_dimension":Z,"inner_updates":INNER}
    payload=pack(arrays,metadata)
    aw=float(sum(adapt_t)); qw=float(sum(query_t)); ops=16*QUERY*D
    if method in ("mirror","native_givens","leo","rank3","full"): ops+=16*SUPPORT*INNER*D*(2 if method!="full" else 1)
    elif method=="independent": ops+=16*SUPPORT*D*D
    m={"query_rmse":float(np.mean(scores)),"query_rmse_sd":float(np.std(scores,ddof=1)),"payload_bytes":len(payload),
       "adaptation_wall_s":aw,"inference_wall_s":qw,"end_to_end_wall_s":aw+qw,"adaptation_examples":0 if method=="no_adapt" else 16*SUPPORT,
       "query_examples":16*QUERY,"evaluation_active_ops_proxy":ops,"payload_sha256":hashlib.sha256(payload).hexdigest()}
    return m,payload

def run(seed:int,out:Path):
    out.mkdir(parents=True,exist_ok=True); results={}; states={}
    for method in ("full","mirror","native_givens","leo","rank3"):
        base,state,tm=train(method,seed); states[method]=(base,state,tm)
        em,payload=evaluate(method,base,state,seed); m={**em,**tm,"active_ops_proxy":em["evaluation_active_ops_proxy"]+tm["training_active_ops_proxy"]}
        results[method]=m;(out/f"{method}_payload.npz").write_bytes(payload)
    full_base=states["full"][0]
    for method in ("no_adapt","independent"):
        em,payload=evaluate(method,full_base,[],seed); em.update({"outer_updates":0,"inner_updates":0,"train_examples":0,"train_wall_s":0.0,"training_active_ops_proxy":0,"active_ops_proxy":em["evaluation_active_ops_proxy"]})
        results[method]=em;(out/f"{method}_payload.npz").write_bytes(payload)
    data={"experiment_id":"MA-444","seed":seed,"split":"dev","task":{"dimension":D,"latent_dimension":Z,"support":SUPPORT,"query":QUERY,"inner_updates":INNER,"outer_updates":OUTER,"tasks_per_update":TASKS},"methods":results}
    (out/"metrics.json").write_text(json.dumps(data,indent=2,sort_keys=True)+"\n"); return data

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__': main()
