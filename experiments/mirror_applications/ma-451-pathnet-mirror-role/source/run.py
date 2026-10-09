#!/usr/bin/env python3
"""Frozen MA-451 PathNet path/module-role mechanism screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
D,INNER,OUTER,TASKS,SUPPORT,QUERY,LR=4,4,160,8,12,40,.06
ROUTES=[(0,0),(0,1),(1,0),(1,1)]
ROLE_ANGLES=[(-.55,.4),(.55,-.4)]

def rotate(w,z):
    out=w.clone()
    for k,(i,j) in enumerate(((0,1),(2,3))):
        c,s=torch.cos(z[k]),torch.sin(z[k]);out[i]=c*w[i]-s*w[j];out[j]=s*w[i]+c*w[j]
    return out

def teacher(seed):
    r=np.random.default_rng(seed+771)
    return torch.tensor(r.normal(scale=.65,size=(2,D)),dtype=torch.float32),torch.tensor(r.normal(scale=.65,size=(2,D)),dtype=torch.float32)

def sample_task(rng,n,path,angle,a_true,b_true):
    z=torch.tensor(angle,dtype=torch.float32);w=rotate(a_true[path[0]]+b_true[path[1]],z)
    x=torch.tensor(rng.normal(size=(n,D)),dtype=torch.float32); noise=torch.tensor(rng.normal(scale=.02,size=n),dtype=torch.float32)
    return x,x@w+noise

def weight(method,a,b,path,z,state):
    base=a[path[0]]+b[path[1]]
    if method in ('mirror','native_givens'):return rotate(base,z)
    if method=='rank2':return base+state[0][path[0]*2+path[1]]@z
    return base

def mse(x,y,w):return ((x@w-y)**2).mean()

def initialize(method,seed):
    torch.manual_seed(seed+({"path_only":1,"mirror":2,"native_givens":2,"rank2":3}[method]))
    a=torch.randn(2,D)*.15;b=torch.randn(2,D)*.15;state=[]
    if method=='rank2':state=[torch.randn(4,D,2)*.15]
    return a,b,state

def adapt(method,a,b,path,state,x,y):
    if method=='path_only':return torch.zeros(2)
    z=torch.zeros(2,dtype=torch.float32,requires_grad=True)
    for _ in range(INNER):
        w=weight(method,a.detach(),b.detach(),path,z,[q.detach() for q in state]);g,=torch.autograd.grad(mse(x,y,w),z)
        z=(z-LR*g).detach().requires_grad_(True)
    return z.detach()

def train(method,seed):
    a,b,state=initialize(method,seed);a_true,b_true=teacher(seed);rng=np.random.default_rng(seed+1001)
    outer_lr={'path_only':.035,'mirror':.025,'native_givens':.025,'rank2':.02}[method]
    start=time.perf_counter()
    for _ in range(OUTER):
        grads=[torch.zeros_like(a),torch.zeros_like(b)]+[torch.zeros_like(q) for q in state]
        for _ in range(TASKS):
            route=int(rng.integers(0,4));path=ROUTES[route];angle=rng.uniform(-.65,.65,size=2)
            x,y=sample_task(rng,SUPPORT+QUERY,path,angle,a_true,b_true);xs,xq=x[:SUPPORT],x[SUPPORT:];ys,yq=y[:SUPPORT],y[SUPPORT:]
            if method=='path_only':z=torch.zeros(2)
            else:z=adapt(method,a,b,path,state,xs,ys)
            aa=a.detach().clone().requires_grad_(True);bb=b.detach().clone().requires_grad_(True)
            ss=[q.detach().clone().requires_grad_(True) for q in state]
            pred=weight(method,aa,bb,path,z,ss);gs=torch.autograd.grad(mse(xq,yq,pred),[aa,bb]+ss)
            for i,g in enumerate(gs):grads[i]+=g.detach()
        scale=1/TASKS;a=(a-outer_lr*grads[0]*scale).detach();b=(b-outer_lr*grads[1]*scale).detach()
        state=[(q-outer_lr*g*scale).detach() for q,g in zip(state,grads[2:])]
    wall=time.perf_counter()-start
    seen=OUTER*TASKS*(QUERY+(0 if method=='path_only' else SUPPORT))
    factor=0 if method=='path_only' else (2 if method in ('mirror','native_givens','rank2') else 0)
    tr_ops=OUTER*TASKS*(QUERY*D+SUPPORT*INNER*D*factor)
    return a,b,state,{'outer_updates':OUTER,'inner_updates':OUTER*TASKS*INNER if method!='path_only' else 0,'train_examples':seen,'train_wall_s':wall,'training_active_ops_proxy':tr_ops}

def serialize(arrays,meta):
    buf=io.BytesIO();d={k:np.asarray(v) for k,v in arrays.items()};d['__schema_json__']=np.frombuffer(json.dumps(meta,sort_keys=True).encode(),dtype=np.uint8);np.savez(buf,**d);return buf.getvalue()

def evaluate(method,a,b,state,seed):
    at,bt=teacher(seed);rng=np.random.default_rng(seed+8801);scores=[];vectors=[];codes=[];routes=[];adapt_w=[];query_w=[]
    for ri,path in enumerate(ROUTES):
        for angle in ROLE_ANGLES:
            x,y=sample_task(rng,SUPPORT+QUERY,path,angle,at,bt);xs,xq=x[:SUPPORT],x[SUPPORT:];ys,yq=y[:SUPPORT],y[SUPPORT:]
            st=time.perf_counter()
            if method=='independent':w=torch.linalg.lstsq(xs,ys).solution;z=None
            elif method=='path_only':w=weight(method,a,b,path,torch.zeros(2),[]);z=None
            else:z=adapt(method,a,b,path,state,xs,ys);w=weight(method,a,b,path,z,state)
            if method=='path_only':adapt_w.append(0.)
            else:adapt_w.append(time.perf_counter()-st)
            qt=time.perf_counter();scores.append(float(torch.sqrt(mse(xq,yq,w))));query_w.append(time.perf_counter()-qt);vectors.append(w.detach().numpy());routes.append(path)
            if z is not None:codes.append(z.numpy())
    arrays={}
    if method in ('path_only','mirror','native_givens','rank2'):
        arrays={'layer_a':a.detach().numpy(),'layer_b':b.detach().numpy(),'route_ids':np.asarray(routes,dtype=np.int8)}
        if method in ('mirror','native_givens'):arrays['role_codes']=np.stack(codes).astype(np.float32)
        if method=='rank2':arrays['rank2_route_bases']=state[0].detach().numpy();arrays['adapter_codes']=np.stack(codes).astype(np.float32)
    else:arrays={'task_vectors':np.stack(vectors).astype(np.float32)}
    fam='path-givens-role-v1' if method in ('mirror','native_givens') else method
    payload=serialize(arrays,{'format':'MA451-path-bank-v1','method_family':fam,'tasks':8,'dtype':'float32','dims':D,'inner_updates':INNER})
    ops=8*QUERY*D
    if method in ('mirror','native_givens','rank2'):ops+=8*SUPPORT*INNER*D*2
    if method=='independent':ops+=8*SUPPORT*D*D
    aw=float(sum(adapt_w));qw=float(sum(query_w))
    return {'query_rmse':float(np.mean(scores)),'query_rmse_sd':float(np.std(scores,ddof=1)),'payload_bytes':len(payload),'adaptation_wall_s':aw,'inference_wall_s':qw,'end_to_end_wall_s':aw+qw,'adaptation_examples':0 if method=='path_only' else 8*SUPPORT,'query_examples':8*QUERY,'evaluation_active_ops_proxy':ops,'payload_sha256':hashlib.sha256(payload).hexdigest()},payload

def run(seed,out):
    out.mkdir(parents=True,exist_ok=True);res={};states={}
    for method in ('path_only','mirror','native_givens','rank2'):
        a,b,state,tm=train(method,seed);states[method]=(a,b,state,tm);em,payload=evaluate(method,a,b,state,seed)
        res[method]={**em,**tm,'active_ops_proxy':em['evaluation_active_ops_proxy']+tm['training_active_ops_proxy']};(out/f'{method}_payload.npz').write_bytes(payload)
    em,payload=evaluate('independent',*states['path_only'][:2],[],seed)
    em.update({'outer_updates':0,'inner_updates':0,'train_examples':0,'train_wall_s':0.,'training_active_ops_proxy':0,'active_ops_proxy':em['evaluation_active_ops_proxy']});res['independent']=em;(out/'independent_payload.npz').write_bytes(payload)
    d={'experiment_id':'MA-451','seed':seed,'split':'dev','task':{'dim':D,'support':SUPPORT,'query':QUERY,'inner':INNER,'outer':OUTER,'tasks':TASKS},'methods':res}
    (out/'metrics.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
