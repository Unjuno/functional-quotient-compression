#!/usr/bin/env python3
"""Frozen MA-453 dynamic route plus logical role-view mechanism screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
D=4;OUTER=180;BATCH=8;NEVAL=64
ROLE_ANGLES=[(-.55,.4),(.55,-.4)]

def rotate(w,z):
    out=w.clone()
    for k,(i,j) in enumerate(((0,1),(2,3))):
        c,s=torch.cos(z[k]),torch.sin(z[k]);out[i]=c*w[i]-s*w[j];out[j]=s*w[i]+c*w[j]
    return out

def teacher(seed):
    r=np.random.default_rng(seed+453);return torch.tensor(r.normal(scale=.7,size=(2,D)),dtype=torch.float32)

def data(rng,n,module,role,true_blocks):
    target=rotate(true_blocks[module],torch.tensor(ROLE_ANGLES[role],dtype=torch.float32))
    x=torch.tensor(rng.normal(size=(n,D)),dtype=torch.float32);noise=torch.tensor(rng.normal(scale=.02,size=n),dtype=torch.float32)
    return x,x@target+noise

def mse(x,y,w):return ((x@w-y)**2).mean()

def init(method,seed):
    torch.manual_seed(seed+({'shared':1,'routing':2,'mirror':3,'native_givens':3,'rank2':4,'role_vector':5,'independent':6}[method]))
    if method=='shared':return [torch.randn(D)*.15]
    if method in ('routing','mirror','native_givens'):return [torch.randn(2,D)*.15]+([torch.zeros(2,2)] if method in ('mirror','native_givens') else [])
    if method=='rank2':return [torch.randn(2,D)*.15,torch.randn(D,2)*.15,torch.zeros(2,2)]
    if method=='role_vector':return [torch.randn(2,D)*.15,torch.zeros(2,D)]
    return [torch.randn(2,2,D)*.15]

def wmethod(method,params,module,role):
    if method=='shared':return params[0]
    if method=='routing':return params[0][module]
    if method in ('mirror','native_givens'):return rotate(params[0][module],params[1][role])
    if method=='rank2':return params[0][module]+params[1]@params[2][role]
    if method=='role_vector':return params[0][module]+params[1][role]
    return params[0][module,role]

def train(method,seed):
    params=init(method,seed);true=teacher(seed);rng=np.random.default_rng(seed+1001);lr={'shared':.025,'routing':.025,'mirror':.02,'native_givens':.02,'rank2':.02,'role_vector':.02,'independent':.035}[method]
    t=time.perf_counter()
    for _ in range(OUTER):
        grads=[torch.zeros_like(p) for p in params]
        for _ in range(BATCH):
            module=int(rng.integers(2));role=int(rng.integers(2));x,y=data(rng,1,module,role,true)
            local=[p.detach().clone().requires_grad_(True) for p in params];w=wmethod(method,local,module,role)
            gs=torch.autograd.grad(mse(x,y,w),local)
            for i,g in enumerate(gs):grads[i]+=g.detach()
        params=[(p-lr*g/BATCH).detach() for p,g in zip(params,grads)]
    wall=time.perf_counter()-t;factor={'shared':1,'routing':2,'mirror':3,'native_givens':3,'rank2':3,'role_vector':3,'independent':2}[method]
    return params,{'outer_updates':OUTER,'train_examples':OUTER*BATCH,'train_wall_s':wall,'training_active_ops_proxy':OUTER*BATCH*D*factor}

def pack(arrays,meta):
    b=io.BytesIO();d={k:np.asarray(v) for k,v in arrays.items()};d['__schema_json__']=np.frombuffer(json.dumps(meta,sort_keys=True).encode(),dtype=np.uint8);np.savez(b,**d);return b.getvalue()

def eval_method(method,params,seed):
    true=teacher(seed);rng=np.random.default_rng(seed+8801);scores=[];qw=[]
    for module in range(2):
        for role in range(2):
            x,y=data(rng,NEVAL,module,role,true);w=wmethod(method,params,module,role);t=time.perf_counter();scores.append(float(torch.sqrt(mse(x,y,w))));qw.append(time.perf_counter()-t)
    arrays={'router_table':np.array([0,1],dtype=np.int8),'function_contexts':np.array([[m,r] for m in range(2) for r in range(2)],dtype=np.int8)}
    if method=='shared':arrays['shared_block']=params[0].numpy()
    elif method=='routing':arrays['physical_blocks']=params[0].numpy()
    elif method in ('mirror','native_givens'):arrays['physical_blocks']=params[0].numpy();arrays['role_codes']=params[1].numpy()
    elif method=='rank2':arrays['physical_blocks']=params[0].numpy();arrays['residual_basis']=params[1].numpy();arrays['role_codes']=params[2].numpy()
    elif method=='role_vector':arrays['physical_blocks']=params[0].numpy();arrays['role_vectors']=params[1].numpy()
    else:arrays['logical_blocks']=params[0].numpy()
    fam='routed-givens-role-v1' if method in ('mirror','native_givens') else method
    raw=pack(arrays,{'format':'MA453-routed-bank-v1','method_family':fam,'functions':4,'router':'fixed-context-argmax','dim':D,'dtype':'float32'})
    factor={'shared':1,'routing':2,'mirror':3,'native_givens':3,'rank2':4,'role_vector':3,'independent':2}[method]
    ops=2*NEVAL*D*factor
    m={'query_rmse':float(np.mean(scores)),'query_rmse_sd':float(np.std(scores,ddof=1)),'payload_bytes':len(raw),'inference_wall_s':float(sum(qw)),'end_to_end_wall_s':float(sum(qw)),'router_accuracy':1.0,'evaluation_active_ops_proxy':ops,'active_ops_per_example':factor*D,'query_examples':4*NEVAL,'payload_sha256':hashlib.sha256(raw).hexdigest()}
    return m,raw

def run(seed,out):
    out.mkdir(parents=True,exist_ok=True);res={}
    for method in ('shared','routing','mirror','native_givens','rank2','role_vector','independent'):
        params,tm=train(method,seed);em,raw=eval_method(method,params,seed);res[method]={**em,**tm,'active_ops_proxy':em['evaluation_active_ops_proxy']+tm['training_active_ops_proxy']};(out/f'{method}_payload.npz').write_bytes(raw)
    d={'experiment_id':'MA-453','seed':seed,'split':'dev','task':{'dim':D,'modules':2,'roles':2,'train_updates':OUTER,'eval_examples_per_function':NEVAL},'methods':res};(out/'metrics.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
