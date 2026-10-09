#!/usr/bin/env python3
"""Frozen MA-452 factorized path × role address screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
D=4; OUTER=160; EPISODES=8; NQUERY=48; NSUPPORT=12
ROUTES=[(0,0),(0,1),(1,0),(1,1)]
TRAIN_PAIRS=[(0,0),(0,1),(1,1),(1,2),(2,0),(2,2),(3,0),(3,1)]
TEST_PAIRS=[(0,2),(1,0),(2,1),(3,2)]
ANGLES=[(-.55,.4),(.1,-.65),(.6,.5)]

def rotate(w,z):
    out=w.clone()
    for k,(i,j) in enumerate(((0,1),(2,3))):
        c,s=torch.cos(z[k]),torch.sin(z[k]);out[i]=c*w[i]-s*w[j];out[j]=s*w[i]+c*w[j]
    return out

def teacher(seed):
    rng=np.random.default_rng(seed+771)
    return torch.tensor(rng.normal(scale=.65,size=(2,D)),dtype=torch.float32),torch.tensor(rng.normal(scale=.65,size=(2,D)),dtype=torch.float32)

def make_data(rng,n,pair,a_true,b_true):
    path,role=pair;z=torch.tensor(ANGLES[role],dtype=torch.float32);w=rotate(a_true[ROUTES[path][0]]+b_true[ROUTES[path][1]],z)
    x=torch.tensor(rng.normal(size=(n,D)),dtype=torch.float32);noise=torch.tensor(rng.normal(scale=.02,size=n),dtype=torch.float32)
    return x,x@w+noise

def mse(x,y,w):return ((x@w-y)**2).mean()

def initialize(method,seed):
    torch.manual_seed(seed+({'path_only':1,'mirror':2,'native_givens':2,'rank2':3,'role_bias':4}[method]))
    a=torch.randn(2,D)*.15;b=torch.randn(2,D)*.15;extra=[]
    if method in ('mirror','native_givens','rank2'):extra.append(torch.zeros(3,2))
    if method=='rank2':extra.insert(0,torch.randn(D,2)*.15)
    if method=='role_bias':extra=[torch.zeros(3,D)]
    return a,b,extra

def model_weight(method,a,b,path,role,extra):
    base=a[ROUTES[path][0]]+b[ROUTES[path][1]]
    if method in ('mirror','native_givens'):return rotate(base,extra[0][role])
    if method=='rank2':return base+extra[0]@extra[1][role]
    if method=='role_bias':return base+extra[0][role]
    return base

def train(method,seed):
    a,b,extra=initialize(method,seed);at,bt=teacher(seed);rng=np.random.default_rng(seed+1001)
    lr={'path_only':.035,'mirror':.02,'native_givens':.02,'rank2':.02,'role_bias':.02}[method]
    start=time.perf_counter()
    for _ in range(OUTER):
        params=[a,b]+extra;grads=[torch.zeros_like(p) for p in params]
        for _ in range(EPISODES):
            pair=TRAIN_PAIRS[int(rng.integers(len(TRAIN_PAIRS)))];x,y=make_data(rng,NQUERY,pair,at,bt)
            local=[p.detach().clone().requires_grad_(True) for p in params]
            w=model_weight(method,local[0],local[1],pair[0],pair[1],local[2:]);gs=torch.autograd.grad(mse(x,y,w),local)
            for i,g in enumerate(gs):grads[i]+=g.detach()
        params=[(p-lr*g/EPISODES).detach() for p,g in zip(params,grads)]
        a,b,extra=params[0],params[1],params[2:]
    wall=time.perf_counter()-start
    factor={'path_only':1,'mirror':2,'native_givens':2,'rank2':2,'role_bias':2}[method]
    ops=OUTER*EPISODES*NQUERY*D*factor
    return a,b,extra,{'outer_updates':OUTER,'inner_updates':0,'train_examples':OUTER*EPISODES*NQUERY,'train_wall_s':wall,'training_active_ops_proxy':ops}

def payload_bytes(arrays,meta):
    buf=io.BytesIO();d={k:np.asarray(v) for k,v in arrays.items()};d['__schema_json__']=np.frombuffer(json.dumps(meta,sort_keys=True).encode(),dtype=np.uint8);np.savez(buf,**d);return buf.getvalue()

def evaluate(method,a,b,extra,seed):
    at,bt=teacher(seed);rng=np.random.default_rng(seed+8801);scores=[];vectors=[];adapt_time=[];query_time=[];routes=[];roles=[]
    for pair in TEST_PAIRS:
        x,y=make_data(rng,NSUPPORT+NQUERY,pair,at,bt);xs,xq=x[:NSUPPORT],x[NSUPPORT:];ys,yq=y[:NSUPPORT],y[NSUPPORT:]
        st=time.perf_counter()
        if method=='independent':w=torch.linalg.lstsq(xs,ys).solution;adapt_time.append(time.perf_counter()-st)
        else:w=model_weight(method,a,b,pair[0],pair[1],extra);adapt_time.append(0.)
        qt=time.perf_counter();scores.append(float(torch.sqrt(mse(xq,yq,w))));query_time.append(time.perf_counter()-qt);vectors.append(w.detach().numpy())
        routes.append(ROUTES[pair[0]]);roles.append(pair[1])
    arrays={'route_ids':np.asarray(routes,dtype=np.int8),'role_ids':np.asarray(roles,dtype=np.int8)}
    if method=='independent':arrays['task_vectors']=np.stack(vectors).astype(np.float32)
    else:
        arrays['layer_a']=a.detach().numpy();arrays['layer_b']=b.detach().numpy()
        if method in ('mirror','native_givens'):arrays['role_codebook']=extra[0].detach().numpy()
        elif method=='rank2':arrays['rank2_basis']=extra[0].detach().numpy();arrays['role_codebook']=extra[1].detach().numpy()
        elif method=='role_bias':arrays['role_vectors']=extra[0].detach().numpy()
    family='factorized-givens-role-v1' if method in ('mirror','native_givens') else method
    raw=payload_bytes(arrays,{'format':'MA452-factorized-bank-v1','method_family':family,'heldout_tasks':len(TEST_PAIRS),'dtype':'float32','dim':D})
    ops=len(TEST_PAIRS)*NQUERY*D
    if method=='independent':ops+=len(TEST_PAIRS)*NSUPPORT*D*D
    elif method in ('mirror','native_givens','rank2','role_bias'):ops*=2
    aw=float(sum(adapt_time));qw=float(sum(query_time))
    return {'query_rmse':float(np.mean(scores)),'query_rmse_sd':float(np.std(scores,ddof=1)),'payload_bytes':len(raw),'adaptation_wall_s':aw,'inference_wall_s':qw,'end_to_end_wall_s':aw+qw,'adaptation_examples':len(TEST_PAIRS)*NSUPPORT if method=='independent' else 0,'query_examples':len(TEST_PAIRS)*NQUERY,'evaluation_active_ops_proxy':ops,'payload_sha256':hashlib.sha256(raw).hexdigest()},raw

def run(seed,out):
    out.mkdir(parents=True,exist_ok=True);res={};states={}
    for method in ('path_only','mirror','native_givens','rank2','role_bias'):
        a,b,extra,tm=train(method,seed);states[method]=(a,b,extra);em,raw=evaluate(method,a,b,extra,seed)
        res[method]={**em,**tm,'active_ops_proxy':em['evaluation_active_ops_proxy']+tm['training_active_ops_proxy']};(out/f'{method}_payload.npz').write_bytes(raw)
    em,raw=evaluate('independent',None,None,[],seed);em.update({'outer_updates':0,'inner_updates':0,'train_examples':0,'train_wall_s':0.,'training_active_ops_proxy':0,'active_ops_proxy':em['evaluation_active_ops_proxy']});res['independent']=em;(out/'independent_payload.npz').write_bytes(raw)
    d={'experiment_id':'MA-452','seed':seed,'split':'dev','task':{'dim':D,'train_pairs':TRAIN_PAIRS,'heldout_pairs':TEST_PAIRS,'examples_per_update':NQUERY,'outer':OUTER,'episodes_per_update':EPISODES},'methods':res}
    (out/'metrics.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);x=p.parse_args();print(json.dumps(run(x.seed,x.out),sort_keys=True))
if __name__=='__main__':main()
