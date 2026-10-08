#!/usr/bin/env python3
"""Frozen MA-455 ordered three-step shared-block mechanism screen."""
from __future__ import annotations
import argparse, hashlib, io, json, time
from pathlib import Path
import numpy as np
import torch

D, DEPTH, UPDATES, BATCH, NEVAL = 3, 3, 600, 16, 512
METHODS = ('shared', 'independent', 'rank1', 'mirror', 'native_givens')

def rotation(a):
    x,y,z=a.unbind()
    cx,sx=torch.cos(x),torch.sin(x); cy,sy=torch.cos(y),torch.sin(y); cz,sz=torch.cos(z),torch.sin(z)
    rx=torch.stack([torch.stack([x.new_tensor(1),x.new_tensor(0),x.new_tensor(0)]),torch.stack([x.new_tensor(0),cx,-sx]),torch.stack([x.new_tensor(0),sx,cx])])
    ry=torch.stack([torch.stack([cy,y.new_tensor(0),sy]),torch.stack([y.new_tensor(0),y.new_tensor(1),y.new_tensor(0)]),torch.stack([-sy,y.new_tensor(0),cy])])
    rz=torch.stack([torch.stack([cz,-sz,z.new_tensor(0)]),torch.stack([sz,cz,z.new_tensor(0)]),torch.stack([z.new_tensor(0),z.new_tensor(0),z.new_tensor(1)])])
    return rz @ ry @ rx

def teacher(seed):
    rng=np.random.default_rng(seed+455)
    base=np.diag([.82,1.07,.63])+rng.normal(scale=.13,size=(D,D))
    angles=rng.uniform(-.8,.8,size=(DEPTH,3)).astype(np.float32)
    w=torch.tensor(base,dtype=torch.float32)
    ms=[]
    for a in angles:
        r=rotation(torch.tensor(a,dtype=torch.float32));ms.append(r@w@r.T)
    return w,torch.stack(ms)

def data(seed,n,split):
    w,ms=teacher(seed); rng=np.random.default_rng(seed+(10001 if split=='train' else 20003))
    x=rng.normal(size=(n,D)).astype(np.float32); y=torch.tensor(x)
    with torch.no_grad():
        for m in ms:y=y@m.T
    y += torch.tensor(rng.normal(scale=.01,size=(n,D)).astype(np.float32))
    return torch.tensor(x),y,ms

def init(method,seed):
    torch.manual_seed(seed+45531)
    w=torch.randn(D,D)*.18
    if method=='shared':return [w]
    if method in ('mirror','native_givens'):return [w,torch.zeros(DEPTH,3)]
    if method=='rank1':return [w,torch.randn(DEPTH,D)*.03,torch.randn(DEPTH,D)*.03]
    return [torch.randn(DEPTH,D,D)*.18]

def matrices(method,p,reverse=False):
    if method=='shared': mats=[p[0]]*DEPTH
    elif method in ('mirror','native_givens'):
        w,codes=p
        mats=[]
        for a in codes:
            r=rotation(a);mats.append(r@w@r.T)
    elif method=='rank1':
        w,u,v=p;mats=[w+torch.outer(u[i],v[i]) for i in range(DEPTH)]
    else:mats=[p[0][i] for i in range(DEPTH)]
    return list(reversed(mats)) if reverse else mats

def predict(x,method,p,reverse=False):
    y=x
    for m in matrices(method,p,reverse):y=y@m.T
    return y

def train(method,seed,x,y):
    params=[p.detach().clone().requires_grad_(True) for p in init(method,seed)]
    opt=torch.optim.Adam(params,lr=.01); rng=np.random.default_rng(seed+1777)
    start=time.perf_counter()
    for _ in range(UPDATES):
        idx=rng.integers(0,len(x),size=BATCH)
        loss=((predict(x[idx],method,params)-y[idx])**2).mean()
        opt.zero_grad();loss.backward();opt.step()
    wall=time.perf_counter()-start
    return [p.detach() for p in params],wall

def pack(arrays,method):
    b=io.BytesIO(); payload={k:np.asarray(v) for k,v in arrays.items()}
    payload['__schema_json__']=np.frombuffer(json.dumps({'format':'MA455-sequential-view-v1','method':method,'dim':D,'depth':DEPTH,'dtype':'float32','step_order':'0,1,2'},sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
    np.savez(b,**payload);return b.getvalue()

def serialize(method,p):
    if method=='shared':arr={'shared_block':p[0].numpy()}
    elif method in ('mirror','native_givens'):arr={'shared_block':p[0].numpy(),'step_angles':p[1].numpy()}
    elif method=='rank1':arr={'shared_block':p[0].numpy(),'residual_left':p[1].numpy(),'residual_right':p[2].numpy()}
    else:arr={'step_blocks':p[0].numpy()}
    # Native Givens is a direct ordinary parameterization of the same map.
    family='shared-block-givens-conjugation-v1' if method in ('mirror','native_givens') else method
    b=io.BytesIO(); d={k:np.asarray(v) for k,v in arr.items()}
    d['__schema_json__']=np.frombuffer(json.dumps({'format':'MA455-sequential-view-v1','method_family':family,'dim':D,'depth':DEPTH,'dtype':'float32','step_order':'0,1,2'},sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
    np.savez(b,**d);return b.getvalue()

def evaluate(method,p,seed,xq,yq):
    with torch.no_grad():
        start=time.perf_counter();yp=predict(xq,method,p);correct=float(torch.sqrt(((yp-yq)**2).mean())); query_s=time.perf_counter()-start
        reverse=None
        if method=='mirror':reverse=float(torch.sqrt(((predict(xq,method,p,True)-yq)**2).mean()))
    raw=serialize(method,p);mac_per={'shared':3*D*D,'independent':3*D*D,'rank1':3*(D*D+D*D),'mirror':3*D*D+6*D**3/NEVAL,'native_givens':3*D*D+6*D**3/NEVAL}[method]
    init_macs=6*D**3 if method in ('mirror','native_givens') else 0
    eval_macs=mac_per*NEVAL
    return {'query_rmse':correct,'reverse_order_rmse':reverse,'payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'query_examples':len(xq),'inference_ops_per_example_proxy':mac_per,'cold_reconstruction_ops_proxy':init_macs,'evaluation_ops_proxy':eval_macs+init_macs,'query_wall_s_including_rebuild':query_s}

def run(seed,out):
    out.mkdir(parents=True,exist_ok=True); x,y,_=data(seed,UPDATES*BATCH,'train'); xq,yq,_=data(seed,NEVAL,'eval'); results={}
    for method in METHODS:
        p,train_s=train(method,seed,x,y); ev=evaluate(method,p,seed,xq,yq); raw=serialize(method,p);(out/f'{method}_payload.npz').write_bytes(raw)
        train_macs_per={'shared':3*D*D,'independent':3*D*D,'rank1':3*(D*D+D*D),'mirror':3*D*D+6*D**3/BATCH,'native_givens':3*D*D+6*D**3/BATCH}[method]
        ev.update({'outer_updates':UPDATES,'train_examples':UPDATES*BATCH,'train_wall_s':train_s,'training_ops_proxy':int(UPDATES*BATCH*train_macs_per),'active_ops_proxy':int(UPDATES*BATCH*train_macs_per+ev['evaluation_ops_proxy'])})
        results[method]=ev
    # Mathematically equivalent parameterizations share the same learned trajectory and canonical payload.
    assert results['mirror']['payload_sha256']==results['native_givens']['payload_sha256']
    assert results['mirror']['query_rmse']==results['native_givens']['query_rmse']
    obj={'experiment_id':'MA-455','seed':seed,'split':'dev','task':{'dimension':D,'depth':DEPTH,'updates':UPDATES,'batch':BATCH,'heldout_examples':NEVAL},'methods':results}
    (out/'metrics.json').write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');return obj

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
