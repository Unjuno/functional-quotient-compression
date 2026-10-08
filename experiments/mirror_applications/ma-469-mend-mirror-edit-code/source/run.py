#!/usr/bin/env python3
"""Frozen MA-469 MEND/full-update versus Mirror edit-code mechanism screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
from torch import nn
D,EDITS,TRAIN_EDITS=6,20,16;SUPPORT,QUERY,NLOCAL=24,128,256;UPDATES,BATCH=2200,8
METHODS=('mend','mirror','native_lowrank','rome','independent_fit','no_edit')

def world(seed):
    rng=np.random.default_rng(seed+469);base=torch.tensor(rng.normal(scale=.2,size=(D,D)).astype(np.float32));basis=torch.tensor(rng.normal(scale=.24,size=(2,D,D)).astype(np.float32));codes=torch.tensor(rng.normal(scale=.35,size=(EDITS,2)).astype(np.float32));keys=[];xs=[];ys=[];signals=[];deltas=[];xq=[];yl=[];xl=[]
    for e in range(EDITS):
        k=rng.normal(size=D).astype(np.float32);k/=np.linalg.norm(k);keys.append(k);delta=torch.einsum('r,rij->ij',codes[e],basis);deltas.append(delta)
        rr=np.random.default_rng(seed+5000+e);x=torch.tensor(k+rr.normal(scale=.28,size=(SUPPORT,D)).astype(np.float32));y=x@(base+delta).T+torch.tensor(rr.normal(scale=.01,size=(SUPPORT,D)).astype(np.float32));res=y-x@base.T;g=torch.einsum('ni,nj->ij',res,x)/SUPPORT;signals.append(g.reshape(-1));xs.append(x);ys.append(y)
        rq=np.random.default_rng(seed+70000+e);xv=torch.tensor(k+rq.normal(scale=.22,size=(QUERY,D)).astype(np.float32));xq.append(xv);yl.append(xv@(base+delta).T)
        local=torch.tensor(rq.normal(size=(NLOCAL,D)).astype(np.float32));xl.append(local)
    return {'base':base,'basis':basis,'codes':codes,'keys':torch.tensor(np.array(keys)),'deltas':torch.stack(deltas),'signals':torch.stack(signals),'xs':xs,'ys':ys,'xq':xq,'yq':yl,'xl':xl}

class Editor(nn.Module):
    def __init__(self,out,seed):
        super().__init__();torch.manual_seed(seed+46901);self.fc1=nn.Linear(D*D,32);self.fc2=nn.Linear(32,out)
    def forward(self,g):return self.fc2(torch.tanh(self.fc1(g)))

def init(method,seed):
    if method in ('mend','mirror','native_lowrank'):
        out=D*D if method=='mend' else 2;net=Editor(out,seed)
        if method in ('mirror','native_lowrank'):
            torch.manual_seed(seed+46902);basis=nn.Parameter(torch.randn(2,D,D)*.1);return {'net':net,'basis':basis}
        return {'net':net}
    return {}

def gen(method,s,g):
    if method=='no_edit':return torch.zeros(D,D)
    if method=='independent_fit':raise ValueError('independent fit is support-derived')
    if method=='rome':raise ValueError('ROME is support-derived')
    out=s['net'](g)
    if method=='mend':return out.reshape(D,D)
    return torch.einsum('r,rij->ij',out,s['basis'])

def params(method,s):return list(s['net'].parameters())+[s['basis']] if method in ('mirror','native_lowrank') else list(s['net'].parameters())

def support_controls(world):
    rome=[];ind=[]
    for x,y in zip(world['xs'],world['ys']):
        res=y-x@world['base'].T;k=x.mean(0);r=res.mean(0);rome.append(torch.outer(r,k)/(k@k+1e-6))
        ind.append(torch.linalg.lstsq(x,res).solution.T.detach())
    return torch.stack(rome),torch.stack(ind)

def pack(arr,meta):
    b=io.BytesIO();d={k:np.asarray(v) for k,v in arr.items()};d['__schema_json__']=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);np.savez(b,**d);return b.getvalue()

def serialize(method,s,world,updates,codes=None):
    common={'base':world['base'].numpy(),'edit_keys':world['keys'].numpy()}
    if method=='mend':common.update({'editor_fc1_weight':s['net'].fc1.weight.detach().numpy(),'editor_fc1_bias':s['net'].fc1.bias.detach().numpy(),'editor_fc2_weight':s['net'].fc2.weight.detach().numpy(),'editor_fc2_bias':s['net'].fc2.bias.detach().numpy(),'edit_updates':updates.numpy()});fam='mend-full-update-v1'
    elif method in ('mirror','native_lowrank'):common.update({'editor_fc1_weight':s['net'].fc1.weight.detach().numpy(),'editor_fc1_bias':s['net'].fc1.bias.detach().numpy(),'editor_fc2_weight':s['net'].fc2.weight.detach().numpy(),'editor_fc2_bias':s['net'].fc2.bias.detach().numpy(),'edit_basis':s['basis'].detach().numpy(),'edit_codes':codes.numpy()});fam='shared-rank2-edit-code-v1'
    elif method=='rome':common.update({'rome_updates':updates.numpy()});fam='analytic-rank1-edit-v1'
    elif method=='independent_fit':common.update({'independent_updates':updates.numpy()});fam='independent-support-edit-v1'
    else:fam='frozen-no-edit-v1'
    return pack(common,{'format':'MA469-edit-system-v1','method_family':fam,'dimension':D,'edits':EDITS,'dtype':'float32'})

def train(method,seed,w):
    if method not in ('mend','mirror','native_lowrank'):return None,0.
    s=init(method,seed);opt=torch.optim.Adam(params(method,s),lr=.01);rng=np.random.default_rng(seed+46911);start=time.perf_counter()
    for _ in range(UPDATES):
        es=rng.integers(TRAIN_EDITS,size=BATCH);loss=0
        for e in es:
            g=w['signals'][e];target=w['deltas'][e];pred=gen(method,s,g);loss=loss+((pred-target)**2).mean()
        loss=loss/BATCH;opt.zero_grad();loss.backward();opt.step()
    return {k:(v.detach() if isinstance(v,torch.Tensor) else v) for k,v in s.items()},time.perf_counter()-start

def evaluate(method,s,seed,w,wall,rome,ind):
    gen_start=time.perf_counter();updates=[];codes=[]
    with torch.no_grad():
        for e in range(EDITS):
            if method in ('mend','mirror','native_lowrank'):
                delta=gen(method,s,w['signals'][e]);updates.append(delta)
                if method in ('mirror','native_lowrank'):codes.append(s['net'](w['signals'][e]).reshape(2))
            elif method=='rome':updates.append(rome[e])
            elif method=='independent_fit':updates.append(ind[e])
            else:updates.append(torch.zeros(D,D))
        updates=torch.stack(updates)
    generation_wall=time.perf_counter()-gen_start
    query_start=time.perf_counter();edit=[];loc=[]
    with torch.no_grad():
        for e in range(TRAIN_EDITS,EDITS):
            x=w['xq'][e];pred=x@(w['base']+updates[e]).T;edit.append(float(torch.sqrt(((pred-w['yq'][e])**2).mean())));xl=w['xl'][e];loc.append(float(torch.sqrt(((xl@updates[e].T)**2).mean())))
    query_wall=time.perf_counter()-query_start
    codes_t=torch.stack(codes) if codes else None;raw=serialize(method,s or {},w,updates,codes_t)
    if method in ('mend','mirror','native_lowrank'):
        out_dim=D*D if method=='mend' else 2
        # Count editor input, output, optional basis decode, and one model application.
        mac=D*D + D*D*32 + 32*out_dim + (2*D*D if method in ('mirror','native_lowrank') else 0)
    else:
        mac=D*D
    train_examples=UPDATES*BATCH if method in ('mend','mirror','native_lowrank') else EDITS*SUPPORT
    m={'heldout_edit_rmse':float(np.mean(edit)),'per_edit_rmse':edit,'locality_rmse':float(np.mean(loc)),'per_edit_locality_rmse':loc,'payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'optimizer_updates':UPDATES if method in ('mend','mirror','native_lowrank') else 0,'support_examples_seen':train_examples,'training_wall_s':wall,'edit_generation_wall_s':generation_wall,'edit_query_wall_s':query_wall,'edit_query_examples':(EDITS-TRAIN_EDITS)*QUERY,'inference_ops_proxy_per_example':D*D,'edit_generation_ops_proxy':EDITS*mac,'training_ops_proxy':(UPDATES*BATCH*mac*3 if method in ('mend','mirror','native_lowrank') else EDITS*SUPPORT*D*D),'active_ops_proxy':EDITS*mac+((UPDATES*BATCH*mac*3) if method in ('mend','mirror','native_lowrank') else EDITS*SUPPORT*D*D)}
    return m,raw

def run(seed,out):
    out.mkdir(parents=True,exist_ok=True);w=world(seed);rome,ind=support_controls(w);result={};states={}
    for method in METHODS:
        s,wall=train(method,seed,w);states[method]=s;m,raw=evaluate(method,s,seed,w,wall,rome,ind);result[method]=m;(out/f'{method}_payload.npz').write_bytes(raw)
    assert result['mirror']['payload_sha256']==result['native_lowrank']['payload_sha256']
    assert result['mirror']['per_edit_rmse']==result['native_lowrank']['per_edit_rmse']
    d={'experiment_id':'MA-469','seed':seed,'split':'dev','task':{'dimension':D,'edits':EDITS,'editor_train_edits':TRAIN_EDITS,'heldout_edits':EDITS-TRAIN_EDITS,'support_per_edit':SUPPORT,'query_per_edit':QUERY},'methods':result};(out/'metrics.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
