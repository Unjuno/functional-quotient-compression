#!/usr/bin/env python3
"""Frozen MA-470 local routed multi-edit shared-basis mechanism screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
from torch import nn
D,EDITS,TRAIN_EDITS=24,20,16
SUPPORT,QUERY,NLOCAL=24,128,512
UPDATES,BATCH=2200,8
RADIUS=1.8
METHODS=('mend','mirror','native_lowrank','rome','independent_fit','no_edit')

def world(seed):
    rng=np.random.default_rng(seed+470)
    base=torch.tensor(rng.normal(scale=.2,size=(D,D)).astype(np.float32))
    basis=torch.tensor(rng.normal(scale=.15,size=(2,D,D)).astype(np.float32))
    codes=torch.tensor(rng.normal(scale=.25,size=(EDITS,2)).astype(np.float32))
    keys=[];deltas=[];signals=[];xs=[];ys=[];xq=[];yq=[]
    for e in range(EDITS):
        k=np.eye(D,dtype=np.float32)[e];keys.append(k)
        delta=torch.einsum('r,rij->ij',codes[e],basis);deltas.append(delta)
        rs=np.random.default_rng(seed+5000+e)
        x=torch.tensor(k+rs.normal(scale=.28,size=(SUPPORT,D)).astype(np.float32))
        y=x@(base+delta).T+torch.tensor(rs.normal(scale=.01,size=(SUPPORT,D)).astype(np.float32))
        res=y-x@base.T
        signals.append(torch.einsum('ni,nj->ij',res,x)/SUPPORT)
        xs.append(x);ys.append(y)
        rq=np.random.default_rng(seed+70000+e)
        xv=torch.tensor(k+rq.normal(scale=.18,size=(QUERY,D)).astype(np.float32))
        xq.append(xv);yq.append(xv@(base+delta).T)
    rl=np.random.default_rng(seed+90000)
    xl=torch.tensor(rl.normal(size=(NLOCAL,D)).astype(np.float32))
    return {'base':base,'basis':basis,'codes':codes,'keys':torch.tensor(np.array(keys)),
            'deltas':torch.stack(deltas),'signals':torch.stack(signals),'xs':xs,'ys':ys,
            'xq':xq,'yq':yq,'xl':xl}

class Editor(nn.Module):
    def __init__(self,out,seed):
        super().__init__();torch.manual_seed(seed+47001)
        self.fc1=nn.Linear(D*D,32);self.fc2=nn.Linear(32,out)
    def forward(self,g):return self.fc2(torch.tanh(self.fc1(g)))

def init(method,seed):
    if method in ('mend','mirror','native_lowrank'):
        out=D*D if method=='mend' else 2;net=Editor(out,seed)
        if method in ('mirror','native_lowrank'):
            torch.manual_seed(seed+47002);basis=nn.Parameter(torch.randn(2,D,D)*.1)
            return {'net':net,'basis':basis}
        return {'net':net}
    return {}

def gen(method,s,g):
    if method=='no_edit':return torch.zeros(D,D)
    if method in ('rome','independent_fit'):raise ValueError('support-derived method')
    out=s['net'](g.reshape(-1))
    return out.reshape(D,D) if method=='mend' else torch.einsum('r,rij->ij',out,s['basis'])

def params(method,s):
    return list(s['net'].parameters())+[s['basis']] if method in ('mirror','native_lowrank') else list(s['net'].parameters())

def route(x,keys):
    d=((keys-x.unsqueeze(0))**2).sum(dim=-1)
    ix=int(torch.argmin(d).item());return ix,float(d[ix])<=RADIUS*RADIUS

def support_controls(w):
    rome=[];ind=[]
    for x,y in zip(w['xs'],w['ys']):
        res=y-x@w['base'].T;k=x.mean(0);r=res.mean(0)
        rome.append(torch.outer(r,k)/(k@k+1e-6))
        ind.append(torch.linalg.lstsq(x,res).solution.T.detach())
    return torch.stack(rome),torch.stack(ind)

def pack(arr,meta):
    b=io.BytesIO();d={k:np.asarray(v) for k,v in arr.items()}
    d['__schema_json__']=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
    np.savez(b,**d);return b.getvalue()

def serialize(method,s,w,updates,codes=None):
    obj={'base':w['base'].numpy(),'edit_keys':w['keys'].numpy(),'route_radius':np.asarray([RADIUS],dtype=np.float32)}
    if method=='mend':
        obj.update({'editor_fc1_weight':s['net'].fc1.weight.detach().numpy(),'editor_fc1_bias':s['net'].fc1.bias.detach().numpy(),
          'editor_fc2_weight':s['net'].fc2.weight.detach().numpy(),'editor_fc2_bias':s['net'].fc2.bias.detach().numpy(),'edit_updates':updates.numpy()})
        family='mend-full-update-bank-v1'
    elif method in ('mirror','native_lowrank'):
        obj.update({'editor_fc1_weight':s['net'].fc1.weight.detach().numpy(),'editor_fc1_bias':s['net'].fc1.bias.detach().numpy(),
          'editor_fc2_weight':s['net'].fc2.weight.detach().numpy(),'editor_fc2_bias':s['net'].fc2.bias.detach().numpy(),
          'edit_basis':s['basis'].detach().numpy(),'edit_codes':codes.numpy()})
        family='shared-rank2-edit-code-v1'
    elif method=='rome':obj.update({'rome_updates':updates.numpy()});family='analytic-rank1-bank-v1'
    elif method=='independent_fit':obj.update({'independent_updates':updates.numpy()});family='independent-support-bank-v1'
    else:family='frozen-no-edit-v1'
    return pack(obj,{'format':'MA470-routed-edit-bank-v1','method_family':family,'dimension':D,'edits':EDITS,'dtype':'float32'})

def train(method,seed,w):
    if method not in ('mend','mirror','native_lowrank'):return None,0.
    s=init(method,seed);opt=torch.optim.Adam(params(method,s),lr=.01)
    rng=np.random.default_rng(seed+47011);start=time.perf_counter()
    for _ in range(UPDATES):
        es=rng.integers(TRAIN_EDITS,size=BATCH);loss=0.
        for e in es:loss=loss+((gen(method,s,w['signals'][e])-w['deltas'][e])**2).mean()
        loss=loss/BATCH;opt.zero_grad();loss.backward();opt.step()
    return {k:(v.detach() if isinstance(v,torch.Tensor) else v) for k,v in s.items()},time.perf_counter()-start

def evaluate(method,s,seed,w,wall,rome,ind):
    gen_start=time.perf_counter();updates=[];codes=[]
    with torch.no_grad():
        for e in range(EDITS):
            if method in ('mend','mirror','native_lowrank'):
                updates.append(gen(method,s,w['signals'][e]))
                if method in ('mirror','native_lowrank'):codes.append(s['net'](w['signals'][e].reshape(-1)).reshape(2))
            elif method=='rome':updates.append(rome[e])
            elif method=='independent_fit':updates.append(ind[e])
            else:updates.append(torch.zeros(D,D))
        updates=torch.stack(updates)
    gen_wall=time.perf_counter()-gen_start
    qstart=time.perf_counter();edit_errors=[];route_correct=[];local_errors=[];local_routed=[];route_counts=[]
    with torch.no_grad():
        for e in range(TRAIN_EDITS,EDITS):
            for x,y in zip(w['xq'][e],w['yq'][e]):
                ix,active=route(x,w['keys']);route_counts.append((ix,active));route_correct.append(ix==e and active)
                delta=updates[ix] if active else torch.zeros(D,D)
                edit_errors.append(float(torch.mean((x@(w['base']+delta).T-y)**2)))
        for x in w['xl']:
            ix,active=route(x,w['keys']);local_routed.append(active)
            delta=updates[ix] if active else torch.zeros(D,D)
            local_errors.append(float(torch.mean((x@(w['base']+delta).T-x@w['base'].T)**2)))
    query_wall=time.perf_counter()-qstart
    # Report RMSE, not MSE accumulated above.
    edit_rmse=float(np.sqrt(np.mean(edit_errors)));locality_rmse=float(np.sqrt(np.mean(local_errors)))
    codes_t=torch.stack(codes) if codes else None;raw=serialize(method,s or {},w,updates,codes_t)
    out_dim=D*D if method=='mend' else 2
    gen_mac=(D*D*32+32*out_dim+(2*D*D if method in ('mirror','native_lowrank') else 0)) if method in ('mend','mirror','native_lowrank') else D*D
    route_arith=EDITS*D
    query_mac=D*D
    inference=route_arith+query_mac
    train_examples=UPDATES*BATCH if method in ('mend','mirror','native_lowrank') else EDITS*SUPPORT
    m={'heldout_edit_rmse':edit_rmse,'per_query_edit_rmse':edit_errors,'locality_rmse':locality_rmse,
       'wrong_route_rate':1.0-float(np.mean(route_correct)),'locality_route_rate':float(np.mean(local_routed)),
       'payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),
       'optimizer_updates':UPDATES if method in ('mend','mirror','native_lowrank') else 0,
       'support_examples_seen':train_examples,'training_wall_s':wall,'edit_generation_wall_s':gen_wall,'edit_query_wall_s':query_wall,
       'edit_query_examples':(EDITS-TRAIN_EDITS)*QUERY,'locality_examples':NLOCAL,
       'edit_generation_ops_proxy':EDITS*gen_mac,'query_ops_proxy_per_example':inference,
       'route_comparisons_per_example':EDITS-1,'route_distance_scalar_ops_per_example':EDITS*D*3,
       'training_ops_proxy':UPDATES*BATCH*gen_mac*3 if method in ('mend','mirror','native_lowrank') else EDITS*SUPPORT*D*D,
       'active_ops_proxy':EDITS*gen_mac+((UPDATES*BATCH*gen_mac*3) if method in ('mend','mirror','native_lowrank') else EDITS*SUPPORT*D*D)+((EDITS-TRAIN_EDITS)*QUERY+NLOCAL)*inference}
    return m,raw

def run(seed,out):
    out.mkdir(parents=True,exist_ok=True);w=world(seed);rome,ind=support_controls(w);results={}
    for method in METHODS:
        s,wall=train(method,seed,w);m,raw=evaluate(method,s,seed,w,wall,rome,ind);results[method]=m;(out/f'{method}_payload.npz').write_bytes(raw)
    assert (out/'mirror_payload.npz').read_bytes()==(out/'native_lowrank_payload.npz').read_bytes()
    assert results['mirror']['heldout_edit_rmse']==results['native_lowrank']['heldout_edit_rmse']
    d={'experiment_id':'MA-470','seed':seed,'split':'dev','task':{'dimension':D,'edits':EDITS,'editor_train_edits':TRAIN_EDITS,'heldout_edits':EDITS-TRAIN_EDITS,'support_per_edit':SUPPORT,'query_per_edit':QUERY,'route_radius':RADIUS},'methods':results}
    (out/'metrics.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
