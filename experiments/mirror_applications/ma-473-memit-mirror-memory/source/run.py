#!/usr/bin/env python3
"""Frozen MA-473 layer-tagged multi-edit tensor factorization screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
from torch import nn
D,LAYERS,EDITS,TRAIN=12,3,10,8
SUPPORT,QUERY,NLOCAL=128,96,512
UPDATES,BATCH=1800,8
RADIUS=1.2
METHODS=('mirror','native_cp','memit_full','independent_fit','rank2_per_edit','no_edit')

def world(seed):
    rng=np.random.default_rng(seed+473)
    base=torch.tensor(rng.normal(scale=.18,size=(LAYERS,D,D)).astype(np.float32))
    basis=torch.tensor(rng.normal(scale=.05,size=(2,LAYERS,D,D)).astype(np.float32))
    codes=torch.tensor(rng.normal(scale=.3,size=(EDITS,2)).astype(np.float32))
    keys=torch.tensor(np.eye(D,dtype=np.float32)[:EDITS]);deltas=[];xs=[];ys=[];signals=[];xq=[];yq=[]
    for e in range(EDITS):
        delta=torch.einsum('r,rlij->lij',codes[e],basis);deltas.append(delta)
        ex=[];ey=[];es=[]
        rs=np.random.default_rng(seed+5000+e)
        for l in range(LAYERS):
            x=torch.tensor(keys[e].numpy()+rs.normal(scale=.28,size=(SUPPORT,D)).astype(np.float32))
            y=x@(base[l]+delta[l]).T+torch.tensor(rs.normal(scale=.01,size=(SUPPORT,D)).astype(np.float32))
            ex.append(x);ey.append(y);es.append((y-x@base[l].T).T@x/SUPPORT)
        xs.append(ex);ys.append(ey);signals.append(torch.stack(es).reshape(-1))
        rq=np.random.default_rng(seed+70000+e)
        x=torch.tensor(keys[e].numpy()+rq.normal(scale=.18,size=(QUERY,D)).astype(np.float32))
        xq.append(x);yq.append(torch.stack([x@(base[l]+delta[l]).T for l in range(LAYERS)],dim=1))
    xl=torch.tensor(np.random.default_rng(seed+90000).normal(size=(NLOCAL,D)).astype(np.float32))
    return {'base':base,'basis':basis,'codes':codes,'keys':keys,'deltas':torch.stack(deltas),'signals':torch.stack(signals),'xs':xs,'ys':ys,'xq':xq,'yq':yq,'xl':xl}

class CodeEditor(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(seed+47301);self.fc1=nn.Linear(LAYERS*D*D,32);self.fc2=nn.Linear(32,2)
    def forward(self,g):return self.fc2(torch.tanh(self.fc1(g.reshape(-1))))

def init(method,seed):
    if method in ('mirror','native_cp'):
        net=CodeEditor(seed);torch.manual_seed(seed+47302);basis=nn.Parameter(torch.randn(2,LAYERS,D,D)*.05)
        return {'net':net,'basis':basis}
    return {}

def decode(code,basis):return torch.einsum('r,rlij->lij',code,basis)
def train(method,seed,w):
    if method not in ('mirror','native_cp'):return None,0.
    s=init(method,seed);opt=torch.optim.Adam(list(s['net'].parameters())+[s['basis']],lr=.008);rng=np.random.default_rng(seed+47311);start=time.perf_counter()
    for _ in range(UPDATES):
        es=rng.integers(TRAIN,size=BATCH);loss=0.
        for e in es:loss=loss+((decode(s['net'](w['signals'][e]),s['basis'])-w['deltas'][e])**2).mean()
        loss=loss/BATCH;opt.zero_grad();loss.backward();opt.step()
    return {k:(v.detach() if isinstance(v,torch.Tensor) else v) for k,v in s.items()},time.perf_counter()-start

def route(x,keys):
    d=((keys-x.unsqueeze(0))**2).sum(-1);i=int(torch.argmin(d).item());return i,float(d[i])<=RADIUS*RADIUS

def support_controls(w):
    ind=[];rank=[]
    for e in range(EDITS):
        rs=[];fits=[]
        for l in range(LAYERS):
            x=w['xs'][e][l];y=w['ys'][e][l]
            fit=torch.linalg.lstsq(x,y-x@w['base'][l].T).solution.T.detach();fits.append(fit)
            u,sv,vh=torch.linalg.svd(fit,full_matrices=False);rr=torch.sqrt(torch.clamp(sv[:2],min=0))
            rs.append((u[:,:2]*rr[None,:],rr[:,None]*vh[:2,:]))
        ind.append(torch.stack(fits));rank.append(rs)
    return torch.stack(ind),rank

def rank_decode(factors):
    mats=[]
    for u,v in factors:mats.append(u@v)
    return torch.stack(mats)

def pack(arr,meta):
    b=io.BytesIO();d={k:np.asarray(v) for k,v in arr.items()};d['__schema_json__']=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);np.savez(b,**d);return b.getvalue()

def serialize(method,w,updates,codes=None,basis=None,factors=None):
    obj={'base':w['base'].numpy(),'edit_keys':w['keys'].numpy(),'route_radius':np.asarray([RADIUS],np.float32)}
    if method in ('mirror','native_cp'):
        obj.update({'shared_layer_basis':basis.numpy(),'edit_codes':codes.numpy()});family='rank2-fact-layer-code-bank-v1'
    elif method in ('memit_full','independent_fit'):obj.update({'layer_updates':updates.numpy()});family='full-layer-update-bank-v1'
    elif method=='rank2_per_edit':
        obj.update({'left_factors':np.stack([[u.numpy() for u,v in f] for f in factors]),'right_factors':np.stack([[v.numpy() for u,v in f] for f in factors])});family='per-edit-rank2-factor-bank-v1'
    else:family='frozen-no-edit-bank-v1'
    return pack(obj,{'format':'MA473-layer-edit-bank-v1','method_family':family,'dimension':D,'layers':LAYERS,'edits':EDITS,'dtype':'float32'})

def serialize_editor(s):
    return pack({'fc1_weight':s['net'].fc1.weight.detach().numpy(),'fc1_bias':s['net'].fc1.bias.detach().numpy(),'fc2_weight':s['net'].fc2.weight.detach().numpy(),'fc2_bias':s['net'].fc2.bias.detach().numpy()},{'format':'MA473-code-editor-v1','dtype':'float32'})

def evaluate(method,s,w,wall,ind,rank):
    start=time.perf_counter();codes=[];updates=[];factors=None
    with torch.no_grad():
        if method in ('mirror','native_cp'):
            for e in range(EDITS):
                c=s['net'](w['signals'][e]);codes.append(c);updates.append(decode(c,s['basis']))
        elif method=='memit_full':updates=list(w['deltas'])
        elif method=='independent_fit':updates=list(ind)
        elif method=='rank2_per_edit':factors=rank;updates=[rank_decode(f) for f in factors]
        else:updates=[torch.zeros(LAYERS,D,D) for _ in range(EDITS)]
    genwall=time.perf_counter()-start;codes_t=torch.stack(codes) if codes else None;updates_t=torch.stack(updates)
    bank=serialize(method,w,updates_t,codes_t,s['basis'] if s else None,factors)
    editor=serialize_editor(s) if method in ('mirror','native_cp') else b''
    qstart=time.perf_counter();edits=[];routed=[];local=[];localroutes=[]
    with torch.no_grad():
        for e in range(TRAIN,EDITS):
            for x,y in zip(w['xq'][e],w['yq'][e]):
                i,on=route(x,w['keys']);routed.append(i==e and on);delta=updates_t[i] if on else torch.zeros(LAYERS,D,D)
                pred=torch.stack([x@(w['base'][l]+delta[l]).T for l in range(LAYERS)],dim=0);edits.append(float(torch.mean((pred-y)**2)))
        for x in w['xl']:
            i,on=route(x,w['keys']);localroutes.append(on);delta=updates_t[i] if on else torch.zeros(LAYERS,D,D)
            pred=torch.stack([x@(w['base'][l]+delta[l]).T for l in range(LAYERS)],dim=0);base=torch.stack([x@w['base'][l].T for l in range(LAYERS)],dim=0);local.append(float(torch.mean((pred-base)**2)))
    querywall=time.perf_counter()-qstart
    genmac=LAYERS*D*D*32+32*2+LAYERS*D*D*2 if method in ('mirror','native_cp') else LAYERS*D*D
    qops=LAYERS*D*D+EDITS*D*3
    trainops=UPDATES*BATCH*genmac*3 if method in ('mirror','native_cp') else EDITS*LAYERS*SUPPORT*D*D
    m={'heldout_edit_rmse':float(np.sqrt(np.mean(edits))),'per_query_edit_mse':edits,'locality_rmse':float(np.sqrt(np.mean(local))),'wrong_route_rate':1-float(np.mean(routed)),'locality_route_rate':float(np.mean(localroutes)),'inference_payload_bytes':len(bank),'editor_payload_bytes':len(editor),'full_online_edit_system_bytes':len(bank)+len(editor),'payload_sha256':hashlib.sha256(bank).hexdigest(),'editor_sha256':hashlib.sha256(editor).hexdigest() if editor else None,'optimizer_updates':UPDATES if method in ('mirror','native_cp') else 0,'support_examples_seen':UPDATES*BATCH if method in ('mirror','native_cp') else EDITS*LAYERS*SUPPORT,'training_wall_s':wall,'code_generation_wall_s':genwall,'query_wall_s':querywall,'code_generation_ops_per_edit':genmac,'query_ops_proxy_per_example':qops,'route_distance_scalar_ops_per_example':EDITS*D*3,'active_ops_proxy':trainops+EDITS*genmac+((EDITS-TRAIN)*QUERY+NLOCAL)*qops}
    return m,bank,editor

def run(seed,out):
    out.mkdir(parents=True,exist_ok=True);w=world(seed);ind,rank=support_controls(w);result={}
    for method in METHODS:
        s,wall=train(method,seed,w);m,b,e=evaluate(method,s,w,wall,ind,rank);result[method]=m;(out/f'{method}_bank.npz').write_bytes(b)
        if e:(out/f'{method}_editor.npz').write_bytes(e)
    assert (out/'mirror_bank.npz').read_bytes()==(out/'native_cp_bank.npz').read_bytes()
    assert result['mirror']['heldout_edit_rmse']==result['native_cp']['heldout_edit_rmse']
    doc={'experiment_id':'MA-473','seed':seed,'split':'dev','task':{'dimension':D,'layers':LAYERS,'edits':EDITS,'train_edits':TRAIN,'support_per_edit_layer':SUPPORT,'route_radius':RADIUS},'methods':result};(out/'metrics.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');return doc

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
