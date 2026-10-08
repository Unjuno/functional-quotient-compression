#!/usr/bin/env python3
"""Frozen MA-475 fixed-retrieval external value memory compression screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
D,KD,RANK,N=64,32,8,256
TRAIN,HELD=192,64
QPER,NLOCAL=4,1024
RADIUS=.35
METHODS=('serac_full','mirror_pca','native_pca','int8_values','no_edit')

def world(seed):
    g=torch.Generator().manual_seed(seed+475)
    raw=torch.randn(D,RANK,generator=g);basis=torch.linalg.qr(raw,mode='reduced').Q
    coeff=torch.randn(N,RANK,generator=g)*.5;values=coeff@basis.T
    keys=torch.randn(N,KD,generator=g);keys=keys/keys.norm(dim=1,keepdim=True)
    qgen=torch.Generator().manual_seed(seed+1475);queries=[]
    for e in range(N):queries.append(keys[e]+torch.randn(QPER,KD,generator=qgen)*.04)
    loc=torch.randn(NLOCAL,KD,generator=torch.Generator().manual_seed(seed+2475))
    return {'basis_teacher':basis,'coeff':coeff,'values':values,'keys':keys,'queries':queries,'locality':loc}

def route(q,keys):
    d=((keys-q.unsqueeze(0))**2).sum(-1);i=int(torch.argmin(d).item());return i,float(d[i])<=RADIUS*RADIUS

def make_values(method,w):
    if method=='serac_full':return w['values'],None,None
    if method in ('mirror_pca','native_pca'):
        _,_,vh=torch.linalg.svd(w['values'][:TRAIN],full_matrices=False);basis=vh[:RANK].T.contiguous();codes=w['values']@basis
        return codes@basis.T,basis,codes
    if method=='int8_values':
        scale=w['values'].abs().amax(dim=1,keepdim=True).clamp_min(1e-8)/127
        q=torch.round(w['values']/scale).clamp(-127,127).to(torch.int8);return q.float()*scale,None,(q,scale)
    return torch.zeros_like(w['values']),None,None

def pack(arr,meta):
    b=io.BytesIO();d={k:np.asarray(v) for k,v in arr.items()};d['__schema_json__']=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);np.savez(b,**d);return b.getvalue()

def serialize(method,w,basis,code_state):
    obj={'keys':w['keys'].numpy(),'route_radius':np.asarray([RADIUS],np.float32)}
    if method in ('serac_full','no_edit'):obj.update({'values':(w['values'] if method=='serac_full' else torch.zeros_like(w['values'])).numpy()});family='explicit-serac-value-bank-v1' if method=='serac_full' else 'no-edit-bank-v1'
    elif method in ('mirror_pca','native_pca'):obj.update({'value_basis':basis.numpy(),'value_codes':code_state.numpy()});family='rank8-shared-pca-value-bank-v1'
    else:
        q,scale=code_state;obj.update({'value_int8':q.numpy(),'value_scales':scale.numpy()});family='per-value-symmetric-int8-v1'
    return pack(obj,{'format':'MA475-external-counterfactual-memory-v1','method_family':family,'n_values':N,'value_dim':D,'key_dim':KD,'dtype':'float32+declared-value-dtype'})

def evaluate(method,w):
    fit_start=time.perf_counter();recon,basis,state=make_values(method,w);fit_wall=time.perf_counter()-fit_start
    pack_start=time.perf_counter();raw=serialize(method,w,basis,state);pack_wall=time.perf_counter()-pack_start
    qstart=time.perf_counter();errs=[];correct=[]
    for e in range(TRAIN,N):
        for q in w['queries'][e]:
            i,on=route(q,w['keys']);correct.append(i==e and on);value=recon[i] if on and method!='no_edit' else torch.zeros(D)
            errs.append(float(torch.mean((value-w['values'][e])**2)))
    locroute=[]
    for q in w['locality']:locroute.append(route(q,w['keys'])[1])
    query_wall=time.perf_counter()-qstart
    rmse=float(np.sqrt(np.mean(errs)));locality_rate=float(np.mean(locroute))
    route_ops=N*KD*3
    decode_ops=D*RANK if method in ('mirror_pca','native_pca') else D
    m={'heldout_value_rmse':rmse,'per_query_value_mse':errs,'heldout_wrong_route_rate':1-float(np.mean(correct)),'locality_false_trigger_rate':locality_rate,'inference_payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'basis_fit_wall_s':fit_wall,'serialization_wall_s':pack_wall,'query_wall_s':query_wall,'basis_fit_ops_proxy':int(TRAIN*D*RANK*3) if method in ('mirror_pca','native_pca') else 0,'write_ops_proxy_per_edit':D*RANK if method in ('mirror_pca','native_pca') else D,'decode_ops_proxy_per_query':decode_ops,'route_distance_scalar_ops_per_query':route_ops,'route_comparisons_per_query':N-1,'active_ops_proxy':(HELD*QPER)*(route_ops+decode_ops)+NLOCAL*route_ops,'optimizer_updates':0,'examples_seen':N*QPER+NLOCAL}
    return m,raw

def run(seed,out):
    out.mkdir(parents=True,exist_ok=True);w=world(seed);res={}
    for method in METHODS:
        m,raw=evaluate(method,w);res[method]=m;(out/f'{method}_payload.npz').write_bytes(raw)
    assert (out/'mirror_pca_payload.npz').read_bytes()==(out/'native_pca_payload.npz').read_bytes()
    assert res['mirror_pca']['heldout_value_rmse']==res['native_pca']['heldout_value_rmse']
    doc={'experiment_id':'MA-475','seed':seed,'split':'dev','task':{'n_values':N,'basis_train':TRAIN,'heldout':HELD,'value_dim':D,'key_dim':KD,'rank':RANK,'route_radius':RADIUS},'methods':res};(out/'metrics.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');return doc

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
