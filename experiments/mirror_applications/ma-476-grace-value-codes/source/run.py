#!/usr/bin/env python3
"""Frozen MA-476 fixed-routing shared-basis plus latent VQ screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
D,KD,R,N,TRAIN=64,32,8,256,192
QPER,NLOCAL=4,1024
RADIUS=.35
METHODS=('serac_full','mirror_pca','native_pca','latent_int8','vq16','vq64','vq128','no_edit')

def world(seed):
    g=torch.Generator().manual_seed(seed+476);basis=torch.linalg.qr(torch.randn(D,R,generator=g),mode='reduced').Q
    coeff=torch.randn(N,R,generator=g)*.5;values=coeff@basis.T
    keys=torch.randn(N,KD,generator=g);keys=keys/keys.norm(dim=1,keepdim=True)
    qg=torch.Generator().manual_seed(seed+1476);queries=[keys[e]+torch.randn(QPER,KD,generator=qg)*.04 for e in range(N)]
    locality=torch.randn(NLOCAL,KD,generator=torch.Generator().manual_seed(seed+2476))
    return {'teacher_basis':basis,'coeff':coeff,'values':values,'keys':keys,'queries':queries,'locality':locality}

def route(q,keys):
    ds=((keys-q.unsqueeze(0))**2).sum(-1);i=int(torch.argmin(ds).item());return i,float(ds[i])<=RADIUS*RADIUS

def kmeans(x,k,seed):
    gen=torch.Generator().manual_seed(seed+k);idx=torch.randperm(x.shape[0],generator=gen)[:k];centers=x[idx].clone()
    for _ in range(50):
        labels=torch.cdist(x,centers).argmin(dim=1);new=centers.clone()
        for j in range(k):
            m=labels==j
            if bool(m.any()):new[j]=x[m].mean(0)
        centers=new
    return centers,torch.cdist(x,centers).argmin(dim=1)

def prepare(method,w,seed):
    t=time.perf_counter()
    if method in ('serac_full','no_edit'):return {'values':w['values'] if method=='serac_full' else torch.zeros_like(w['values'])},time.perf_counter()-t,0
    _,_,vh=torch.linalg.svd(w['values'][:TRAIN],full_matrices=False);basis=vh[:R].T.contiguous();z=w['values']@basis
    svd_ops=TRAIN*D*R*3
    if method in ('mirror_pca','native_pca'):state={'basis':basis,'codes':z};return state,time.perf_counter()-t,svd_ops
    if method=='latent_int8':
        scale=z[:TRAIN].abs().amax(0,keepdim=True).clamp_min(1e-8)/127;q=torch.round(z/scale).clamp(-127,127).to(torch.int8);return {'basis':basis,'q':q,'scale':scale},time.perf_counter()-t,svd_ops
    k=int(method[2:]);centers,labels=kmeans(z[:TRAIN],k,seed);all_labels=torch.cdist(z,centers).argmin(dim=1).to(torch.uint8)
    return {'basis':basis,'centers':centers,'labels':all_labels},time.perf_counter()-t,svd_ops+TRAIN*k*R*50*3

def reconstruct(method,w,state):
    if method=='serac_full':return state['values']
    if method=='no_edit':return torch.zeros_like(w['values'])
    if method in ('mirror_pca','native_pca'):return state['codes']@state['basis'].T
    if method=='latent_int8':return (state['q'].float()*state['scale'])@state['basis'].T
    return state['centers'][state['labels'].long()]@state['basis'].T

def pack(arr,meta):
    b=io.BytesIO();d={k:np.asarray(v) for k,v in arr.items()};d['__schema_json__']=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);np.savez(b,**d);return b.getvalue()

def serialize(method,w,state):
    obj={'keys':w['keys'].numpy(),'route_radius':np.asarray([RADIUS],np.float32)}
    if method in ('serac_full','no_edit'):
        if method=='serac_full':obj['values']=state['values'].numpy()
        family='explicit-value-bank-v1' if method=='serac_full' else 'no-edit-bank-v1'
    elif method in ('mirror_pca','native_pca'):
        obj.update({'value_basis':state['basis'].numpy(),'value_codes':state['codes'].numpy()});family='rank8-continuous-value-bank-v1'
    elif method=='latent_int8':obj.update({'value_basis':state['basis'].numpy(),'latent_int8':state['q'].numpy(),'latent_scales':state['scale'].numpy()});family='rank8-int8-latent-bank-v1'
    else:obj.update({'value_basis':state['basis'].numpy(),'vq_centers':state['centers'].numpy(),'vq_indices':state['labels'].numpy()});family='rank8-vq-value-bank-v1'
    return pack(obj,{'format':'MA476-fixed-route-value-bank-v1','method_family':family,'codebook_size':int(method[2:]) if method.startswith('vq') else None,'n_values':N,'value_dim':D,'latent_dim':R,'dtype':'typed-arrays'})

def evaluate(method,w,seed):
    state,prep_wall,fit_ops=prepare(method,w,seed);recon=reconstruct(method,w,state);raw=serialize(method,w,state)
    qstart=time.perf_counter();err=[];correct=[];false=[]
    for e in range(TRAIN,N):
        for q in w['queries'][e]:
            i,on=route(q,w['keys']);correct.append(i==e and on);value=recon[i] if on and method!='no_edit' else torch.zeros(D);err.append(float(torch.mean((value-w['values'][e])**2)))
    for q in w['locality']:false.append(route(q,w['keys'])[1])
    query_wall=time.perf_counter()-qstart;route_ops=N*KD*3
    decode=D*R if method in ('mirror_pca','native_pca','latent_int8') or method.startswith('vq') else D
    vqops=int(method[2:])*R*3 if method.startswith('vq') else 0
    m={'heldout_value_rmse':float(np.sqrt(np.mean(err))),'per_query_mse':err,'heldout_wrong_route_rate':1-float(np.mean(correct)),'locality_false_trigger_rate':float(np.mean(false)),'inference_payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'preprocess_wall_s':prep_wall,'query_wall_s':query_wall,'basis_fit_ops_proxy':fit_ops,'vq_fit_ops_proxy':TRAIN*int(method[2:])*R*50*3 if method.startswith('vq') else 0,'write_ops_proxy_per_edit':D*R if method in ('mirror_pca','native_pca','latent_int8') else (int(method[2:])*R if method.startswith('vq') else D),'decode_ops_proxy_per_query':decode,'vq_lookup_ops_proxy_per_query':vqops,'route_distance_scalar_ops_per_query':route_ops,'route_comparisons_per_query':N-1,'active_ops_proxy':((N-TRAIN)*QPER)*(route_ops+decode+vqops)+NLOCAL*route_ops}
    return m,raw

def run(seed,out):
    out.mkdir(parents=True,exist_ok=True);w=world(seed);result={}
    for method in METHODS:
        m,raw=evaluate(method,w,seed);result[method]=m;(out/f'{method}_payload.npz').write_bytes(raw)
    assert (out/'mirror_pca_payload.npz').read_bytes()==(out/'native_pca_payload.npz').read_bytes()
    doc={'experiment_id':'MA-476','seed':seed,'split':'dev','task':{'n_values':N,'train':TRAIN,'heldout':N-TRAIN,'value_dim':D,'latent_dim':R,'route_radius':RADIUS},'methods':result};(out/'metrics.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');return doc

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
