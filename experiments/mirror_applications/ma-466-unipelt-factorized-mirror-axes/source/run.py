#!/usr/bin/env python3
"""Frozen MA-466 UniPELT-style component gate screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
from torch import nn
D,TASKS,LAYERS,C,R=4,16,8,3,2;UPDATES,BATCH,N=2500,32,64
METHODS=('unipelt','mirror','native_cp','shared')

def world(seed):
    rng=np.random.default_rng(seed+466);base=rng.normal(scale=.25,size=(LAYERS,D,D)).astype(np.float32)
    la=rng.normal(scale=.55,size=(LAYERS,D,1)).astype(np.float32);lb=rng.normal(scale=.55,size=(LAYERS,1,D)).astype(np.float32)
    au=rng.normal(scale=.4,size=(LAYERS,D,2)).astype(np.float32);av=rng.normal(scale=.4,size=(LAYERS,2,D)).astype(np.float32);prefix=rng.normal(scale=.3,size=(LAYERS,D)).astype(np.float32)
    ta=rng.normal(scale=.75,size=(TASKS,R)).astype(np.float32);lbasis=rng.normal(scale=.75,size=(LAYERS,R)).astype(np.float32);cb=rng.normal(scale=.75,size=(C,R)).astype(np.float32)
    logits=np.einsum('tr,lr,cr->tlc',ta,lbasis,cb);gates=1/(1+np.exp(-logits))
    comp={'base':torch.tensor(base),'lora_a':torch.tensor(la),'lora_b':torch.tensor(lb),'adapter_u':torch.tensor(au),'adapter_v':torch.tensor(av),'prefix':torch.tensor(prefix)}
    xtr=np.empty((TASKS,LAYERS,N,D),np.float32);ytr=np.empty_like(xtr);xq=np.empty_like(xtr);yq=np.empty_like(xtr)
    for t in range(TASKS):
        for l in range(LAYERS):
            for split,xbank,ybank,offset in [('train',xtr,ytr,1000),('query',xq,yq,90000)]:
                rr=np.random.default_rng(seed+offset+t*LAYERS+l);x=rr.normal(size=(N,D)).astype(np.float32);xt=torch.tensor(x);y=xt@comp['base'][l].T
                f0=(xt@comp['lora_a'][l])@comp['lora_b'][l]
                f1=torch.tanh(xt@comp['adapter_u'][l])@comp['adapter_v'][l]
                f2=comp['prefix'][l].expand(N,D)
                for c,f in enumerate((f0,f1,f2)):y=y+float(gates[t,l,c])*f
                y=y+torch.tensor(rr.normal(scale=.02,size=(N,D)).astype(np.float32));xbank[t,l]=x;ybank[t,l]=y.numpy()
    return comp,torch.tensor(gates,dtype=torch.float32),{k:torch.tensor(v) for k,v in {'xtr':xtr,'ytr':ytr,'xq':xq,'yq':yq}.items()},(ta,lbasis,cb)

def model(method,seed):
    if method=='unipelt':torch.manual_seed(seed+46601);return {'logits':nn.Parameter(torch.zeros(TASKS,LAYERS,C))}
    if method in ('mirror','native_cp'):
        torch.manual_seed(seed+46602);return {'task':nn.Parameter(torch.randn(TASKS,R)*.15),'layer':nn.Parameter(torch.randn(LAYERS,R)*.15),'component':nn.Parameter(torch.randn(C,R)*.15)}
    torch.manual_seed(seed+46603);return {'logits':nn.Parameter(torch.zeros(C))}

def gate_values(method,s,t,l):
    if method=='unipelt':return torch.sigmoid(s['logits'][t,l])
    if method in ('mirror','native_cp'):return torch.sigmoid(torch.einsum('r,cr,r->c',s['task'][t],s['component'],s['layer'][l]))
    return torch.sigmoid(s['logits'])

def components(x,comp,l):
    return (x@comp['lora_a'][l]@comp['lora_b'][l],torch.tanh(x@comp['adapter_u'][l])@comp['adapter_v'][l],comp['prefix'][l].expand(x.shape[0],D))

def predict(method,s,comp,t,l,x,ablate=None):
    gs=gate_values(method,s,t,l)
    if ablate is not None:gs=gs.clone();gs[ablate]=0
    fs=components(x,comp,l);y=x@comp['base'][l].T
    for c,f in enumerate(fs):y=y+gs[c]*f
    return y

def train(method,seed,comp,data):
    s=model(method,seed);opt=torch.optim.Adam(list(s.values()),lr=.01);rng=np.random.default_rng(seed+46611);start=time.perf_counter()
    for _ in range(UPDATES):
        t=int(rng.integers(TASKS));l=int(rng.integers(LAYERS));idx=rng.integers(N,size=BATCH);x=data['xtr'][t,l,idx];y=data['ytr'][t,l,idx];pred=predict(method,s,comp,t,l,x);loss=((pred-y)**2).mean();opt.zero_grad();loss.backward();opt.step()
    return {k:v.detach() for k,v in s.items()},time.perf_counter()-start

def arrays(method,s,comp):
    a={k:v.numpy() for k,v in comp.items()}
    if method=='unipelt':a['unipelt_gate_logits']=s['logits'].detach().numpy();fam='unipelt-full-component-gates-v1'
    elif method in ('mirror','native_cp'):a.update({'task_factors':s['task'].detach().numpy(),'layer_factors':s['layer'].detach().numpy(),'component_factors':s['component'].detach().numpy()});fam='cp-rank2-component-gates-v1'
    else:a['shared_gate_logits']=s['logits'].detach().numpy();fam='shared-global-component-gates-v1'
    return a,fam

def serialize(method,s,comp):
    a,fam=arrays(method,s,comp);buf=io.BytesIO();d={k:np.asarray(v) for k,v in a.items()};d['__schema_json__']=np.frombuffer(json.dumps({'format':'MA466-unipelt-component-bank-v1','method_family':fam,'dimension':D,'tasks':TASKS,'layers':LAYERS,'components':['lora','bottleneck_adapter','prefix'],'dtype':'float32'},sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);np.savez(buf,**d);return buf.getvalue()

def evaluate(method,s,comp,data,wall):
    start=time.perf_counter();scores=[];abl={c:[] for c in range(C)};active=[]
    with torch.no_grad():
        for t in range(TASKS):
            for l in range(LAYERS):
                x=data['xq'][t,l];y=data['yq'][t,l];p=predict(method,s,comp,t,l,x);scores.append(float(torch.sqrt(((p-y)**2).mean())));active.append(int((gate_values(method,s,t,l)>.1).sum()))
                if method=='mirror':
                    for c in range(C):pa=predict(method,s,comp,t,l,x,c);abl[c].append(float(torch.sqrt(((pa-y)**2).mean())))
    qw=time.perf_counter()-start;raw=serialize(method,s,comp);mac=16+8+16+4+3+(9 if method in ('mirror','native_cp') else 0)
    m={'mean_query_rmse':float(np.mean(scores)),'per_context_rmse':scores,'active_components_mean':float(np.mean(active)),'ablation_rmse':{str(k):float(np.mean(v)) for k,v in abl.items()} if method=='mirror' else None,'payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'optimizer_updates':UPDATES,'examples_seen':UPDATES*BATCH,'training_ops_proxy':UPDATES*BATCH*mac*3,'evaluation_ops_proxy':TASKS*LAYERS*N*mac,'active_ops_proxy':UPDATES*BATCH*mac*3+TASKS*LAYERS*N*mac,'inference_ops_proxy_per_example':mac,'training_wall_s':wall,'query_wall_s':qw,'query_examples':TASKS*LAYERS*N}
    return m,raw

def run(seed,out):
    out.mkdir(parents=True,exist_ok=True);comp,_,data,_=world(seed);results={};states={}
    for method in METHODS:
        s,wall=train(method,seed,comp,data);states[method]=s;m,raw=evaluate(method,s,comp,data,wall);results[method]=m;(out/f'{method}_payload.npz').write_bytes(raw)
    results['native_cp']['payload_sha256']==results['mirror']['payload_sha256'] or (_ for _ in ()).throw(AssertionError('native CP payload should alias Mirror'))
    assert results['native_cp']['per_context_rmse']==results['mirror']['per_context_rmse']
    d={'experiment_id':'MA-466','seed':seed,'split':'dev','task':{'dimension':D,'tasks':TASKS,'layers':LAYERS,'components':C,'support_per_context':N,'query_per_context':N},'methods':results};(out/'metrics.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
