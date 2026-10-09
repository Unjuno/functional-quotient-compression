#!/usr/bin/env python3
"""MA-599 factorized hash-bucket expert Views with router, Mirror views, and controls."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

INPUT,HIDDEN,OUTPUT,N_EXPERTS=64,128,10,4
BUCKETS,UPDATES,BATCH,LR=2048,800,128,0.01
METHODS=('dense_independent','independent_hash_tables','tied_shared_hash','salted_shared_hash','factorized_hash_view','diagonal_gate','rank4_dense_residual')


def dataset():
    ds=load_digits(); raw=np.ascontiguousarray(ds.data.astype(np.float32)); y=np.ascontiguousarray(ds.target.astype(np.int64))
    x=raw/16.0; digest=hashlib.sha256(raw.tobytes()+y.tobytes()).hexdigest()
    return x,y,digest


def hash_map(bucket_count,seed):
    rng=np.random.default_rng(int(seed)&0xFFFFFFFF)
    idx=rng.integers(0,bucket_count,size=(INPUT,HIDDEN),dtype=np.int64)
    sign=rng.choice(np.asarray([-1.,1.],dtype=np.float32),size=(INPUT,HIDDEN))
    return idx,sign


def collision_stats(idx):
    counts=np.bincount(idx.reshape(-1),minlength=BUCKETS);used=int(np.count_nonzero(counts));n=idx.size
    return {'unique_bucket_count':used,'collision_rate':1-used/n,'max_bucket_load':int(counts.max()),
            'collision_pair_count':int(np.sum(counts*(counts-1)//2))}


def givens_rotate(x,angles):
    a,b=x[...,0::2],x[...,1::2];c,s=torch.cos(angles),torch.sin(angles)
    out=torch.empty_like(x);out[...,0::2]=c*a-s*b;out[...,1::2]=s*a+c*b
    return out


def method_salt_mode(method):return 1 if method in ('salted_shared_hash','independent_hash_tables') else 0

def maps_for(method,hash_salts):
    salts=np.asarray(hash_salts,dtype=np.uint32).reshape(-1)
    return [hash_map(BUCKETS,int(salts[e] if len(salts)==N_EXPERTS else salts[0])) for e in range(N_EXPERTS)]


def init_params(method,router_seed,weight_seed):
    # The router starts identically across methods and sees the same batches/labels.
    rr=np.random.default_rng(router_seed+1000003)
    p={'router_w':torch.nn.Parameter(torch.as_tensor(rr.normal(0,.02,(INPUT,N_EXPERTS)),dtype=torch.float32)),
       'router_b':torch.nn.Parameter(torch.zeros(N_EXPERTS))}
    torch.manual_seed(weight_seed*1009)
    if method=='dense_independent':
        p['w1']=torch.nn.Parameter(torch.empty(N_EXPERTS,INPUT,HIDDEN));p['b1']=torch.nn.Parameter(torch.zeros(N_EXPERTS,HIDDEN))
        p['w2']=torch.nn.Parameter(torch.empty(N_EXPERTS,HIDDEN,OUTPUT));p['b2']=torch.nn.Parameter(torch.zeros(N_EXPERTS,OUTPUT))
        for e in range(N_EXPERTS):
            torch.nn.init.kaiming_uniform_(p['w1'][e].T,a=0.,nonlinearity='relu');torch.nn.init.xavier_uniform_(p['w2'][e].T)
    else:
        ntable=N_EXPERTS if method=='independent_hash_tables' else 1
        p['theta']=torch.nn.Parameter(torch.randn(ntable,BUCKETS)*(2/INPUT)**.5)
        p['b1']=torch.nn.Parameter(torch.zeros(HIDDEN));p['w2']=torch.nn.Parameter(torch.empty(HIDDEN,OUTPUT));p['b2']=torch.nn.Parameter(torch.zeros(OUTPUT))
        torch.nn.init.xavier_uniform_(p['w2'].T)
        if method=='factorized_hash_view':
            p['bucket_basis']=torch.nn.Parameter(torch.randn(BUCKETS,4)*.01);p['expert_code']=torch.nn.Parameter(torch.zeros(N_EXPERTS,4))
        elif method=='diagonal_gate':p['scale']=torch.nn.Parameter(torch.ones(N_EXPERTS,INPUT))
        elif method=='rank4_dense_residual':
            p['u']=torch.nn.Parameter(torch.randn(N_EXPERTS,INPUT,4)*.01);p['v']=torch.nn.Parameter(torch.randn(N_EXPERTS,4,HIDDEN)*.01)
    return p


def expert_w1(p,method,e,maps):
    if method=='dense_independent':return p['w1'][e]
    table=e if method=='independent_hash_tables' else 0
    idx,sg=maps[e];ix=torch.as_tensor(idx);sign=torch.as_tensor(sg)
    w=(p['theta'][table][ix]*sign).reshape(INPUT,HIDDEN)
    if method=='factorized_hash_view':
        effective=p['theta'][0]+p['bucket_basis']@p['expert_code'][e]
        w=(effective[ix]*torch.as_tensor(sg)).reshape(INPUT,HIDDEN)
    if method=='rank4_dense_residual':w=w+p['u'][e]@p['v'][e]
    return w


def raw_expert_logits(p,method,e,x,maps):
    w=expert_w1(p,method,e,maps)
    if method=='dense_independent':
        h=F.relu(x@w+p['b1'][e]);return h@p['w2'][e]+p['b2'][e]
    if method=='diagonal_gate':x=x*p['scale'][e]
    h=F.relu(x@w+p['b1']);return h@p['w2']+p['b2']


def mask_logits(logits,e):
    allowed=[k for k in range(OUTPUT) if k%N_EXPERTS==e]
    out=torch.full_like(logits,-1e9);out[:,allowed]=logits[:,allowed]
    return out


def train(method,xtr,ytr,seed,mi,hash_seed):
    p=init_params(method,seed,seed*100+mi)
    salts=[(hash_seed+e+1)&0xFFFFFFFF for e in range(N_EXPERTS)] if method_salt_mode(method) else [hash_seed]
    maps=maps_for(method,salts)
    opt=torch.optim.AdamW(list(p.values()),lr=LR,weight_decay=0.)
    rng=np.random.default_rng(seed*1000)
    xt=torch.as_tensor(xtr,dtype=torch.float32);yt=torch.as_tensor(ytr,dtype=torch.long);groups=yt%N_EXPERTS
    start=time.perf_counter();trace=[]
    for step in range(UPDATES):
        bi=rng.integers(0,len(xt),size=BATCH);xb,yb=xt[bi],yt[bi];gb=groups[bi]
        loss=F.cross_entropy(xb@p['router_w']+p['router_b'],gb)
        for e in range(N_EXPERTS):
            sel=gb==e
            ex=mask_logits(raw_expert_logits(p,method,e,xb[sel],maps),e)
            loss=loss+F.cross_entropy(ex,yb[sel])
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step in (0,UPDATES-1):trace.append(float(loss.detach()))
    return p,maps,time.perf_counter()-start,trace


def state_arrays(p,method,seed,hash_seed):
    salts=[(hash_seed+e+1)&0xFFFFFFFF for e in range(N_EXPERTS)] if method_salt_mode(method) else [hash_seed]
    a={k:v.detach().cpu().numpy().astype(np.float16) for k,v in p.items()}
    a.update({'method_ascii':np.frombuffer(method.encode('ascii'),dtype=np.uint8),
              'bucket_count':np.asarray([BUCKETS],np.int32),'hash_salts':np.asarray(salts,np.uint32),
              'salt_mode':np.asarray([method_salt_mode(method)],np.uint8),'run_seed':np.asarray([seed],np.uint32),
              'shape':np.asarray([N_EXPERTS,INPUT,HIDDEN,OUTPUT],np.int32),'schema':np.asarray([599,1],np.int32)})
    return a


def save_payload(path,p,method,seed,hash_seed):
    path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**state_arrays(p,method,seed,hash_seed));return path.stat().st_size


def load_payload(path):
    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    method=bytes(a.pop('method_ascii').tolist()).decode('ascii');seed=int(a.pop('run_seed')[0]);salts=a.pop('hash_salts');a.pop('bucket_count');a.pop('salt_mode');a.pop('shape');a.pop('schema')
    p={k:torch.as_tensor(v,dtype=torch.float32) for k,v in a.items()};maps=maps_for(method,salts)
    return method,seed,p,maps


def evaluate(path,xte,yte):
    method,seed,p,maps=load_payload(path);torch.set_num_threads(1)
    x=torch.as_tensor(xte,dtype=torch.float32);y=torch.as_tensor(yte,dtype=torch.long);trueg=y%N_EXPERTS
    router_logits=x@p['router_w']+p['router_b'];route=router_logits.argmax(1)
    predlog=torch.full((len(x),OUTPUT),-1e9);oraclelog=torch.full_like(predlog,-1e9)
    start=time.perf_counter()
    with torch.inference_mode():
        for e in range(N_EXPERTS):
            predsel=route==e;oraclesel=trueg==e
            if predsel.any():
                raw=raw_expert_logits(p,method,e,x[predsel],maps)
                predlog[predsel]=mask_logits(raw,e)
        pred=predlog.argmax(1)
    infer_s=time.perf_counter()-start
    # Oracle and diversity/ablation audits are offline diagnostics, excluded from top-1 throughput.
    start_diag=time.perf_counter();raw_all=[]
    with torch.inference_mode():
        for e in range(N_EXPERTS):
            raw=raw_expert_logits(p,method,e,x,maps);raw_all.append(raw)
            oraclesel=trueg==e
            if oraclesel.any():oraclelog[oraclesel]=mask_logits(raw[oraclesel],e)
        oracle=oraclelog.argmax(1)
        # Hierarchical CE: router probability of the true group plus conditional class probability within that group.
        router_nll=F.cross_entropy(router_logits,trueg,reduction='none')
        cond_nll=torch.zeros(len(x))
        for e in range(N_EXPERTS):
            sel=trueg==e;allowed=[k for k in range(OUTPUT) if k%N_EXPERTS==e]
            cond_nll[sel]=F.cross_entropy(raw_all[e][sel][:,allowed],y[sel]//N_EXPERTS,reduction='none')
        nll=float((router_nll+cond_nll).mean());oracle_nll=float(cond_nll.mean())
        acc=float((pred==y).float().mean());oracle_acc=float((oracle==y).float().mean())
        route_acc=float((route==trueg).float().mean())
        uses=torch.bincount(route,minlength=N_EXPERTS).cpu().numpy()/len(x)
        per_expert=[]
        for e in range(N_EXPERTS):
            sel=trueg==e;per_expert.append(float((oracle[sel]==y[sel]).float().mean()))
        pair_cos=[];pair_pred_disagree=[]
        for i in range(N_EXPERTS):
            for j in range(i+1,N_EXPERTS):
                ai=raw_all[i].reshape(-1);aj=raw_all[j].reshape(-1)
                pair_cos.append(float(F.cosine_similarity(ai[None],aj[None]).item()))
                pair_pred_disagree.append(float((raw_all[i].argmax(1)!=raw_all[j].argmax(1)).float().mean()))
        # Expert ablation: replace the router's top choice with its runner-up when that expert is removed.
        route2=router_logits.argsort(dim=1,descending=True)[:,:2]
        ablation_acc=[]
        for disabled in range(N_EXPERTS):
            alt=route.clone();mask=route==disabled;alt[mask]=route2[mask,1]
            ablated=torch.full((len(x),OUTPUT),-1e9)
            for e in range(N_EXPERTS):
                sel=alt==e
                if sel.any():ablated[sel]=mask_logits(raw_all[e][sel],e)
            ablation_acc.append(float((ablated.argmax(1)==y).float().mean()))
    diag_s=time.perf_counter()-start_diag
    weights=[]
    for e in range(N_EXPERTS):
        w=expert_w1(p,method,e,maps)
        if method=='factorized_hash_view':
            ix,sg=maps[e];eff=p['theta'][0]+p['bucket_basis']@p['expert_code'][e];w=(eff[torch.as_tensor(ix)]*torch.as_tensor(sg)).reshape(INPUT,HIDDEN)
        elif method=='diagonal_gate':w=p['scale'][e][:,None]*w
        weights.append(w.detach().numpy())
    wcos=[]
    for i in range(N_EXPERTS):
        for j in range(i+1,N_EXPERTS):
            wcos.append(float(np.dot(weights[i].ravel(),weights[j].ravel())/(np.linalg.norm(weights[i])*np.linalg.norm(weights[j])+1e-12)))
    collision=[collision_stats(maps[e][0]) for e in range(N_EXPERTS)]
    overlap=[]
    for i in range(N_EXPERTS):
        for j in range(i+1,N_EXPERTS):overlap.append(float(np.mean(maps[i][0]==maps[j][0])))
    return {'router_selected_accuracy':acc,'router_selected_cross_entropy':nll,'oracle_expert_accuracy':oracle_acc,'oracle_expert_cross_entropy':oracle_nll,
            'router_accuracy':route_acc,'expert_route_fractions':uses.tolist(),'per_expert_oracle_accuracy':per_expert,
            'pairwise_raw_logit_cosine_mean':float(np.mean(pair_cos)),'pairwise_raw_argmax_disagreement_mean':float(np.mean(pair_pred_disagree)),
            'pairwise_first_layer_weight_cosine_mean':float(np.mean(wcos)),'pairwise_hash_map_overlap_mean':float(np.mean(overlap)),
            'per_expert_collision':collision,'inference_seconds':infer_s,'inference_examples_per_second':len(x)/infer_s,
            'expert_ablation_accuracy':ablation_acc,'expert_diagnostic_seconds':diag_s,
            'active_experts_per_example':1,'route_confusion':_confusion(route.cpu().numpy(),trueg.cpu().numpy())}


def _confusion(pred,true):
    cm=np.zeros((N_EXPERTS,N_EXPERTS),np.int64)
    for p,t in zip(pred,true):cm[int(t),int(p)]+=1
    return cm.tolist()


def run(seed,split,outdir):
    torch.set_num_threads(1);x,y,digest=dataset()
    if digest!='faf2d98f61250c1cdc0b114323133d3c58da04f5c5d28bb78e4bde3c936a75b1':raise ValueError('digits dataset hash mismatch')
    xtr,xte,ytr,yte=train_test_split(x,y,test_size=.25,random_state=seed,stratify=y)
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True);hash_seed=(seed*7919+120)&0xFFFFFFFF
    rows=[];total_start=time.perf_counter()
    for mi,method in enumerate(METHODS):
        p,maps,train_s,trace=train(method,xtr,ytr,seed,mi,hash_seed)
        payload=outdir/f'{method}.npz';nbytes=save_payload(payload,p,method,seed,hash_seed);ev=evaluate(payload,xte,yte)
        dense_macs=INPUT*HIDDEN+HIDDEN*OUTPUT+INPUT*N_EXPERTS
        extra_ops=0
        if method=='factorized_hash_view':extra_ops=4*BUCKETS+4
        elif method=='diagonal_gate':extra_ops=INPUT
        elif method=='rank4_dense_residual':extra_ops=4*(INPUT+HIDDEN)
        lookups=INPUT*HIDDEN*(N_EXPERTS if method=='independent_hash_tables' else 1)
        rows.append({'method':method,'serialized_bytes':nbytes,'payload_sha256':hashlib.sha256(payload.read_bytes()).hexdigest(),
                     'train_examples':len(xtr),'test_examples':len(xte),'optimizer_updates':UPDATES,'examples_seen':UPDATES*BATCH,
                     'initial_and_final_train_loss':trace,'training_seconds':train_s,'base_macs_per_example':dense_macs,
                     'extra_view_ops_per_example':extra_ops,'hash_expansion_lookups_per_model_load':lookups,**ev})
    report={'experiment_id':'MA-599','seed':seed,'split':split,'dataset':'sklearn.load_digits','sklearn_version':sklearn.__version__,
            'dataset_tensor_sha256':digest,'split_random_state':seed,'stratified':True,'architecture':[N_EXPERTS,INPUT,HIDDEN,OUTPUT],
            'hash_seed':hash_seed,'methods':rows,'common_compute':{'device':'cpu','torch_threads':1,'updates_per_method':UPDATES,
            'input_examples_seen_per_method':UPDATES*BATCH,'total_wall_seconds':time.perf_counter()-total_start}}
    (outdir/'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['dev','fresh'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.seed,a.split,a.out)
if __name__=='__main__':main()
