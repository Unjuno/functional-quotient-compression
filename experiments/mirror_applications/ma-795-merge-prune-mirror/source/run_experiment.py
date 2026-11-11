"""Synthetic overcomplete-expert merge/prune and rank-1 role View screen."""
import csv, hashlib, io, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[1]
SEED=79501; E=8; C=2; D=12; N=128; R=1

def pack(**arrays):
    b=io.BytesIO(); np.savez_compressed(b,**arrays); return b.getvalue()

def nrmse(pred, target, weights=None):
    err=(pred-target)**2; ref=target**2
    if weights is not None:
        ww=np.asarray(weights).reshape(err.shape[:-1]+(1,))
        return float(np.sqrt(np.sum(err*ww)/(np.sum(ref*ww)+1e-12)))
    return float(np.sqrt(np.mean(err)/(np.mean(ref)+1e-12)))

def run():
    rng=np.random.default_rng(SEED)
    cores=rng.normal(0,.12,(C,D,D)).astype(np.float32)
    basis=rng.normal(0,.035,(D,D)).astype(np.float32); basis/=np.linalg.norm(basis)
    coeff=np.array([-.8,-.35,.35,.8,-.75,-.3,.3,.75],np.float32)
    teacher=np.stack([cores[e//4]+coeff[e]*basis for e in range(E)])
    # Small private deviations on the least frequently used role in each cluster.
    for e in (3,7): teacher[e]+=rng.normal(0,.012,(D,D)).astype(np.float32)
    importance=np.tile(np.array([.4,.3,.2,.1],np.float32),C)
    centers=np.zeros((E,D),np.float32)
    for e in range(E):
        centers[e,0]=(-2.5 if e<4 else 2.5); centers[e,1]=(e%4-1.5)*1.2
    xs=[]; ys=[]; xt=[]; yt=[]
    for e in range(E):
        x=rng.normal(0,.35,(N,D)).astype(np.float32)+centers[e]
        z=rng.normal(0,.35,(N,D)).astype(np.float32)+centers[e]
        xs.append(x); ys.append(x@teacher[e].T); xt.append(z); yt.append(z@teacher[e].T)
    xtrain=np.stack(xs); ytrain=np.stack(ys); xtest=np.stack(xt); ytest=np.stack(yt)
    # Fit each overcomplete expert from its permitted development examples (closed-form linear least squares).
    fitted=[]
    for e in range(E):
        a=np.concatenate([xtrain[e],np.ones((N,1),np.float32)],axis=1)
        beta=np.linalg.lstsq(a,ytrain[e],rcond=None)[0]
        fitted.append(beta[:-1].T.astype(np.float32))
    fitted=np.stack(fitted)
    merged=np.stack([np.average(fitted[c*4:(c+1)*4],axis=0,weights=importance[c*4:(c+1)*4]) for c in range(C)])
    residual=fitted-merged[np.arange(E)//4]
    # Shared rank-1 residual basis from overcomplete, routing-weighted merged experts.
    weighted=residual*np.sqrt(importance[:,None,None])
    _,_,vt=np.linalg.svd(weighted.reshape(E,-1),full_matrices=False)
    view_basis=vt[0].reshape(D,D).astype(np.float32)
    view_codes=np.einsum('eij,ij->e',residual,view_basis).astype(np.float32)
    view_recon=merged[np.arange(E)//4]+view_codes[:,None,None]*view_basis
    # PA209-like pruning: retain two highest importance experts per cluster; map pruned roles to closest retained expert.
    retained=[]; rolemap=np.zeros(E,np.int16)
    for c in range(C):
        group=np.arange(c*4,(c+1)*4); keep=group[np.argsort(-importance[group])[:2]]; retained.extend(keep.tolist())
        for e in group: rolemap[e]=keep[np.argmin([np.linalg.norm(fitted[e]-fitted[k]) for k in keep])]
    retained=np.asarray(retained,np.int16); prune_weights=fitted[rolemap]
    cluster_centers=np.stack([np.average(centers[c*4:(c+1)*4],axis=0,weights=importance[c*4:(c+1)*4]) for c in range(C)])
    cluster_route=np.argmin(((centers[:,None]-cluster_centers[None])**2).sum(-1),axis=1)
    # Small-from-start control is a two-expert fit from pooled cluster examples, without the overcomplete role bank.
    small=np.zeros_like(merged)
    for c in range(C):
        xx=xtrain[c*4:(c+1)*4].reshape(-1,D); yy=ytrain[c*4:(c+1)*4].reshape(-1,D)
        beta=np.linalg.lstsq(np.concatenate([xx,np.ones((len(xx),1),np.float32)],axis=1),yy,rcond=None)[0]
        small[c]=beta[:-1].T
    small_route=cluster_route
    router_role_centers=centers.copy()
    methods=[]
    router_role=np.broadcast_to(router_role_centers,(E,E,D)).copy()
    payloads=[
      ('independent_overcomplete',fitted,pack(experts=fitted,router_centers=router_role_centers,importance=importance,role_ids=np.arange(E,dtype=np.int16)),0,router_role_centers),
      ('prune_only',prune_weights,pack(experts=fitted[retained],router_centers=router_role_centers,retained_ids=retained,role_map=rolemap,importance=importance),0,router_role_centers),
      ('merge_only',merged[np.arange(E)//4],pack(experts=merged,router_centers=cluster_centers,role_to_cluster=np.arange(E,dtype=np.int16)//4,importance=importance),0,cluster_centers),
      ('small_from_start',small[np.arange(E)//4],pack(experts=small,router_centers=cluster_centers,importance=importance),0,cluster_centers),
      ('merge_plus_rank1_view',view_recon,pack(merged_experts=merged,view_basis=view_basis,view_codes=view_codes,router_centers=router_role_centers,role_to_cluster=np.arange(E,dtype=np.int16)//4,importance=importance),D*D*E,router_role_centers),
      ('merge_plus_native_rank1',view_recon,pack(merged_experts=merged,view_basis=view_basis,view_codes=view_codes,router_centers=router_role_centers,role_to_cluster=np.arange(E,dtype=np.int16)//4,importance=importance),D*D*E,router_role_centers)]
    weights=np.tile(importance[:,None],(1,N)).reshape(-1)
    for name,ws,payload,decode,router in payloads:
        if name in ('merge_plus_rank1_view','merge_plus_native_rank1'):
            loaded=np.load(io.BytesIO(payload)); restored=loaded['merged_experts'][np.arange(E)//4]+loaded['view_codes'][:,None,None]*loaded['view_basis']
        elif name=='independent_overcomplete': restored=np.load(io.BytesIO(payload))['experts']
        elif name=='prune_only':
            loaded=np.load(io.BytesIO(payload)); restored=loaded['experts'][np.searchsorted(loaded['retained_ids'],loaded['role_map'])]
        else:
            loaded=np.load(io.BytesIO(payload)); restored=loaded['experts'][np.arange(E)//4]
        rt=float(np.max(np.abs(restored-ws)))
        if rt>1e-5: raise AssertionError(f'{name} payload roundtrip {rt}')
        start=time.perf_counter()
        # Role ID is recoverable by nearest role centroid; merge/small controls route at cluster granularity.
        pred=[]
        for e in range(E):
            if name in ('merge_only','small_from_start'):
                route=int(small_route[e]); w=ws[route]
            else:
                route=e if name!='prune_only' else e
                w=ws[route]
            pred.append(xtest[e]@w.T)
        elapsed=time.perf_counter()-start; pred=np.stack(pred)
        perrole=[nrmse(pred[e],ytest[e]) for e in range(E)]
        methods.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),
          weighted_test_output_nrmse=nrmse(pred,ytest,weights.reshape(E,N)),mean_per_role_nrmse=float(np.mean(perrole)),max_per_role_nrmse=float(np.max(perrole)),
          active_macs_per_example=D*D+(len(router)*D),reconstruction_macs_total=decode,train_examples=E*N,test_examples=E*N,optimizer_updates=0,
          wall_time_s=elapsed,payload_roundtrip_max_abs=rt,retained_expert_count=len(retained) if name=='prune_only' else (C if name in ('merge_only','small_from_start','merge_plus_rank1_view','merge_plus_native_rank1') else E),
          status_note='exact native low-rank alias' if name=='merge_plus_rank1_view' else ''))
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=methods[0].keys(),lineterminator='\n');w.writeheader();w.writerows(methods)
    return methods

if __name__=='__main__':
    for row in run(): print(row)
