"""Closed-form RegMean and Mirror-coordinate operators for MA-715."""
from __future__ import annotations
import hashlib, io
from pathlib import Path
from typing import Any
import torch


def aggregate_stats(grams:list[torch.Tensor], crosses:list[torch.Tensor], alpha:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    a=alpha.to(dtype=torch.float64)
    g=torch.einsum('i,ijk->jk',a,torch.stack(grams))
    c=torch.einsum('i,ijk->jk',a,torch.stack(crosses))
    return g,c


def regmean_from_stats(g:torch.Tensor,c:torch.Tensor,ridge:float)->torch.Tensor:
    d=g.shape[0]
    eye=torch.eye(d,dtype=g.dtype,device=g.device)
    return torch.linalg.solve((g+ridge*eye).T,c.T).T


def solve_mirror_code(w0:torch.Tensor,basis:torch.Tensor,g:torch.Tensor,
                       c:torch.Tensor,ridge:float)->torch.Tensor:
    """Minimize the regularized RegMean quadratic directly in coefficients."""
    out_dtype=w0.dtype
    w0=w0.to(torch.float64); basis=basis.to(torch.float64)
    g=g.to(torch.float64); c=c.to(torch.float64)
    d=g.shape[0]; eye=torch.eye(d,dtype=g.dtype,device=g.device)
    gr=g+ridge*eye
    residual=c-w0@gr
    r=basis.shape[0]
    h=torch.empty(r,r,dtype=torch.float64)
    b=torch.empty(r,dtype=torch.float64)
    bg=[basis[j]@gr for j in range(r)]
    for j in range(r):
        b[j]=torch.sum(basis[j]*residual)
        for k in range(r): h[j,k]=torch.sum(bg[j]*basis[k])
    # Tiny numerical jitter only protects near-null redundant SVD directions.
    jitter=max(float(torch.linalg.norm(h,ord=2))*1e-12,1e-14)
    m=torch.linalg.solve(h+jitter*torch.eye(r,dtype=h.dtype),b)
    return m.to(dtype=out_dtype)


def fit_frobenius_code(w0:torch.Tensor,basis:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    flat=basis.reshape(basis.shape[0],-1).to(torch.float64)
    delta=(target-w0).reshape(-1).to(torch.float64)
    return (flat@delta).to(dtype=w0.dtype)


def decode(w0:torch.Tensor,basis:torch.Tensor,codes:torch.Tensor)->torch.Tensor:
    if codes.ndim==1:return w0+torch.einsum('r,roi->oi',codes,basis)
    return w0.unsqueeze(0)+torch.einsum('nr,roi->noi',codes,basis)


def orthogonal_basis(vectors:torch.Tensor,rank:int)->torch.Tensor:
    """Rows of vectors are flattened deltas; return rank orthonormal matrices."""
    mat=vectors.reshape(vectors.shape[0],-1).to(torch.float64)
    _,_,vh=torch.linalg.svd(mat,full_matrices=False)
    if rank>vh.shape[0]: raise ValueError(f'rank {rank} exceeds available basis {vh.shape[0]}')
    return vh[:rank].reshape(rank,*vectors.shape[1:]).to(torch.float32)


def serialize(payload:dict[str,Any],path:Path)->dict[str,Any]:
    stream=io.BytesIO();torch.save(payload,stream);blob=stream.getvalue();path.write_bytes(blob)
    return {'path':str(path),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}


def replay_payload(path:Path,expected:torch.Tensor)->float:
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if 'models' in obj: actual=obj['models']
    elif 'source_models' in obj['shared']:
        shared=obj['shared'];actual=[]
        for alpha in obj['alpha']:
            actual.append(sum(float(alpha[i])*shared['source_models'][i] for i in range(len(alpha))))
        actual=torch.stack(actual)
    else:
        w0=obj['shared']['w0'];basis=obj['shared']['basis'];codes=obj['codes']
        actual=decode(w0,basis,codes)
    delta=float((actual-expected).abs().max())
    # Float32 reduction order can differ by one ULP across a serialized replay.
    # Require a strict bound and report the observed maximum difference.
    if delta>5e-8:raise AssertionError(f'payload replay differs by {delta}')
    return delta


def regression_metrics(weights:torch.Tensor,domains:list[torch.Tensor],labels:torch.Tensor,
                       alpha:torch.Tensor)->tuple[float,float]:
    import torch.nn.functional as F
    acc=0.0;nll=0.0
    for i,x in enumerate(domains):
        logits=x.to(dtype=weights.dtype)@weights.T
        acc+=float(alpha[i])*float((logits.argmax(-1)==labels).float().mean())
        nll+=float(alpha[i])*float(F.cross_entropy(logits,labels))
    return acc,nll


def weighted_merge_error(weights:torch.Tensor,full:torch.Tensor,g:torch.Tensor)->float:
    diff=(weights-full).to(torch.float64);g=g.to(torch.float64)
    num=torch.trace(diff@g@diff.T).clamp_min(0).sqrt()
    den=torch.trace(full.to(torch.float64)@g@full.to(torch.float64).T).clamp_min(1e-20).sqrt()
    return float(num/den)


def features_for_rotation(images:torch.Tensor,k:int)->torch.Tensor:
    rotated=torch.rot90(images,k=k,dims=(-2,-1)).reshape(images.shape[0],-1).to(torch.float64)/16.0
    return torch.cat([rotated,torch.ones(images.shape[0],1,dtype=torch.float64)],dim=1)


def fit_ridge_classifier(x:torch.Tensor,labels:torch.Tensor,lam:float)->torch.Tensor:
    y=torch.nn.functional.one_hot(labels.to(torch.int64),num_classes=10).to(torch.float64)
    n=x.shape[0];gram=x.T@x/n;rhs=x.T@y/n
    penalty=torch.eye(x.shape[1],dtype=x.dtype);penalty[-1,-1]=0.0
    w=torch.linalg.solve(gram+lam*penalty,rhs).T.contiguous()
    return w.to(torch.float32)


def split_indices(labels,seed:int):
    from sklearn.model_selection import train_test_split
    import numpy as np
    idx=np.arange(len(labels));labels_np=np.asarray(labels)
    train,rest=train_test_split(idx,test_size=.4,random_state=seed,stratify=labels_np)
    dev,audit=train_test_split(rest,test_size=.5,random_state=seed+1,stratify=labels_np[rest])
    return train.tolist(),dev.tolist(),audit.tolist()


def source_statistics(source_models:list[torch.Tensor],train_domains:list[torch.Tensor])->tuple[list[torch.Tensor],list[torch.Tensor]]:
    grams=[];crosses=[]
    for w,x in zip(source_models,train_domains):
        xd=x.to(torch.float64);g=xd.T@xd/xd.shape[0]
        grams.append(g);crosses.append(w.to(torch.float64)@g)
    return grams,crosses


def make_alphas(seed:int,n:int,concentration:float=.7)->torch.Tensor:
    import numpy as np
    a=np.random.default_rng(seed).dirichlet([concentration]*4,size=n)
    return torch.from_numpy(a.astype('float32'))


def regmean_for_alpha(grams,crosses,alpha,ridge):
    g,c=aggregate_stats(grams,crosses,alpha)
    return regmean_from_stats(g,c,ridge),g,c


def basis_library_payload(method:str,seed:int,w0:torch.Tensor,basis:torch.Tensor,codes:torch.Tensor,request_count:int=16):
    return {'schema':'MA-715/inference-v1','experiment':'MA-715','world_seed':seed,'method':method,
            'request_ids':list(range(request_count)),'shared':{'w0':w0,'basis':basis},'codes':codes}
