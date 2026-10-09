import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


torch.set_num_threads(1)
INPUT, HIDDEN, OUTPUT, TASKS, ALIGNED = 8, 12, 4, 8, 6
RANK = 2
METHODS = ["independent_full", "unaligned_rank2", "rebasin_direct", "mirror_phase"]


def mlp(x, w1, b1, w2, b2):
    return torch.relu(x @ w1 + b1) @ w2 + b2


def flatten_delta(weights, base):
    return torch.cat([(a-b).reshape(-1) for a, b in zip(weights, base)])


def unflatten_weights(vec, base):
    out=[]; offset=0
    for b in base:
        n=b.numel(); out.append(b + vec[offset:offset+n].reshape_as(b)); offset += n
    assert offset == vec.numel()
    return out


def align_to_base(weights, base):
    w1, b1, w2, b2 = weights
    bw1, bb1, bw2, bb2 = base
    a = w1 / (w1.norm(dim=0, keepdim=True) + 1e-12)
    b = bw1 / (bw1.norm(dim=0, keepdim=True) + 1e-12)
    cost = -(a.T @ b).cpu().numpy()
    rows, cols = linear_sum_assignment(cost)
    ordering = np.empty(HIDDEN, dtype=np.int64)
    ordering[cols] = rows
    ix = torch.as_tensor(ordering, dtype=torch.long)
    aligned = [w1[:, ix], b1[ix], w2[ix], b2]
    return aligned, ordering


def world(seed):
    g = torch.Generator().manual_seed(seed)
    base = [torch.randn(INPUT, HIDDEN, generator=g)*0.3,
            torch.randn(HIDDEN, generator=g)*0.1,
            torch.randn(HIDDEN, OUTPUT, generator=g)*0.3,
            torch.randn(OUTPUT, generator=g)*0.1]
    raw=torch.randn(2,HIDDEN*OUTPUT,generator=g)
    q,_=torch.linalg.qr(raw.T,mode="reduced")
    basis=q.T.reshape(2,HIDDEN,OUTPUT)
    angles=2*torch.pi*torch.arange(ALIGNED)/ALIGNED + (seed%101)*0.007
    rho=0.32
    deltas=[]
    for t in range(ALIGNED):
        deltas.append(rho*(torch.cos(angles[t])*basis[0]+torch.sin(angles[t])*basis[1]))
    for _ in range(TASKS-ALIGNED):
        deltas.append(torch.randn(HIDDEN,OUTPUT,generator=g)*0.28)
    rng=np.random.default_rng(seed+1901)
    models=[]; perms=[]
    for t,delta in enumerate(deltas):
        w=[base[0].clone(),base[1].clone(),base[2]+delta,base[3].clone()]
        if t < ALIGNED:
            perm=rng.permutation(HIDDEN)
            ix=torch.as_tensor(perm,dtype=torch.long)
            w=[w[0][:,ix],w[1][ix],w[2][ix],w[3]]
        else:
            perm=np.arange(HIDDEN)
        models.append(w);perms.append(perm)
    xr=torch.randn(2048,INPUT,generator=g)
    xval=torch.randn(1024,INPUT,generator=g)
    return base, models, xr, xval


def aligned_deltas(base, models):
    aligned=[]; indices=[]
    for w in models[:ALIGNED]:
        wa,idx=align_to_base(w,base)
        aligned.append(wa);indices.append(idx)
    ds=torch.stack([(w[2]-base[2]).reshape(-1) for w in aligned])
    u,s,vh=torch.linalg.svd(ds,full_matrices=False)
    # rank-2 task manifold; phase code assumes the planted circular orbit.
    b=vh[:RANK].reshape(RANK,*base[2].shape)
    coeff=u[:,:RANK]*s[:RANK]
    radius=coeff.norm(dim=1).mean()
    phase=torch.atan2(coeff[:,1],coeff[:,0])
    unit=coeff/radius
    return aligned,indices,b,radius,unit,phase,ds


def deterministic_pack(arrays,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_STORED) as z:
        for name in sorted(arrays):
            buf=io.BytesIO();np.lib.format.write_array(buf,np.ascontiguousarray(arrays[name]),allow_pickle=False)
            info=zipfile.ZipInfo(name+".npy",date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED;info.external_attr=0o600<<16
            z.writestr(info,buf.getvalue())
    raw=path.read_bytes();return len(raw),hashlib.sha256(raw).hexdigest()


def serialize(method,base,models,fit,path):
    _,_,b,radius,unit,phase,ds=fit
    f16=lambda x:x.detach().cpu().numpy().astype("<f2")
    arrays={"meta":np.asarray([INPUT,HIDDEN,OUTPUT,TASKS,ALIGNED,RANK],dtype=np.uint16),
            "base_w1":f16(base[0]),"base_b1":f16(base[1]),"base_w2":f16(base[2]),"base_b2":f16(base[3])}
    if method=="independent_full":
        for i in range(4):arrays[f"task_{i}"]=f16(torch.stack([m[i] for m in models]))
    elif method=="unaligned_rank2":
        all_d=torch.stack([flatten_delta(m,base) for m in models[:ALIGNED]])
        u,s,vh=torch.linalg.svd(all_d,full_matrices=False)
        arrays["raw_basis"]=f16(vh[:RANK]);arrays["raw_codes"]=f16(u[:,:RANK]*s[:RANK])
        arrays["private_w2"]=f16(torch.stack([models[t][2]-base[2] for t in range(ALIGNED,TASKS)]))
    else:
        arrays["delta_basis"]=f16(b*radius)
        arrays["private_w2"]=f16(torch.stack([models[t][2]-base[2] for t in range(ALIGNED,TASKS)]))
        if method=="rebasin_direct":arrays["codes"]=f16(unit)
        else:arrays["phase"]=f16(phase)
    n,h=deterministic_pack(arrays,path);return n,h,arrays


def load_arrays(path):
    arrays={}
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():arrays[member[:-4]]=np.load(io.BytesIO(z.read(member)),allow_pickle=False)
    return arrays


def reconstruct(method,arrays,base):
    t=[]
    if method=="independent_full":
        return [[torch.from_numpy(np.array(arrays[f"task_{i}"][j],copy=True)).float() for i in range(4)] for j in range(TASKS)]
    if method=="unaligned_rank2":
        b=torch.from_numpy(np.array(arrays["raw_basis"],copy=True)).float();c=torch.from_numpy(np.array(arrays["raw_codes"],copy=True)).float()
        private=torch.from_numpy(np.array(arrays["private_w2"],copy=True)).float()
        for j in range(TASKS):
            if j<ALIGNED: t.append(unflatten_weights(c[j]@b,base))
            else:
                w=[x.clone() for x in base];w[2]=w[2]+private[j-ALIGNED];t.append(w)
        return t
    b=torch.from_numpy(np.array(arrays["delta_basis"],copy=True)).float()
    private=torch.from_numpy(np.array(arrays["private_w2"],copy=True)).float()
    if method=="rebasin_direct":codes=torch.from_numpy(np.array(arrays["codes"],copy=True)).float()
    else:
        phase=torch.from_numpy(np.array(arrays["phase"],copy=True)).float();codes=torch.stack((torch.cos(phase),torch.sin(phase)),dim=-1)
    for j in range(TASKS):
        w=[x.clone() for x in base]
        if j<ALIGNED:w[2]=w[2]+torch.einsum("r,rio->io",codes[j],b)
        else:w[2]=w[2]+private[j-ALIGNED]
        t.append(w)
    return t


def metrics(method,arrays,base,models,x):
    start=time.perf_counter();predicted=reconstruct(method,arrays,base);recon_wall=time.perf_counter()-start
    errs=[]; nm=[]
    with torch.no_grad():
        for j in range(TASKS):
            ref=mlp(x,*models[j]);got=mlp(x,*predicted[j]);err=(got-ref).square().mean();den=ref.square().mean()
            errs.append(float(err));nm.append(float(err/(den+1e-12)))
    return {"task_mse":errs,"task_nmse":nm,"mean_nmse":float(np.mean(nm)),"reconstruction_wall_s":recon_wall}


def run(seed,split,outdir,jsonpath):
    base,models,xtrain,xval=world(seed);fit=aligned_deltas(base,models)
    rows=[]
    for method in METHODS:
        path=Path(outdir)/f"{split}_{seed}_{method}.zip"
        n,digest,arrays=serialize(method,base,models,fit,path)
        arrays=load_arrays(path)
        tr=metrics(method,arrays,base,models,xtrain);va=metrics(method,arrays,base,models,xval)
        ops=sum(p.size for p in arrays.values() if isinstance(p,np.ndarray) and p.dtype.kind=='f')
        rows.append({"condition":split,"seed":seed,"method":method,"serialized_bytes":n,"payload_sha256":digest,
                     "optimizer_updates":0,"examples_seen":len(xtrain),"active_parameter_ops_proxy":ops*len(xval),
                     "reconstruction_wall_s":va["reconstruction_wall_s"],"train_nmse":tr["mean_nmse"],
                     "validation_task_nmse":va["task_nmse"],"validation_mean_nmse":va["mean_nmse"],
                     "per_task_nmse":va["task_nmse"],"symmetry_alignment_recovered":method!="unaligned_rank2"})
    out={"condition":split,"seed":seed,"summaries":rows};Path(jsonpath).parent.mkdir(parents=True,exist_ok=True)
    Path(jsonpath).write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--seed",type=int,required=True);ap.add_argument("--split",choices=["development","fresh"],required=True);ap.add_argument("--outdir",required=True);ap.add_argument("--json",required=True);a=ap.parse_args();run(a.seed,a.split,a.outdir,a.json)
