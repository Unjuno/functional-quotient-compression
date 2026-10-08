#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"artifacts"
SEEDS=(33801,33802,33811,33812,33813);NTRAIN=48;NTASK=64;NH=12;DIN=8;DOUT=4;NCAL=256;NEVAL=1024

def pack(arrays,meta):
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in sorted(arrays):
            raw=io.BytesIO();np.save(raw,np.asarray(arrays[name]),allow_pickle=False)
            i=zipfile.ZipInfo(name+".npy",(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,raw.getvalue())
        i=zipfile.ZipInfo("metadata.json",(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,json.dumps(meta,sort_keys=True,separators=(",",":")).encode())
    payload=b.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        a={k[:-4]:np.load(io.BytesIO(z.read(k)),allow_pickle=False) for k in z.namelist() if k.endswith(".npy")};m=json.loads(z.read("metadata.json"))
    return payload,a,m

def infer(net,x):
    h=np.tanh(x@net["W1"].T+net["b1"])
    return h@net["W2"].T+net["b2"]

def vec(net):return np.concatenate([net[k].ravel() for k in ("W1","b1","W2","b2")])
def unvec(v):
    i=0;out={}
    for name,shape in (("W1",(NH,DIN)),("b1",(NH,)),("W2",(DOUT,NH)),("b2",(DOUT,))):
        n=int(np.prod(shape));out[name]=v[i:i+n].reshape(shape).astype(np.float32);i+=n
    return out

def canonicalize(net,xcal,href):
    h=np.tanh(xcal@net["W1"].T+net["b1"]);corr=href.T@h/(np.linalg.norm(href,axis=0)[:,None]*np.linalg.norm(h,axis=0)[None,:]+1e-30)
    pairs=sorted(((abs(float(corr[j,i])),j,i) for j in range(NH) for i in range(NH)),reverse=True)
    mapping={};usedj=set();usedi=set()
    for _,j,i in pairs:
        if j not in usedj and i not in usedi:mapping[i]=j;usedj.add(j);usedi.add(i)
    out={k:v.copy() for k,v in net.items()};w1=np.empty_like(net["W1"]);b1=np.empty_like(net["b1"]);w2=np.empty_like(net["W2"]);maxerr=0.
    for i,j in mapping.items():
        s=1. if corr[j,i]>=0 else -1.
        w1[j]=s*net["W1"][i];b1[j]=s*net["b1"][i];w2[:,j]=s*net["W2"][:,i]
        maxerr=max(maxerr,float(np.max(np.abs(href[:,j]-s*h[:,i]))))
    out.update(W1=w1,b1=b1,W2=w2)
    return out,maxerr

def nerr(pred,target):return float(np.mean((pred-target)**2)/(np.mean(target**2)+1e-30))

def world(seed):
    rng=np.random.default_rng(seed);W1=(rng.normal(size=(NH,DIN))*.35).astype(np.float32);b1=(rng.normal(size=NH)*.15).astype(np.float32);W2=(rng.normal(size=(DOUT,NH))*.22).astype(np.float32);b2=(rng.normal(size=DOUT)*.05).astype(np.float32)
    z=rng.normal(size=(DOUT*NH,2));q,_=np.linalg.qr(z);basis=q.T.reshape(2,DOUT,NH).astype(np.float32);amp=.18
    phases=(np.arange(NTASK)*2*np.pi/NTASK+rng.uniform(0,2*np.pi)).astype(np.float32)
    can=[]
    for t,theta in enumerate(phases):
        d=amp*(np.cos(theta)*basis[0]+np.sin(theta)*basis[1]);net={"W1":W1.copy(),"b1":b1.copy(),"W2":W2+d,"b2":b2.copy()}
        perm=rng.permutation(NH);sgn=rng.choice(np.array([-1.,1.],np.float32),size=NH)
        gau={"W1":sgn[:,None]*net["W1"][perm],"b1":sgn*net["b1"][perm],"W2":net["W2"][:,perm]*sgn[None,:],"b2":net["b2"].copy()};can.append(gau)
    off={"W1":W1.copy(),"b1":b1.copy(),"W2":W2+rng.normal(size=(DOUT,NH)).astype(np.float32)*.3,"b2":b2.copy()}
    perm=rng.permutation(NH);sgn=rng.choice(np.array([-1.,1.],np.float32),size=NH);offg={"W1":sgn[:,None]*off["W1"][perm],"b1":sgn*off["b1"][perm],"W2":off["W2"][:,perm]*sgn[None,:],"b2":off["b2"].copy()}
    xcal=rng.normal(size=(NCAL,DIN)).astype(np.float32);xeval=rng.normal(size=(NEVAL,DIN)).astype(np.float32)
    base={"W1":W1,"b1":b1,"W2":W2,"b2":b2}
    return base,can,offg,xcal,xeval,phases

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dev-only",action="store_true");args=ap.parse_args();OUT.mkdir(exist_ok=True);rows=[]
    for seed in SEEDS[:2] if args.dev_only else SEEDS:
        base,rawtasks,rawoff,xcal,xeval,phases=world(seed);href=np.tanh(xcal@base["W1"].T+base["b1"])
        aligned=[];alignerr=[]
        for net in rawtasks+[rawoff]:
            c,e=canonicalize(net,xcal,href);aligned.append(c);alignerr.append(e)
        truth=[infer(n,xeval) for n in aligned];basevec=vec(base);rawvec=np.stack([vec(n) for n in rawtasks]);deltas=np.stack([n["W2"]-base["W2"] for n in aligned[:NTASK]])
        # SVD basis is fit only on task checkpoints 0..47, then codes are projected for all 64 views.
        _,s,vt=np.linalg.svd(deltas[:NTRAIN].reshape(NTRAIN,-1),full_matrices=False);B=vt[:2].reshape(2,DOUT,NH).astype(np.float32)
        coeff=np.einsum("tij,kij->tk",deltas,B);radius=float(np.mean(np.linalg.norm(coeff[:NTRAIN],axis=1)));theta=np.arctan2(coeff[:,1],coeff[:,0]).astype(np.float32)
        direct_delta=np.einsum("tk,kij->tij",coeff,B);mirror_delta=radius*(np.cos(theta)[:,None,None]*B[0]+np.sin(theta)[:,None,None]*B[1])
        # Raw full-checkpoint PCA rank-2 control, fit only on the 48 training checkpoint vectors.
        rawtrain=rawvec[:NTRAIN];rawmean=rawtrain.mean(axis=0);_,sr,vtr=np.linalg.svd(rawtrain-rawmean,full_matrices=False);RB=vtr[:2];rawcodes=(rawvec-rawmean)@RB.T;rawrecon=rawmean+rawcodes@RB
        # Required rank to preserve >=99% raw parameter variance (report only; no rank selection on fresh outputs).
        rr=int(np.searchsorted(np.cumsum(sr**2)/np.sum(sr**2),.99)+1)
        # Simple fixed-width payload variants.
        shared={"W1":base["W1"],"b1":base["b1"],"W2":base["W2"],"b2":base["b2"],"basis":B}
        cases={
          "independent_64":({f"task_{i}_{k}":aligned[i][k] for i in range(NTASK) for k in ("W1","b1","W2","b2")},{"kind":"independent","tasks":64}),
          "hard_tied":({k:base[k] for k in base},{"kind":"tied"}),
          "raw_checkpoint_pca_rank2":({"mean":rawmean.astype(np.float32),"basis":RB.astype(np.float32),"codes":rawcodes.astype(np.float32)},{"kind":"raw_pca","rank":2}),
          "normalized_direct_coefficients":({**shared,"coefficients":coeff.astype(np.float32),"private_offorbit":(aligned[-1]["W2"]-base["W2"]).astype(np.float32)},{"kind":"normalized_direct","rank":2,"private":True}),
          "normalized_mirror_phase":({**shared,"phase":theta,"radius":np.array([radius],np.float32),"private_offorbit":(aligned[-1]["W2"]-base["W2"]).astype(np.float32)},{"kind":"normalized_phase","rank":2,"private":True}),
          "normalized_mirror_no_private":({**shared,"phase":theta,"radius":np.array([radius],np.float32)},{"kind":"normalized_phase","rank":2,"private":False})}
        for name,(arrays,meta) in cases.items():
            t0=time.perf_counter();payload,ld,md=pack(arrays,meta)
            if name=="independent_64": preds=[infer({k:ld[f"task_{i}_{k}"] for k in ("W1","b1","W2","b2")},xeval) for i in range(NTASK)]+[infer(aligned[-1],xeval)]
            elif name=="hard_tied": preds=[infer({k:ld[k] for k in base},xeval) for _ in range(NTASK)]+[infer({k:ld[k] for k in base},xeval)]
            elif name=="raw_checkpoint_pca_rank2":preds=[infer(unvec(v),xeval) for v in rawrecon]+[infer(unvec(rawmean),xeval)]
            else:
                preds=[]
                for i in range(NTASK):
                    if name=="normalized_direct_coefficients":d=ld["coefficients"][i]@ld["basis"].reshape(2,-1);delta=d.reshape(DOUT,NH)
                    else:
                        ph=float(ld["phase"][i]);delta=float(ld["radius"][0])*(np.cos(ph)*ld["basis"][0]+np.sin(ph)*ld["basis"][1])
                    net={k:ld[k].copy() for k in base};net["W2"]=net["W2"]+delta;preds.append(infer(net,xeval))
                noff={k:ld[k].copy() for k in base}
                if "private_offorbit" in ld:noff["W2"]+=ld["private_offorbit"]
                else:noff["W2"]+=np.zeros_like(noff["W2"])
                preds.append(infer(noff,xeval))
            elapsed=time.perf_counter()-t0
            e=[nerr(a,b) for a,b in zip(preds,truth)];held=e[NTRAIN:NTASK];
            rows.append({"seed":seed,"method":name,"payload_bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest(),"heldout_aligned_mean_nMSE":float(np.mean(held)),"heldout_aligned_max_nMSE":max(held),"offorbit_nMSE":e[-1],"alignment_max_hidden_error":max(alignerr),"raw_rank_for_99pct_variance":rr,"normalized_delta_rank":int(np.sum(s>1e-7)),"examples_per_task":NEVAL,"calibration_examples_per_checkpoint":NCAL,"optimizer_updates":0,"fit_compute_proxy_SVD_multiplies":int(NTRAIN*(DOUT*NH)**2),"active_forward_MACs_65_tasks":int(NEVAL*(DIN*NH+NH*DOUT)*65),"wall_seconds_serialize_eval":elapsed})
    out=OUT/("development.csv" if args.dev_only else "results.csv")
    with out.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator="\n");w.writeheader();w.writerows(rows)
    for r in rows:print(json.dumps(r,sort_keys=True))
if __name__=="__main__":main()
