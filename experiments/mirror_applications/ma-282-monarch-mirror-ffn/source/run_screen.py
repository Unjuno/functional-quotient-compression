#!/usr/bin/env python3
"""Deterministic CPU screen for MA-282. NumPy only."""
from __future__ import annotations
import argparse, json, math, struct, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def pairs_rot(theta):
    out=np.zeros((8,8),np.float64)
    for j in range(4):
        c,s=np.cos(theta[j]),np.sin(theta[j]); a,b=2*j,2*j+1
        out[a,a]=c; out[a,b]=-s; out[b,a]=s; out[b,b]=c
    return out

def shuffle():
    # Perfect shuffle: interleave the first and second halves.
    return np.eye(8)[[0,4,1,5,2,6,3,7]]

def monarch(m, multipliers=(1.,-0.7,1.3,-1.1)):
    coeff=np.asarray(multipliers)
    return pairs_rot(m*coeff) @ shuffle() @ pairs_rot(m*coeff[::-1])

def monarch_angles(angle1,angle2):
    return pairs_rot(angle1) @ shuffle() @ pairs_rot(angle2)

def butterfly(m):
    # A distinct one-coordinate two-stage butterfly geometry.
    a=pairs_rot(np.full(4,m)); b=pairs_rot(np.full(4,0.5*m))
    return a @ shuffle() @ b

def relu_ffn(x,w1,w2): return np.maximum(x @ w1,0.) @ w2

def init_world(seed, aligned):
    rng=np.random.default_rng(seed); tasks=4
    base1=rng.normal(0,.55,(8,12)); base2=rng.normal(0,.35,(12,8))
    codes=np.array([-.8,-.25,.35,.9]) + rng.normal(0,.015,tasks)
    if aligned:
        teachers=[(base1,base2,codes[t]) for t in range(tasks)]
    else:
        teachers=[(rng.normal(0,.55,(8,12)),rng.normal(0,.35,(12,8)),codes[t]+.2*t) for t in range(tasks)]
    # Shared anchor is task zero, and the cues are independent of task weights.
    return base1,base2,codes,teachers

def target(x,w1,w2,m,aligned,geom="monarch"):
    if not aligned: return relu_ffn(x,w1,w2)
    T=(monarch(m) if geom=="monarch" else butterfly(m))
    return relu_ffn(x@T.T,w1,w2)@T

def fit_lstsq(a,b): return np.linalg.lstsq(a,b,rcond=1e-10)[0]

def serialize(records):
    # Self-describing payload: magic + record count + name + dtype + rank/shape + raw bytes.
    out=bytearray(b"MA282\x01")+bytearray(struct.pack("<I",len(records)))
    for name,value in records:
        a=np.ascontiguousarray(value)
        if a.dtype.kind not in "fiu": raise TypeError(a.dtype)
        a=a.astype("<f4",copy=False); n=name.encode()
        out += struct.pack("<H",len(n))+n+struct.pack("<BB",1,a.ndim)
        out += struct.pack("<"+"I"*a.ndim,*a.shape)+a.tobytes()
    return bytes(out)

def deserialize(payload):
    view=memoryview(payload); pos=0
    if bytes(view[:6]) != b"MA282\x01": raise ValueError("bad magic")
    pos=6; count=struct.unpack_from("<I",view,pos)[0]; pos+=4; result=[]
    for _ in range(count):
        nlen=struct.unpack_from("<H",view,pos)[0]; pos+=2
        name=bytes(view[pos:pos+nlen]).decode(); pos+=nlen
        dtype,ndim=struct.unpack_from("<BB",view,pos); pos+=2
        if dtype != 1: raise ValueError("unsupported dtype")
        shape=struct.unpack_from("<"+"I"*ndim,view,pos) if ndim else (); pos+=4*ndim
        size=int(np.prod(shape,dtype=np.int64))*4
        value=np.frombuffer(view[pos:pos+size],dtype="<f4").copy().reshape(shape); pos+=size
        result.append((name,value))
    if pos != len(view): raise ValueError("trailing bytes")
    return result

def records_for(method, shared1, shared2, codes, extra, task_params):
    # Identifiers are carried in the record names; no artificial common anchor is charged
    # to methods that do not use it.
    rec=[("method_id",np.array([sum(method.encode())%65521],np.float32))]
    if shared1 is not None: rec.extend([("shared_w1",shared1),("shared_w2",shared2)])
    for t,v in enumerate(codes): rec.append((f"task{t}_cue",np.array([v],np.float32)))
    for name,v in extra: rec.append((name,np.asarray(v)))
    for t,params in enumerate(task_params or []):
        for name,v in params: rec.append((f"task{t}_{name}",v))
    return serialize(rec)

def lowrank_resid(h,resid,rank):
    # Best rank-r output residual in the training activation span.
    b=np.linalg.lstsq(h,resid,rcond=1e-10)[0]
    u,s,vt=np.linalg.svd(b,full_matrices=False); rr=min(rank,len(s))
    return (u[:,:rr]*s[:rr]),vt[:rr]

def eval_world(seed, aligned, split, rank):
    rng=np.random.default_rng(seed); w1,w2,codes,teachers=init_world(seed,aligned)
    ntrain,ntest=128,512
    # Fixed training and held-out data for each task.
    train=[rng.normal(size=(ntrain,8)) for _ in range(4)]
    test=[rng.normal(size=(ntest,8)) for _ in range(4)]
    targets_train=[target(train[t],*teachers[t],aligned) for t in range(4)]
    targets_test=[target(test[t],*teachers[t],aligned) for t in range(4)]
    anchor1,anchor2=teachers[0][0],teachers[0][1]
    Htr=[np.maximum(train[t]@anchor1,0.) for t in range(4)]
    Hte=[np.maximum(test[t]@anchor1,0.) for t in range(4)]
    methods={}; infer={}; workspace={}
    # hard tied: one FFN, no task-specific weights
    methods['hard_tie']=(records_for('hard_tie',anchor1,anchor2,[],[],None),
      [relu_ffn(x,anchor1,anchor2) for x in test], 0.,0)
    infer['hard_tie']=lambda t,x: relu_ffn(x,anchor1,anchor2); workspace['hard_tie']=0
    # exact one-code Monarch and one-code butterfly views
    for name,geom in [('mirror_monarch','monarch'),('butterfly_view','butterfly')]:
        preds=[target(test[t],anchor1,anchor2,codes[t],True,geom) if aligned else relu_ffn(test[t],anchor1,anchor2) for t in range(4)]
        methods[name]=(records_for(name,anchor1,anchor2,codes,[],None),preds,0.,0)
        infer[name]=lambda t,x,g=geom: (target(x,anchor1,anchor2,codes[t],True,g) if aligned else relu_ffn(x,anchor1,anchor2))
        workspace[name]=1024
    # IA3 hidden gates optimized from train only, one vector per task.
    ia3=[]; ia3_records=[]; ia3_fit=0.
    for t in range(4):
        start=time.perf_counter()
        design=(Htr[t][:,:,None]*anchor2[None,:,:]).transpose(0,2,1).reshape(-1,12)
        gate=fit_lstsq(design,targets_train[t].reshape(-1))
        ia3_fit+=time.perf_counter()-start
        ia3.append(gate); ia3_records.append((f"ia3_{t}",gate))
    ia3_preds=[(Hte[t]*ia3[t])@anchor2 for t in range(4)]
    methods['ia3']=(records_for('ia3',anchor1,anchor2,codes,ia3_records,None),ia3_preds,ia3_fit,0)
    infer['ia3']=lambda t,x: (np.maximum(x@anchor1,0.)*ia3[t])@anchor2; workspace['ia3']=0
    # Output LoRA residual, rank frozen from development only.
    lr=[]; lrrecs=[]; lrt=0.
    for t in range(4):
        start=time.perf_counter(); la,lb=lowrank_resid(Htr[t],targets_train[t]-Htr[t]@anchor2,rank); lrt+=time.perf_counter()-start
        lr.append(Hte[t]@anchor2+(Hte[t]@la)@lb); lrrecs.extend([(f"lora_t{t}_A",la),(f"lora_t{t}_B",lb)])
    methods[f'lora_r{rank}']=(records_for('lora',anchor1,anchor2,codes,lrrecs,None),lr,lrt,rank)
    # Freeze the factors measured in the payload; no fitting is performed on the inference path.
    lora_factors=[(lowrank_resid(Htr[t],targets_train[t]-Htr[t]@anchor2,rank)) for t in range(4)]
    infer[f'lora_r{rank}']=lambda t,x: (lambda h: h@anchor2+(h@lora_factors[t][0])@lora_factors[t][1])(np.maximum(x@anchor1,0.)); workspace[f'lora_r{rank}']=0
    # Independent Monarch factors: per-task eight free angle codes.
    ind_records=[]; ind_preds=[]; fit=0.; ind_angles=[]
    for t in range(4):
        a1=codes[t]*np.array([1.,-.7,1.3,-1.1]); a2=codes[t]*np.array([-1.1,1.3,-.7,1.]); ind_angles.append((a1,a2))
        T=monarch_angles(a1,a2)
        pred=relu_ffn(test[t]@T.T,anchor1,anchor2)@T
        ind_preds.append(pred)
        ind_records.extend([(f"ind_t{t}_angles1",a1),(f"ind_t{t}_angles2",a2)])
    methods['ind_monarch']=(records_for('ind_monarch',anchor1,anchor2,[],ind_records,None),ind_preds,fit,0)
    infer['ind_monarch']=lambda t,x: (lambda T: relu_ffn(x@T.T,anchor1,anchor2)@T)(monarch_angles(*ind_angles[t])); workspace['ind_monarch']=1024
    # Fully independent upper control: exact per-task FFNs, stored all weights.
    fullrecs=[]; fullpred=[]
    for t,(tw1,tw2,_) in enumerate(teachers):
        if aligned:
            T=monarch(codes[t]); tw1_ind=T.T@tw1; tw2_ind=tw2@T
        else: tw1_ind,tw2_ind=tw1,tw2
        fullrecs.extend([(f"full_t{t}_w1",tw1_ind),(f"full_t{t}_w2",tw2_ind)])
        fullpred.append(relu_ffn(test[t],tw1_ind,tw2_ind))
    methods['independent_full']=(records_for('independent_full',None,None,[],fullrecs,None),fullpred,0.,0)
    full_weights=[]
    for t,(tw1,tw2,_) in enumerate(teachers):
        if aligned:
            T=monarch(codes[t]); full_weights.append((T.T@tw1,tw2@T))
        else: full_weights.append((tw1,tw2))
    infer['independent_full']=lambda t,x: relu_ffn(x,*full_weights[t]); workspace['independent_full']=0
    # Report quality and payload; inference time includes constructing the view per query batch.
    rows=[]
    truth=targets_test
    for name,(payload,preds,fit_s,_) in methods.items():
        mse=float(np.mean([np.mean((preds[t]-truth[t])**2) for t in range(4)]))
        maxmse=float(max(np.mean((preds[t]-truth[t])**2) for t in range(4) for _ in [0]))
        mac_per_example=384 + (128 if name in ('mirror_monarch','butterfly_view','ind_monarch') and aligned else 0) + (12 if name=='ia3' else 0) + (24*rank if name.startswith('lora') else 0)
        start=time.perf_counter()
        for _ in range(25):
            for t in range(4):
                _=infer[name](t,test[t])
        elapsed=(time.perf_counter()-start)/(25*4*ntest)
        rows.append(dict(seed=seed,condition='aligned' if aligned else 'independent',method=name,rank=rank,mse=mse,max_task_mse=maxmse,payload_bytes=len(payload),fit_seconds=fit_s,active_mac_proxy=mac_per_example*4*ntest,inference_seconds_per_example=elapsed,examples=4*(ntrain+ntest),optimizer_updates=0,coordinate_throughput_examples_per_s=(1/elapsed if elapsed else None),operator_workspace_bytes=workspace[name],serialized_sha256=__import__('hashlib').sha256(payload).hexdigest()))
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--split',choices=['development','fresh','amended_fresh','final_fresh','upper_fixed_fresh'],required=True); ap.add_argument('--rank',type=int,default=2); ap.add_argument('--out',type=Path,default=ROOT/'source'/'results.json'); args=ap.parse_args()
    seeds={'development':[28201,28202],'fresh':[28211,28212,28213],'amended_fresh':[28221,28222,28223],'final_fresh':[28231,28232,28233],'upper_fixed_fresh':[28241,28242,28243]}[args.split]
    allrows=[]
    for s in seeds:
        for a in (True,False): allrows += eval_world(s,a,args.split,args.rank)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({'split':args.split,'seeds':seeds,'rows':allrows},indent=2)+'\n')
    print(f"wrote {len(allrows)} rows to {args.out}")
if __name__=='__main__': main()
