#!/usr/bin/env python3
"""MA-1171 isolated Stage-0: exact paid-byte joint codec with Mirror/non-Mirror controls.

Preregistered before seeing dev/fresh results at GitHub protocol commit
5002b45a76d27c8b0e1632e70abb5c3c9d989122.

This is not an LLM or a reconstruction of native Compress-then-Serve.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import struct
import time
import numpy as np

METHODS = ('native_direct', 'native_linear', 'mirror_orientation')
BITS = (2,4,8)
PRIV = (0,1)
SEEDS = {'dev': (11,12,13), 'fresh': (101,102,103,104,105)}
ALPHAS = (1.,.5,0.)
BUDGET_RATIOS = (.65,.8,1.,1.15)
N_TASK=6
N_BLOCK=4
D=8
MAX_STATES=18**4
HEADER=b'FQCM1'


def packbits(values, bits):
    arr=np.asarray(values,dtype=np.uint8).ravel()
    assert bits in BITS
    assert np.all(arr < 1<<bits)
    stream=bytearray((len(arr)*bits+7)//8)
    offset=0
    for v in arr:
        j=offset//8
        shift=offset%8
        value=int(v)<<shift
        stream[j] |= value & 255
        if j+1<len(stream) and (shift+bits)>8:
            stream[j+1] |= (value>>8)&255
        offset+=bits
    return bytes(stream)


def unpackbits(data,bits,count):
    out=[]
    offset=0
    for _ in range(count):
        j=offset//8;sh=offset%8
        val=(data[j]>>sh)
        if sh+bits>8:
            val |= data[j+1]<<(8-sh)
        out.append(val&((1<<bits)-1));offset+=bits
    return np.asarray(out,dtype=np.uint8)


def rotate2(vec, angle):
    c,s=np.cos(angle),np.sin(angle)
    a,b=vec[...,0],vec[...,1]
    return np.stack([c*a-s*b,s*a+c*b],axis=-1)


def input_batch(rng,n,mix):
    u=rng.normal(size=(D,D))
    q,_=np.linalg.qr(u)
    raw=rng.normal(size=(n,D))
    scales=np.asarray([1.6,.8,1.3,0.5,.7,1.5,.9,1.8])
    # Dense mixing: task errors from different blocks cannot be treated independently.
    return raw @ np.diag(scales) @ (np.eye(D)*.58+q*.42)


def _source_and_targets(seed,alpha):
    rng=np.random.default_rng(158137*seed+int(alpha*17)+487)
    base=rng.normal(size=(N_BLOCK,2))*.9
    def draw_true(count):
        angles=rng.uniform(-.65,.65,size=(count,N_BLOCK))
        orbit=rotate2(np.broadcast_to(base,(count,N_BLOCK,2)),angles)
        # off-orbit component independent per role/block
        noise=rng.normal(size=(count,N_BLOCK,2))*.58
        return (orbit+(1-alpha)*noise).reshape(count,D)
    source_true=draw_true(6)
    target_true=draw_true(N_TASK)
    source_train=[];source_fit=[]
    for k in range(6):
        x=input_batch(rng,128,seed)
        y=x@source_true[k]+rng.normal(0,.016,size=len(x))
        source_train.append(x)
        source_fit.append(np.linalg.lstsq(x,y,rcond=None)[0])
    target_support=[];target_fit=[]
    for k in range(N_TASK):
        xs=input_batch(rng,96,seed)
        ys=xs@target_true[k]+rng.normal(0,.025,size=len(xs))
        target_support.append(xs)
        target_fit.append(np.linalg.lstsq(xs,ys,rcond=None)[0])
    probe=np.concatenate([input_batch(rng,256,seed) for _ in range(3)],axis=0)
    test=np.stack([input_batch(rng,512,seed) for _ in range(N_TASK)])
    return np.asarray(source_fit),np.asarray(target_fit),target_true,probe,test


def quantize(v,bits):
    v=np.asarray(v,dtype=np.float64)
    lo=np.float32(float(v.min()))
    hi=np.float32(float(v.max()))
    levels=(1<<bits)-1
    step=np.float32((hi-lo)/levels if hi>lo else 0.)
    if step==0.:
        ints=np.zeros_like(v,dtype=np.uint8)
    else:
        ints=np.rint((v-float(lo))/float(step)).clip(0,levels).astype(np.uint8)
    return lo,step,ints


def _fit_block(source,target,method,bits,private):
    # source [source_tasks,2], target [K,2] are independently fitted from allowed source/support only
    src_mean=source.mean(axis=0).astype(np.float32)
    if method=='native_direct':
        source_basis=np.empty(0,dtype=np.float32)
        vals=target
        info=[]; codes=[]
        for j in range(2):
            lo,step,q=quantize(vals[:,j],bits)
            info.extend([lo,step]);codes.append(q)
        packed=packbits(np.stack(codes,axis=1),bits)
        rec=np.stack([float(info[2*j])+float(info[2*j+1])*codes[j] for j in range(2)],axis=1)
        mt=0
    elif method=='native_linear':
        cov=source-source.mean(axis=0,keepdims=True)
        _,_,vt=np.linalg.svd(cov,full_matrices=False)
        direction=vt[0]
        if direction[0]<0:direction=-direction
        source_basis=np.concatenate([src_mean,direction.astype(np.float32)])
        vals=(target-np.asarray(src_mean))@direction
        lo,step,q=quantize(vals,bits)
        info=[lo,step];packed=packbits(q,bits)
        rec=np.asarray(src_mean)[None,:]+(float(lo)+float(step)*q)[:,None]*direction[None,:]
        mt=1
    elif method=='mirror_orientation':
        # One nontrivial 2D function chart per block; source anchor paid.
        anchor=src_mean.astype(np.float32)
        source_basis=anchor
        b0=math.atan2(float(anchor[1]),float(anchor[0]))
        angles=np.arctan2(target[:,1],target[:,0])-b0
        angles=(angles+math.pi)%(2*math.pi)-math.pi
        lo,step,q=quantize(angles,bits)
        info=[lo,step];packed=packbits(q,bits)
        rec=rotate2(np.broadcast_to(anchor,(N_TASK,2)),float(lo)+float(step)*q)
        mt=2
    else:
        raise ValueError(method)
    # Optional single private coordinate per task, fitted *only* to support-estimated target updates.
    if private:
        residual=target-rec
        pos=np.argmax(np.abs(residual),axis=1).astype(np.uint8)
        vals=np.asarray([residual[i,p] for i,p in enumerate(pos)],dtype=np.float16)
        rec=np.array(rec,dtype=np.float64,copy=True)
        rec[np.arange(N_TASK),pos]+=vals.astype(np.float64)
        pblob=b''.join(struct.pack('<Be',int(p),float(v)) for p,v in zip(pos,vals))
    else:
        pblob=b''
    prefix=bytes([mt,bits,private])
    basisbytes=np.asarray(source_basis,dtype='<f4').tobytes()
    infobytes=np.asarray(info,dtype='<f4').tobytes()
    blob=prefix+basisbytes+infobytes+packed+pblob
    if len(blob)>=65535:raise ValueError('block codec overflow')
    return dict(method=method,bits=bits,private=private,blob=blob,rec=rec,byte_count=len(blob)+2)


def decode_block(blob):
    mt,bits,private=blob[:3]
    if mt not in (0,1,2) or bits not in BITS or private not in (0,1):raise ValueError('invalid method bits/private')
    basis_n={0:0,1:4,2:2}[mt]
    info_n={0:4,1:2,2:2}[mt]
    offset=3
    basis=np.frombuffer(blob,dtype='<f4',offset=offset,count=basis_n).astype(np.float64);offset+=basis_n*4
    info=np.frombuffer(blob,dtype='<f4',offset=offset,count=info_n).astype(np.float64);offset+=info_n*4
    count=N_TASK*(2 if mt==0 else 1)
    ncode=(count*bits+7)//8
    q=unpackbits(blob[offset:offset+ncode],bits,count).astype(np.float64);offset+=ncode
    if mt==0:
        q=q.reshape(N_TASK,2)
        rec=info[[0,2]][None,:]+info[[1,3]][None,:]*q
    elif mt==1:
        val=info[0]+info[1]*q
        rec=basis[:2][None,:]+val[:,None]*basis[2:][None,:]
    else:
        angle=info[0]+info[1]*q
        rec=rotate2(np.broadcast_to(basis,(N_TASK,2)),angle)
    if private:
        if offset+3*N_TASK!=len(blob):raise ValueError('wrong private state length')
        rec=np.array(rec,copy=True)
        for k in range(N_TASK):
            pos,val=struct.unpack_from('<Be',blob,offset+3*k)
            if pos>1:raise ValueError('invalid private index')
            rec[k,pos]+=val
        offset+=3*N_TASK
    if offset!=len(blob):raise ValueError('trailing block bytes')
    return rec


def make_full_blob(options):
    out=bytearray(HEADER+bytes([N_BLOCK,N_TASK]))
    for c in options:
        out.extend(struct.pack('<H',len(c['blob'])))
        out.extend(c['blob'])
    return bytes(out)


def decode_full(blob):
    if not blob.startswith(HEADER) or len(blob)<7:raise ValueError('codec invalid header')
    if tuple(blob[5:7])!=(N_BLOCK,N_TASK):raise ValueError('codec invalid shape')
    off=7; dec=[]
    for _ in range(N_BLOCK):
        if off+2>len(blob):raise ValueError('truncated length')
        n=struct.unpack_from('<H',blob,off)[0];off+=2
        if off+n>len(blob):raise ValueError('truncated block')
        dec.append(decode_block(blob[off:off+n]));off+=n
    if off!=len(blob):raise ValueError('trailing codec bytes')
    return np.concatenate(dec,axis=1)


def enum_exact(options,G,w_target,limit_bytes):
    nc=len(options[0]);assert nc==18
    idx=np.indices((nc,)*N_BLOCK,dtype=np.int16).reshape(N_BLOCK,-1)
    # The full coupled quadratic deviation to the *support-fitted* function, not audit oracle.
    E=np.stack([np.stack([w_target[:,2*b:2*b+2]-c['rec'] for c in choices]) for b,choices in enumerate(options)])
    singles=[];pairs={}
    for b in range(N_BLOCK):
        gb=G[2*b:2*b+2,2*b:2*b+2]
        singles.append(np.einsum('tki,ij,tkj->t',E[b],gb,E[b])/N_TASK)
        for c in range(b):
            cross=G[2*b:2*b+2,2*c:2*c+2]
            pairs[b,c]=2*np.einsum('tki,ij,s kj->ts',E[b],cross,E[c])/N_TASK
    cost=np.zeros(idx.shape[1],dtype=np.int32)+7
    distortion=np.zeros(idx.shape[1],dtype=np.float64)
    for b in range(N_BLOCK):
        j=idx[b]
        cost+=np.array([x['byte_count'] for x in options[b]])[j]
        distortion+=singles[b][j]
    for (b,c),q in pairs.items():distortion+=q[idx[b],idx[c]]
    # Allow tiny cancellation to produce 0; never negative under PSD beyond rounding.
    if distortion.min() < -1e-8:raise AssertionError('Gram non-PSD?')
    distortion=np.maximum(0.,distortion)
    mirr=np.array([[1 if x['method']=='mirror_orientation' else 0 for x in block] for block in options])
    is_mirror=np.zeros(idx.shape[1],dtype=np.bool_)
    for b in range(N_BLOCK):is_mirror |= mirr[b][idx[b]].astype(bool)
    sums=[0]*N_BLOCK
    for b in range(N_BLOCK):sums[b]=mirr[b][idx[b]]
    allsame=np.ones(idx.shape[1],dtype=np.bool_)
    meths=[np.array([METHODS.index(x['method']) for x in block]) for block in options]
    for b in range(1,N_BLOCK):allsame &= meths[b][idx[b]]==meths[0][idx[0]]
    results=[]
    for B in limit_bytes:
        enabled=cost<=B
        def select(mask):
            take=enabled & mask
            if not np.any(take):return None
            k=int(np.argmin(np.where(take,distortion,np.inf)))
            chosen=[options[b][int(idx[b,k])] for b in range(N_BLOCK)]
            blob=make_full_blob(chosen)
            assert len(blob)==cost[k] and len(blob)<=B
            rec=decode_full(blob)
            if not np.allclose(rec,np.concatenate([o['rec'] for o in chosen],axis=1),atol=1e-6,rtol=1e-6):raise AssertionError('codec mismatch')
            return (float(distortion[k]),int(cost[k]),int(k),chosen,blob,rec)
        allbest=select(np.ones(idx.shape[1],dtype=bool))
        native=select(~is_mirror)
        fixed_method=select(allsame)
        results.append((B,allbest,native,fixed_method))
    return results


def run_world(seed,alpha):
    src,support,truth,probe,test=_source_and_targets(seed,alpha)
    G=probe.T@probe/len(probe)
    off=np.linalg.norm(G-np.diag(np.diag(G)))/np.linalg.norm(G)
    options=[]
    for b in range(N_BLOCK):
        choices=[]
        for method,bits,private in itertools.product(METHODS,BITS,PRIV):
            c=_fit_block(src[:,2*b:2*b+2],support[:,2*b:2*b+2],method,bits,private)
            assert np.allclose(decode_block(c['blob']),c['rec'],atol=1e-6,rtol=1e-6)
            choices.append(c)
        assert len(choices)==18
        options.append(choices)
    direct8=[next(x for x in options[b] if x['method']=='native_direct' and x['bits']==8 and x['private']==0) for b in range(N_BLOCK)]
    native_budget=len(make_full_blob(direct8))
    budgets=[int(math.floor(native_budget*r)) for r in BUDGET_RATIOS]
    result=enum_exact(options,G,support,budgets)
    def audit_mse(rec):return float(np.mean([(np.mean((test[k]@(rec[k]-truth[k]))**2)) for k in range(N_TASK)]))
    def learned_dev_mse(rec):return float(np.mean([np.mean((probe@(rec[k]-support[k]))**2) for k in range(N_TASK)]))
    def decode_p95_ms(blob):
        times=[]
        for _ in range(5):decode_full(blob)
        for _ in range(40):
            tick=time.perf_counter_ns();decode_full(blob)
            times.append((time.perf_counter_ns()-tick)/1e6)
        return float(np.quantile(times,.95))
    out=[]
    for ratio,(budget,allb,nat,same) in zip(BUDGET_RATIOS,result):
        for label,sel in [('joint_with_mirror',allb),('joint_nonmirror',nat),('one_method_joint',same)]:
            if sel is None:
                out.append(dict(seed=seed,alpha=alpha,budget_ratio=ratio,budget_bytes=budget,baseline_direct8_bytes=native_budget,kind=label,feasible=0,bytes=-1,dev_objective=float('nan'),fresh_mse=float('nan'),test_vs_support=float('nan'),mirror_blocks=-1,methods='',qbits='',private='',coupling_offdiag_rel=off,codec_sha256='',cold_decode_p95_ms=float('nan')))
                continue
            val,byte_cost,_,chosen,blob,rec=sel
            out.append(dict(seed=seed,alpha=alpha,budget_ratio=ratio,budget_bytes=budget,baseline_direct8_bytes=native_budget,kind=label,feasible=1,bytes=byte_cost,dev_objective=val,fresh_mse=audit_mse(rec),test_vs_support=learned_dev_mse(rec),mirror_blocks=sum(x['method']=='mirror_orientation' for x in chosen),methods=';'.join(x['method'] for x in chosen),qbits=';'.join(str(x['bits']) for x in chosen),private=';'.join(str(x['private']) for x in chosen),coupling_offdiag_rel=off,codec_sha256=hashlib.sha256(blob).hexdigest(),cold_decode_p95_ms=decode_p95_ms(blob)))
    # Additional independent controls and negative metrics do not reuse test to fit params.
    truthloss=audit_mse(support)
    for item in out:item['support_oracle_mse']=truthloss
    return out


def run(phase,path):
    path=Path(path);path.mkdir(exist_ok=True,parents=True)
    starts=time.perf_counter()
    rows=[]
    for seed in SEEDS[phase]:
        for alpha in ALPHAS:
            rows.extend(run_world(seed,alpha))
        print(json.dumps({'phase':phase,'world':seed,'rows':len([r for r in rows if r['seed']==seed]),'elapsed_s':round(time.perf_counter()-starts,2)}),flush=True)
    columns=list(rows[0])
    with (path/f'{phase}_raw.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=columns);wr.writeheader();wr.writerows(rows)
    return rows


def verdict(rows):
    comparisons=[]
    groups={}
    for r in rows:
        if r['kind'] in ('joint_with_mirror','joint_nonmirror'):
            groups.setdefault((r['seed'],r['alpha'],r['budget_ratio']),{})[r['kind']]=r
    for (seed,alpha,ratio),g in sorted(groups.items()):
        a=g['joint_with_mirror'];b=g['joint_nonmirror']
        if a['feasible'] and b['feasible']:
            # True heldout loss matters. Missing relative denominator is not a pass.
            advance=(b['fresh_mse']-a['fresh_mse'])/max(1e-12,b['fresh_mse'])
            comparisons.append(dict(seed=seed,alpha=alpha,budget_ratio=ratio,improvement=advance,both_feasible=True,mirror_blocks=a['mirror_blocks']))
    return comparisons


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=['dev','fresh','all'],default='all')
    ap.add_argument('--output',default='results/ma1171')
    args=ap.parse_args()
    if args.phase in ('dev','all'):run('dev',args.output)
    if args.phase in ('fresh','all'):run('fresh',args.output)
    out=Path(args.output)
    if (out/'fresh_raw.csv').exists():
        with (out/'fresh_raw.csv').open() as f: rows=list(csv.DictReader(f))
        # Need restore numeric for aggregate.
        for r in rows:
            for z in ('seed','alpha','budget_ratio','fresh_mse','bytes','feasible','mirror_blocks'):
                r[z]=float(r[z]) if z!='seed' else int(r[z])
        com=verdict(rows)
        with (out/'fresh_comparison.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['seed','alpha','budget_ratio','improvement','both_feasible','mirror_blocks']);w.writeheader();w.writerows(com)
        print(json.dumps({'fresh_compared':len(com),'aligned_budget_1_mean_gain':float(np.mean([x['improvement'] for x in com if x['alpha']==1 and x['budget_ratio']==1])) if any(x['alpha']==1 and x['budget_ratio']==1 for x in com) else None,'out':str(out)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
