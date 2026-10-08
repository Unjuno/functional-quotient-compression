"""Deterministic synthetic task-vector Mirror compression screen (MA-292)."""
from __future__ import annotations
import hashlib, struct, time
import numpy as np

D=128
N_TASKS=8
N_EXAMPLES=512
DTYPE=np.dtype('<f4')
METHODS=('shared_only','mirror_angle','svd_rank2','svd_rank2_fp16','svd_rank1','independent_full')

def _payload_pack(records):
    out=bytearray(b'MA292\x01'); out.extend(struct.pack('<H',len(records)))
    for name,value in records:
        n=name.encode(); out.extend(struct.pack('<H',len(n))); out.extend(n)
        if isinstance(value,np.ndarray):
            a=np.asarray(value)
            if a.dtype.kind=='f':
                target=np.dtype('<f2') if a.dtype.itemsize==2 else DTYPE
                a=np.asarray(a,dtype=target,order='C')
            elif a.dtype.kind in 'ui': a=np.asarray(a,dtype=np.dtype(a.dtype).newbyteorder('<'),order='C')
            else: raise TypeError(a.dtype)
            dt=a.dtype.str.encode(); out.extend(struct.pack('<B',len(dt))); out.extend(dt); out.extend(struct.pack('<B',a.ndim))
            for d in a.shape: out.extend(struct.pack('<I',int(d)))
            data=a.tobytes(order='C')
        else:
            data=value if isinstance(value,bytes) else value.encode(); out.extend(b'\x00\x01'); out.extend(struct.pack('<I',len(data)))
        out.extend(struct.pack('<I',len(data))); out.extend(data)
    return bytes(out)

def _payload_unpack(payload):
    if payload[:6]!=b'MA292\x01': raise ValueError('invalid magic')
    p=6; count=struct.unpack_from('<H',payload,p)[0]; p+=2; out=[]
    for _ in range(count):
        n=struct.unpack_from('<H',payload,p)[0]; p+=2; name=payload[p:p+n].decode(); p+=n
        dn=payload[p]; p+=1; dt=payload[p:p+dn].decode() if dn else ''; p+=dn; ndim=payload[p]; p+=1
        shape=[]
        for _ in range(ndim): shape.append(struct.unpack_from('<I',payload,p)[0]); p+=4
        nb=struct.unpack_from('<I',payload,p)[0]; p+=4; data=payload[p:p+nb]; p+=nb
        value=np.frombuffer(data,dtype=np.dtype(dt)).reshape(shape).copy() if dt else data
        out.append((name,value))
    if p!=len(payload): raise ValueError('trailing bytes')
    return out

def make_world(seed,condition,support_count):
    rng=np.random.default_rng(seed)
    theta0=rng.normal(scale=.2,size=D)
    if condition=='aligned':
        q,_=np.linalg.qr(rng.normal(size=(D,2))); a,b=q[:,0],q[:,1]
        sa=rng.uniform(0,2*np.pi,size=support_count); ta=rng.uniform(0,2*np.pi,size=N_TASKS)
        support=np.stack([.25*(a*np.cos(t)+b*np.sin(t)) + rng.normal(scale=.0025,size=D) for t in sa])
        tasks=np.stack([.25*(a*np.cos(t)+b*np.sin(t)) + rng.normal(scale=.0025,size=D) for t in ta])
    elif condition=='independent':
        support=rng.normal(size=(support_count,D)); support=.25*support/np.linalg.norm(support,axis=1,keepdims=True)
        tasks=rng.normal(size=(N_TASKS,D)); tasks=.25*tasks/np.linalg.norm(tasks,axis=1,keepdims=True)
    else: raise ValueError(condition)
    x=rng.normal(size=(N_EXAMPLES,D))
    return theta0,support,tasks,x

def discover_basis(support,rank=2):
    _,_,vt=np.linalg.svd(support,full_matrices=False)
    return vt[:rank].T.copy()

def _serialize(method,theta0,basis,scale,params):
    rec=[('method',method),('theta0',theta0)]
    if method in ('mirror_angle','svd_rank2','svd_rank2_fp16','svd_rank1'):
        rec.append(('basis',basis))
        if method=='mirror_angle': rec.append(('radius',np.asarray([scale],dtype=np.float32)))
    if method!='shared_only': rec.append(('task_ids',np.arange(len(params),dtype=np.uint8)))
    rec += [(f'task_{i}',p) for i,p in enumerate(params) if np.asarray(p).size]
    return _payload_pack(rec)

def reconstruct(payload,task_count=None):
    r=dict(_payload_unpack(payload)); method=r['method'].decode(); theta=r['theta0']
    ids=r.get('task_ids'); n=len(ids) if ids is not None else (task_count or N_TASKS)
    basis=r.get('basis'); scale=float(r['radius'][0]) if 'radius' in r else None
    out=[]
    for i in range(n):
        p=r.get(f'task_{i}')
        if method=='shared_only': delta=np.zeros(D)
        elif method=='mirror_angle':
            angle=float(np.asarray(p).reshape(-1)[0]); delta=scale*(basis[:,0]*np.cos(angle)+basis[:,1]*np.sin(angle))
        elif method in ('svd_rank2','svd_rank2_fp16'): delta=basis@p
        elif method=='svd_rank1': delta=basis[:,0]*float(np.asarray(p).reshape(-1)[0])
        elif method=='independent_full': delta=p
        else: raise ValueError(method)
        out.append(theta+delta)
    return out

def _fit_payload(method,theta0,basis,scale,tasks):
    if method=='shared_only': params=[]
    elif method=='mirror_angle':
        params=[]
        for d in tasks:
            c=basis.T@d; params.append(np.asarray([np.arctan2(c[1],c[0])],dtype=np.float64))
    elif method=='svd_rank2': params=[basis.T@d for d in tasks]
    elif method=='svd_rank2_fp16': params=[(basis.T@d).astype(np.float16) for d in tasks]
    elif method=='svd_rank1': params=[np.asarray([basis[:,0]@d]) for d in tasks]
    elif method=='independent_full': params=[d.copy() for d in tasks]
    else: raise ValueError(method)
    return _serialize(method,theta0,basis,scale,params)

def _task_mse(x,target,pred):
    y=x@target; yp=x@pred
    return float(np.mean((y-yp)**2))

def _infer(method,x,theta0,basis,scale,param):
    if method=='mirror_angle':
        a=float(np.asarray(param).reshape(-1)[0]); return x@theta0+scale*((x@basis[:,0])*np.cos(a)+(x@basis[:,1])*np.sin(a))
    if method in ('svd_rank2','svd_rank2_fp16'): return x@theta0+(x@basis)@param
    if method=='svd_rank1': return x@theta0+(x@basis[:,0])*float(np.asarray(param).reshape(-1)[0])
    if method=='independent_full': return x@(theta0+param)
    return x@theta0

def run_world(seed,condition,support_count,split):
    theta0,support,tasks,x=make_world(seed,condition,support_count)
    basis2=discover_basis(support,2); basis1=basis2[:,:1]
    support_radius=float(np.mean(np.linalg.norm(support@basis2,axis=1)))
    rows=[]; payloads={}; reconstructions={}; params_by={}; start_all=time.perf_counter()
    for method in METHODS:
        start=time.perf_counter()
        basis=basis1 if method=='svd_rank1' else basis2
        payload=_fit_payload(method,theta0,basis,support_radius,tasks)
        rec=reconstruct(payload,task_count=N_TASKS)
        assert _payload_pack(_payload_unpack(payload))==payload
        table=dict(_payload_unpack(payload)); params_by[method]=[table.get(f'task_{i}',np.zeros(0)) for i in range(N_TASKS)]
        reconstructions[method]=[v-theta0 for v in rec]; payloads[method]=payload
        elapsed=time.perf_counter()-start
        yerrs=[_task_mse(x,theta0+d,th) for d,th in zip(tasks,rec)]
        pairs=[]
        for i in range(N_TASKS):
            for j in range(i+1,N_TASKS):
                for sign in (-1,1):
                    target_delta=tasks[i]+sign*tasks[j]
                    pred_delta=reconstructions[method][i]+sign*reconstructions[method][j]
                    pairs.append(_task_mse(x,theta0+target_delta,theta0+pred_delta))
        total=len(payload); prior=len(_fit_payload(method,theta0,basis,support_radius,tasks[:-1]))
        # Approximate fixed MAC proxy: SVD support discovery, per-task projection/code fit, decode.
        basis_mac=support_count*support_count*D + support_count*D*2
        per_task={'mirror_angle':D*4,'svd_rank2':D*4,'svd_rank2_fp16':D*4,'svd_rank1':D*2,'independent_full':D,'shared_only':0}[method]
        ops=basis_mac+N_TASKS*per_task
        # Coordinate-path forward throughput, measured after payload reconstruction.
        infer_start=time.perf_counter()
        for _ in range(100):
            for i in range(N_TASKS): _=_infer(method,x,theta0,basis,support_radius,params_by[method][i])
        infer_s=time.perf_counter()-infer_start
        ips=100*N_TASKS*N_EXAMPLES/max(infer_s,1e-12)
        rows.append(dict(split=split,seed=seed,condition=condition,support_tasks=support_count,method=method,task_count=N_TASKS,task_output_mse=float(np.mean(yerrs)),max_task_output_mse=float(np.max(yerrs)),pair_arithmetic_mse=float(np.mean(pairs)),serialized_bytes=total,incremental_task_bytes=(total-prior),optimizer_updates=0,task_vectors_seen=support_count+N_TASKS,active_compute_proxy=ops,compression_wall_time_s=elapsed,inference_examples_per_s=ips,payload_sha256=hashlib.sha256(payload).hexdigest(),max_delta_abs_error=max(float(np.max(np.abs((v-theta0)-tasks[i]))) for i,v in enumerate(rec))))
    return rows

def run(seed,split,support_count):
    rows=[]
    for condition in ('aligned','independent'): rows.extend(run_world(seed,condition,support_count,split))
    return rows

def deterministic_rows(seed,split,support_count):
    a=run(seed,split,support_count); b=run(seed,split,support_count)
    keys=('task_output_mse','pair_arithmetic_mse','serialized_bytes','incremental_task_bytes','payload_sha256')
    if [[r[k] for k in keys] for r in a]!=[[r[k] for k in keys] for r in b]: raise AssertionError('non-deterministic replay')
    return a
