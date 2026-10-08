"""Small deterministic NumPy screen for MA-297."""
from __future__ import annotations
import hashlib, struct, time
import numpy as np
N_IN,N_OUT,N_ATOMS,N_TASKS,N_EXAMPLES,UPDATES=12,8,24,5,64,200
DTYPE=np.dtype('<f4')

def make_world(seed,condition,support_pairs):
    rng=np.random.default_rng(seed)
    dictionary=rng.normal(size=(N_IN,N_ATOMS)); dictionary/=np.maximum(np.linalg.norm(dictionary,axis=0,keepdims=True),1e-12)
    coeff=np.zeros((N_ATOMS,N_OUT)); active=np.arange(min(N_ATOMS,max(2,support_pairs*2))); coeff[active]=rng.normal(scale=.55,size=(len(active),N_OUT))
    base=dictionary@coeff
    xtrain=rng.normal(size=(N_EXAMPLES,N_IN)); xtest=rng.normal(size=(512,N_IN)); teachers=[]
    for task in range(N_TASKS):
        if condition=='aligned':
            angle=.13*task+rng.normal(scale=.025); theta=np.eye(N_ATOMS)
            for p in range(support_pairs):
                i,j=2*p,2*p+1; c,s=np.cos(angle+.07*p),np.sin(angle+.07*p)
                theta[i,i]=theta[j,j]=c; theta[i,j]=-s; theta[j,i]=s
            teachers.append(dictionary@theta@coeff)
        else: teachers.append(rng.normal(scale=.35,size=(N_IN,N_OUT)))
    # Fixed discovery: select shared atoms by task-0 correlation and freeze.
    phi=xtrain@dictionary; y0=xtrain@teachers[0]
    score=np.linalg.norm(phi.T@y0,axis=1)/(np.linalg.norm(phi,axis=0)+1e-9)
    ids=np.argsort(score)[-max(2,support_pairs*2):]
    return xtrain,xtest,teachers,dictionary[:,ids].copy(),ids

def pack(records):
    out=bytearray(b'MA297\x01'); out.extend(struct.pack('<H',len(records)))
    for name,value in records:
        nb=name.encode(); out.extend(struct.pack('<H',len(nb))); out.extend(nb)
        if isinstance(value,np.ndarray):
            a=np.asarray(value,dtype=DTYPE,order='C'); dtype=a.dtype.str.encode(); out.extend(struct.pack('<B',len(dtype))); out.extend(dtype); out.extend(struct.pack('<B',a.ndim))
            for dim in a.shape: out.extend(struct.pack('<I',int(dim)))
            data=a.tobytes()
        else:
            data=value if isinstance(value,bytes) else value.encode(); out.extend(b'\x00\x01'); out.extend(struct.pack('<I',len(data)))
        out.extend(struct.pack('<I',len(data))); out.extend(data)
    return bytes(out)

def unpack(payload):
    if payload[:6]!=b'MA297\x01': raise ValueError('bad payload magic')
    pos=6; count=struct.unpack_from('<H',payload,pos)[0]; pos+=2; records=[]
    for _ in range(count):
        n=struct.unpack_from('<H',payload,pos)[0]; pos+=2; name=payload[pos:pos+n].decode(); pos+=n
        dn=payload[pos]; pos+=1; dtype=payload[pos:pos+dn].decode() if dn else ''; pos+=dn; ndim=payload[pos]; pos+=1
        shape=[]
        for _ in range(ndim): shape.append(struct.unpack_from('<I',payload,pos)[0]); pos+=4
        nb=struct.unpack_from('<I',payload,pos)[0]; pos+=4; data=payload[pos:pos+nb]; pos+=nb
        value=np.frombuffer(data,dtype=np.dtype(dtype)).reshape(shape).copy() if dtype else data; records.append((name,value))
    if pos!=len(payload): raise ValueError('trailing bytes')
    return records

def serialize(method,shared,basis,params,ids):
    rec=[('method',method)]
    if method!='independent_full': rec += [('shared',shared),('basis',basis)]
    if method!='shared_only': rec.append(('task_ids',np.asarray(ids,dtype=np.float32)))
    rec += [(f'task_{i}',p) for i,p in enumerate(params) if np.asarray(p).size]
    return pack(rec)

def mse(x,w,y): return float(np.mean((x@w-y)**2))

def reconstruct(payload, task_count=None):
    """Rebuild logical task functions from the paid serialized state."""
    records=dict(unpack(payload)); method=records['method'].decode(); base=records.get('shared'); basis=records.get('basis')
    ids=records.get('task_ids'); n=len(ids) if ids is not None else (task_count or N_TASKS); functions=[]
    for t in range(n):
        param=records.get(f'task_{t}',np.zeros((0,N_OUT)))
        if method=='shared_only': fn=base
        elif method=='mirror' and t>0:
            angle=float(param.reshape(-1)[0]); c,s=np.cos(angle),np.sin(angle); rot=np.array([[c,-s],[s,c]])
            base_coef=np.linalg.lstsq(basis,base,rcond=None)[0][:2]
            fn=base+basis[:,:2]@((rot-np.eye(2))@base_coef)
        elif method in ('coeff2','lowrank1') and t>0: fn=base+basis[:,:2]@param
        elif method=='independent_full': fn=param
        else: fn=base
        functions.append(fn)
    return functions

def infer_one(method,x,base,basis,param):
    """Execute the task view from its physical state without a dense weight cache."""
    if method=='mirror' and np.asarray(param).size:
        angle=float(np.asarray(param).reshape(-1)[0]); c,s=np.cos(angle),np.sin(angle); rot=np.array([[c,-s],[s,c]])
        base_coef=np.linalg.lstsq(basis,base,rcond=None)[0][:2]
        return x@base + (x@basis[:,:2])@((rot-np.eye(2))@base_coef)
    if method in ('coeff2','lowrank1') and np.asarray(param).size:
        return x@base + (x@basis[:,:2])@param
    if method=='independent_full': return x@param
    return x@base

def run_world(seed,condition,lr,support_pairs,split):
    xtr,xte,teachers,basis,ids=make_world(seed,condition,support_pairs)
    ytr=[xtr@w for w in teachers]; yte=[xte@w for w in teachers]
    # Same shared initialization for all methods; task-0 support regression on discovered basis.
    base=basis@np.linalg.lstsq(xtr@basis,ytr[0],rcond=None)[0]
    methods=['shared_only','mirror','coeff2','lowrank1','independent_full']; rows=[]
    for method in methods:
        fns=[]; params=[]; previous=None; start=time.perf_counter(); active=0
        for t,teacher in enumerate(teachers):
            if method=='independent_full':
                p=np.linalg.lstsq(xtr,ytr[t],rcond=None)[0]; fn=p
            elif t==0: p,fn=np.zeros((0,N_OUT)),base.copy()
            elif method=='shared_only': p,fn=np.zeros((0,N_OUT)),base.copy()
            elif method=='mirror':
                pair=basis[:,:2]; base_coef=np.linalg.lstsq(xtr@basis,xtr@base,rcond=None)[0][:2]
                train_residual=ytr[t]-xtr@base; pair_acts=xtr@pair
                grid=np.linspace(-np.pi,np.pi,4097); best=(float('inf'),0.,base)
                # Fit the single functional coordinate using training examples only.
                for angle in grid:
                    c,s=np.cos(angle),np.sin(angle); rot=np.array([[c,-s],[s,c]])
                    delta=pair@((rot-np.eye(2))@base_coef)
                    err=np.mean((pair_acts@((rot-np.eye(2))@base_coef)-train_residual)**2)
                    candidate=base+delta
                    if err<best[0]: best=(err,float(angle),candidate)
                p,fn=np.array([best[1]]),best[2]
            else:
                # Ordinary same-basis coefficient control: unconstrained coefficients
                # over precisely the first two discovered atoms.
                pair=basis[:,:2]
                p=np.linalg.lstsq(xtr@pair,ytr[t]-xtr@base,rcond=None)[0]; fn=base+pair@p
                if method=='lowrank1':
                    # Rank-1 output residual over the shared representation.
                    u,svals,vh=np.linalg.svd(p,full_matrices=False); p=(u[:, :1]*svals[:1])@vh[:1, :]; fn=base+pair@p
            fns.append(fn); params.append(p)
            payload=serialize(method,base,basis,params,np.arange(t+1)); old=serialize(method,base,basis,params[:-1],np.arange(t)) if t else b''
            decoded=reconstruct(payload,task_count=t+1)
            replay_diff=max(float(np.max(np.abs(decoded[j]-fns[j]))) for j in range(t+1))
            fns=decoded
            errs=[mse(xte,fns[j],yte[j]) for j in range(t+1)]
            forgetting=0. if previous is None else max([errs[j]-previous[j] for j in range(t)]+[0.])
            previous=errs.copy(); active += (4097*N_EXAMPLES*N_IN*N_OUT if method=='mirror' and t else (N_EXAMPLES*N_IN*N_OUT + N_IN*N_IN*N_OUT if method=='independent_full' else 2*N_EXAMPLES*N_IN*N_OUT if method in ('coeff2','lowrank1') and t else 0))
            rows.append(dict(split=split,seed=seed,condition=condition,learning_rate=lr,method=method,tasks_seen=t+1,mean_seen_mse=float(np.mean(errs)),forgetting_abs=float(forgetting),shared_basis_bytes=len(serialize('shared_only',base,basis,[],[])),inference_payload_bytes=len(payload),incremental_inference_bytes=len(payload)-len(old),train_examples_cumulative=(t+1)*N_EXAMPLES,optimizer_updates_cumulative=0,active_compute_proxy=active,wall_time_s=None,inference_examples_per_s=None,payload_sha256=hashlib.sha256(payload).hexdigest(),retention_task0_mse=errs[0],reconstruction_max_abs_diff=replay_diff))
        elapsed=time.perf_counter()-start
        infer_start=time.perf_counter()
        for _ in range(200):
            for j in range(N_TASKS): _=infer_one(method,xte,base,basis,params[j])
        infer_elapsed=time.perf_counter()-infer_start
        throughput=200*N_TASKS*len(xte)/max(infer_elapsed,1e-12)
        for row in rows[-N_TASKS:]: row['wall_time_s']=elapsed; row['inference_examples_per_s']=throughput
    return rows

def run(seed,split,lr,support_pairs):
    return sum((run_world(seed,c,lr,support_pairs,split) for c in ('aligned','independent')),[])

def deterministic_rows(seed,split,lr,support_pairs):
    rows=run(seed,split,lr,support_pairs); rows2=run(seed,split,lr,support_pairs)
    keys=('mean_seen_mse','inference_payload_bytes','incremental_inference_bytes','payload_sha256')
    if [[r[k] for k in keys] for r in rows]!=[[r[k] for k in keys] for r in rows2]: raise AssertionError('non-deterministic replay')
    return rows
