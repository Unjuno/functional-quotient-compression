#!/usr/bin/env python3
"""MA-299 sequential allocation screen; deterministic NumPy CPU only."""
from __future__ import annotations
import argparse, json, struct, time, hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
D=16; TASKS=6

def generator():
    g=np.zeros((D,D),np.float64)
    for i in range(0,D,2): g[i,i+1]=-1.; g[i+1,i]=1.
    return g
G=generator()

def rotation(theta): return np.cos(theta)*np.eye(D)+np.sin(theta)*G

def build_world(seed):
    rng=np.random.default_rng(seed); base=rng.normal(0,.25,(D,D)); angles=[0.,.27,-.51,.83]
    mats=[base]
    mats += [base@rotation(m) for m in angles[1:]]
    mats += [rng.normal(0,.25,(D,D))]
    mats += [base@rotation(.43)+rng.normal(0,.015,(D,D))]
    world=[]
    for w in mats:
        sets=[]
        for n in (256,128,512):
            x=rng.normal(size=(n,D)); y=x@w; sets.append((x,y))
        world.append(sets)
    return world

def fit_full(x,y): return np.linalg.lstsq(x,y,rcond=1e-10)[0]

def normalized_mse(pred,y): return float(np.mean((pred-y)**2)/(np.mean(y*y)+1e-12))

def fit_angle(x,y,base):
    y0=x@base; y1=y0@G
    a=float(np.sum(y0*y0)); b=float(np.sum(y1*y1)); c=float(np.sum(y0*y1))
    d=float(np.sum(y0*y)); e=float(np.sum(y1*y))
    grid=np.linspace(-np.pi,np.pi,4097)
    cs=np.cos(grid); sn=np.sin(grid)
    loss=(cs*cs*a+sn*sn*b+2*cs*sn*c-2*cs*d-2*sn*e)
    return float(grid[int(np.argmin(loss))])

def fit_coeff(x,y,base):
    y0=x@base; y1=y0@G
    design=np.column_stack([y0.reshape(-1),y1.reshape(-1)])
    return np.linalg.lstsq(design,y.reshape(-1),rcond=1e-10)[0]

def fit_rank1(x,y,base):
    w=fit_full(x,y); residual=w-base
    u,s,vt=np.linalg.svd(residual,full_matrices=False)
    return u[:,0],float(s[0]),vt[0]

def serialize(records):
    out=bytearray(b"MA299\x01")+bytearray(struct.pack("<I",len(records)))
    for name,value in records:
        a=np.ascontiguousarray(value); name=name.encode()
        if a.dtype.kind=='f': dtype=1; a=a.astype('<f4',copy=False)
        elif a.dtype.kind in 'ui': dtype=2; a=a.astype('u1',copy=False)
        else: raise TypeError(a.dtype)
        out+=struct.pack('<H',len(name))+name+struct.pack('<BB',dtype,a.ndim)
        out+=struct.pack('<'+'I'*a.ndim,*a.shape)+a.tobytes()
    return bytes(out)

def deserialize(payload):
    v=memoryview(payload)
    if bytes(v[:6])!=b"MA299\x01": raise ValueError('bad magic')
    p=6; count=struct.unpack_from('<I',v,p)[0]; p+=4; records={}
    for _ in range(count):
        nl=struct.unpack_from('<H',v,p)[0]; p+=2; name=bytes(v[p:p+nl]).decode(); p+=nl
        dtype,ndim=struct.unpack_from('<BB',v,p); p+=2
        shape=struct.unpack_from('<'+'I'*ndim,v,p) if ndim else (); p+=4*ndim
        dt='<f4' if dtype==1 else 'u1'; n=int(np.prod(shape,dtype=np.int64))*(4 if dtype==1 else 1)
        arr=np.frombuffer(v[p:p+n],dtype=dt).copy().reshape(shape); p+=n; records[name]=arr
    if p!=len(v): raise ValueError('trailing bytes')
    return records

KINDS={'share':0,'view':1,'private':2,'coeff':3,'rank1':4,'full':5}

def serialize_states(method,base,states):
    records=[('meta_method',np.frombuffer(method.encode(),dtype='u1')),
             ('meta_arch',np.frombuffer(b'givens-pair-v1;D=16',dtype='u1'))]
    if method!='independent_full': records.append(('shared_W',base))
    for t,s in enumerate(states):
        records.append((f't{t}_kind',np.array([KINDS[s['kind']]],dtype='u1')))
        if s['kind']=='view': records.append((f't{t}_theta',np.array([s['theta']],np.float32)))
        elif s['kind']=='coeff': records.append((f't{t}_coeff',np.asarray(s['coeff'],np.float32)))
        elif s['kind']=='private': records.append((f't{t}_W',s['W']))
        elif s['kind']=='rank1':
            records.extend([(f't{t}_u',s['u']),(f't{t}_sigma',np.array([s['sigma']],np.float32)),(f't{t}_v',s['v'])])
        elif s['kind']=='full': records.append((f't{t}_W',s['W']))
    return serialize(records)

def predict_records(r,t,x):
    base=r.get('shared_W'); kind=int(r[f't{t}_kind'][0])
    if kind==0: return x@base
    if kind==1: return (x@base)@rotation(float(r[f't{t}_theta'][0]))
    if kind==2 or kind==5: return x@r[f't{t}_W']
    if kind==3:
        c=r[f't{t}_coeff']; y0=x@base; return c[0]*y0+c[1]*(y0@G)
    if kind==4:
        y0=x@base; return y0+((x@r[f't{t}_u'])*float(r[f't{t}_sigma'][0]))[:,None]*r[f't{t}_v'][None,:]
    raise ValueError(kind)

def predict_payload(payload,t,x): return predict_records(deserialize(payload),t,x)

def candidate_for(method,t,xtr,ytr,xv,yv,base,threshold):
    if method=='independent_full': return {'kind':'full','W':fit_full(xtr,ytr)}
    if t==0: return {'kind':'share'}
    if method in ('hard_tie','always_mirror'):
        if method=='hard_tie': return {'kind':'share'}
        angle=fit_angle(xtr,ytr,base); return {'kind':'view','theta':angle}
    if method=='split_tie':
        return {'kind':'share'} if normalized_mse(xv@base,yv)<=threshold else {'kind':'private','W':fit_full(xtr,ytr)}
    if method=='mirror_split':
        angle=fit_angle(xtr,ytr,base); pred=(xv@base)@rotation(angle)
        return {'kind':'view','theta':angle} if normalized_mse(pred,yv)<=threshold else {'kind':'private','W':fit_full(xtr,ytr)}
    if method=='coeff_split':
        coeff=fit_coeff(xtr,ytr,base); y0=xv@base; pred=coeff[0]*y0+coeff[1]*(y0@G)
        return {'kind':'coeff','coeff':coeff} if normalized_mse(pred,yv)<=threshold else {'kind':'private','W':fit_full(xtr,ytr)}
    if method=='rank1_split':
        u,s,v=fit_rank1(xtr,ytr,base); y0=xv@base; pred=y0+((xv@u)*s)[:,None]*v[None,:]
        return {'kind':'rank1','u':u,'sigma':s,'v':v} if normalized_mse(pred,yv)<=threshold else {'kind':'private','W':fit_full(xtr,ytr)}
    raise ValueError(method)

METHODS=['hard_tie','split_tie','coeff_split','rank1_split','mirror_split','always_mirror','independent_full']

def run_world(seed,split,threshold):
    world=build_world(seed); rows=[]
    for method in METHODS:
        tic=time.perf_counter(); base=fit_full(*world[0][0]); states=[]; prefix_bytes=[]; payload=None
        for t,sets in enumerate(world):
            xtr,ytr=sets[0]; xv,yv=sets[1]
            if method=='independent_full': state=candidate_for(method,t,xtr,ytr,xv,yv,base,threshold)
            elif method=='hard_tie': state={'kind':'share'}
            else: state=candidate_for(method,t,xtr,ytr,xv,yv,base,threshold)
            states.append(state); payload=serialize_states(method,base,states)
            prefix_bytes.append(len(payload))
        fit_s=time.perf_counter()-tic
        # All primary metrics and timing replay the exact serialized payload state.
        loaded=deserialize(payload)
        per=[]; validation_per=[]; test_sets=[s[2] for s in world]
        for t,sets in enumerate(world): validation_per.append(normalized_mse(predict_records(loaded,t,sets[1][0]),sets[1][1]))
        for t,(xt,yt) in enumerate(test_sets): per.append(normalized_mse(predict_records(loaded,t,xt),yt))
        max_seen=max(per)
        # Sequentially frozen state means prior task predictions must be bit-stable.
        before=[]; stateprefix=[]
        for t,s in enumerate(world):
            if method=='independent_full': stateprefix.append(states[t])
            else: stateprefix.append(states[t])
            before.append(normalized_mse(predict_records(loaded,t,s[2][0]),s[2][1]))
        retained=max((abs(per[t]-before[t]) for t in range(TASKS)),default=0.)
        t0=time.perf_counter()
        for _ in range(12):
            for t,(xt,yt) in enumerate(test_sets): _=predict_records(loaded,t,xt)
        elapsed=(time.perf_counter()-t0)/(12*TASKS*512)
        private=sum(s['kind']=='private' for s in states)
        views=sum(s['kind']=='view' for s in states)
        coeffs=sum(s['kind']=='coeff' for s in states)
        residuals=sum(s['kind']=='rank1' for s in states)
        # MAC-equivalent fit proxy includes every arriving task; grid objective reductions
        # are conservatively charged as 32 scalar operations per angle candidate.
        base_fit=256*D*D
        angle_fit=2*256*D*D+4097*32
        if method=='hard_tie': fit_mac=base_fit
        elif method=='split_tie': fit_mac=base_fit+5*base_fit
        elif method=='mirror_split': fit_mac=base_fit+5*angle_fit+base_fit
        elif method=='always_mirror': fit_mac=base_fit+5*angle_fit
        elif method=='coeff_split': fit_mac=base_fit+5*(2*256*D*D+256*16*2)+base_fit
        elif method=='rank1_split': fit_mac=base_fit+5*(2*base_fit+2*D*D*D)
        elif method=='independent_full': fit_mac=6*base_fit
        infer_mac=sum(512*(512 if s['kind'] in ('view','coeff') else (384 if s['kind']=='rank1' else 256)) for s in states)
        deltas=[prefix_bytes[0]]+[prefix_bytes[i]-prefix_bytes[i-1] for i in range(1,TASKS)]
        rows.append({'seed':seed,'split':split,'method':method,'threshold':threshold,'normalized_validation_mse_by_task':validation_per,'max_validation_normalized_mse':max(validation_per),'normalized_test_mse_by_task':per,'mean_normalized_test_mse':float(np.mean(per)),'max_normalized_test_mse':max_seen,'serialized_bytes':len(payload),'incremental_bytes_by_task':deltas,'private_splits':private,'view_codes':views,'coefficient_codes':coeffs,'rank1_residuals':residuals,'retention_abs_change':retained,'train_examples':TASKS*256,'validation_examples':TASKS*128,'test_examples':TASKS*512,'optimizer_updates':0,'fit_compute_proxy':fit_mac,'inference_mac_proxy':infer_mac,'fit_wall_seconds':fit_s,'inference_seconds_per_example':elapsed,'inference_examples_per_second':1./elapsed,'operator_workspace_bytes':1024 if method in ('mirror_split','always_mirror','coeff_split') else 0,'payload_sha256':hashlib.sha256(payload).hexdigest()})
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--split',choices=['development','fresh','amended_fresh'],required=True); p.add_argument('--threshold',type=float,default=0.01); p.add_argument('--out',type=Path,default=ROOT/'source'/'results.json'); a=p.parse_args()
    seeds=[29901,29902] if a.split=='development' else ([29911,29912,29913] if a.split=='fresh' else [29921,29922,29923])
    rows=[]
    for seed in seeds: rows.extend(run_world(seed,a.split,a.threshold))
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps({'split':a.split,'seeds':seeds,'threshold':a.threshold,'rows':rows},indent=2)+'\n')
    print(f'wrote {len(rows)} rows to {a.out}')
if __name__=='__main__': main()
