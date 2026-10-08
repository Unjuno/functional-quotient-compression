"""MA-315 fixed shared-orbit plus private-coordinate frontier screen."""
from __future__ import annotations

import hashlib, io, time, zipfile
from pathlib import Path
import numpy as np

D,K,T=32,8,256
NS,NV,NT=32,32,64
FRACTIONS=(0.0,0.25,0.5,1.0)
RESIDUAL_RANKS=(0,2,4,6)
DIRECT_DIMS=(2,4,6,8)
THRESHOLD=1e-4
RADIUS=np.float32(0.25)
GRID=np.linspace(-np.pi,np.pi,720,endpoint=False,dtype=np.float32)
METHODS=("tied","adaptive_direct","mirror_sparse","fixed_said8","independent")


def make_world(seed:int, private_fraction:float):
    if private_fraction not in FRACTIONS: raise ValueError(private_fraction)
    rng=np.random.default_rng(seed)
    theta0=rng.normal(0,0.1,D).astype(np.float32)
    q,_=np.linalg.qr(rng.normal(size=(D,K)))
    basis=q.astype(np.float32)
    private=np.zeros(T,dtype=bool)
    npriv=round(T*private_fraction)
    private[rng.permutation(T)[:npriv]]=True
    angles=rng.uniform(-np.pi,np.pi,T).astype(np.float32)
    residuals=np.zeros((T,K-2),dtype=np.float32)
    residuals[private]=rng.normal(0,0.12,size=(npriv,K-2)).astype(np.float32)
    coeff=np.zeros((T,K),dtype=np.float32)
    coeff[:,0]=RADIUS*np.cos(angles);coeff[:,1]=RADIUS*np.sin(angles)
    coeff[:,2:]=residuals
    targets=theta0[None,:]+coeff@basis.T
    xs=[];ys=[]
    for n in (NS,NV,NT):
        x=rng.normal(size=(T,n,D)).astype(np.float32)
        y=np.einsum('tnd,td->tn',x,targets,optimize=True)
        xs.append(x);ys.append(y)
    return theta0,basis,private,angles,coeff,targets,*xs,*ys


def _norm_mse(x,y,theta):
    pred=x@theta;e=float(np.mean((pred-y)**2))
    return e,e/max(float(np.mean(y*y)),1e-12)


def _fit_direct(x,y,theta0,basis,d):
    resid=y-x@theta0
    z=np.linalg.lstsq(x@basis[:,:d],resid,rcond=None)[0]
    return z.astype(np.float16)


def _fit_mirror(x,y,theta0,basis,k):
    common=x@basis[:,:2]
    private=x@basis[:,2:2+k] if k else np.empty((len(x),0),dtype=np.float32)
    target=y-x@theta0
    cg,sg=np.cos(GRID),np.sin(GRID)
    a=RADIUS*common[:,0];b=RADIUS*common[:,1];target_fit=target.copy()
    ops=NS*len(GRID)*6
    if k:
        # Profile out residual coefficients for every candidate angle by
        # projecting both the common plane and target off the residual span.
        qa=np.linalg.lstsq(private,a,rcond=None)[0];qb=np.linalg.lstsq(private,b,rcond=None)[0]
        qy=np.linalg.lstsq(private,target,rcond=None)[0]
        a=a-private@qa;b=b-private@qb;target_fit=target-private@qy
        ops+=3*(NS*k*k+k**3)+3*NS*k
    errors=np.mean((a[:,None]*cg+b[:,None]*sg-target_fit[:,None])**2,axis=0)
    angle=GRID[int(np.argmin(errors))]
    if k:
        base=RADIUS*(common[:,0]*np.cos(angle)+common[:,1]*np.sin(angle))
        z=np.linalg.lstsq(private,target-base,rcond=None)[0]
    else:z=np.empty(0,dtype=np.float32)
    return np.float16(angle),z.astype(np.float16),ops


def npy_bytes(a):
    b=io.BytesIO();np.lib.format.write_array(b,np.ascontiguousarray(a),allow_pickle=False);return b.getvalue()


def save_payload(path:Path,state):
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(state):
            zi=zipfile.ZipInfo(name+'.npy',date_time=(1980,1,1,0,0,0));zi.compress_type=zipfile.ZIP_STORED;zi.external_attr=0o600<<16
            zf.writestr(zi,npy_bytes(state[name]))
    return path.stat().st_size


def load_payload(path):
    with zipfile.ZipFile(path) as zf:
        return {n[:-4]:np.load(io.BytesIO(zf.read(n)),allow_pickle=False) for n in sorted(zf.namelist())}


def decode(method,state,t):
    if method=='tied':return state['theta0']
    if method=='independent':return state['weights'][t]
    theta0=state['theta0'];basis=state['basis']
    if method=='fixed_said8':
        z=state['codes'][t].astype(np.float32)
    elif method=='adaptive_direct':
        d=int(state['dims'][t]);off=int(np.sum(state['dims'][:t]));z=np.zeros(basis.shape[1],dtype=np.float32);z[:d]=state['codes'][off:off+d].astype(np.float32)
    elif method=='mirror_sparse':
        k=int(state['dims'][t]);off=int(np.sum(state['dims'][:t]+1));z=np.zeros(basis.shape[1],dtype=np.float32)
        a=float(state['codes'][off]);r=float(state['radius'][0]);z[0]=r*np.cos(a);z[1]=r*np.sin(a)
        if k:z[2:2+k]=state['codes'][off+1:off+1+k].astype(np.float32)
    else:raise ValueError(method)
    return theta0+basis@z


def _fit_adaptive(method,theta0,basis,xtr,ytr,xval,yval):
    codes=[];dims=np.zeros(T,dtype=np.uint8);ops=0
    for t in range(T):
        choices=[]
        if method=='adaptive_direct':
            for d in DIRECT_DIMS:
                z=_fit_direct(xtr[t],ytr[t],theta0,basis,d)
                theta=theta0+basis[:,:d]@z.astype(np.float32)
                choices.append((d,z,_norm_mse(xval[t],yval[t],theta)[1]))
                ops+=NS*d*d+d**3
            d,z,_=next((q for q in choices if q[2]<=THRESHOLD),choices[-1])
            dims[t]=d;codes.append(z)
        else:
            for k in RESIDUAL_RANKS:
                a,z,fitops=_fit_mirror(xtr[t],ytr[t],theta0,basis,k);ops+=fitops+NS*k*k+k**3
                theta=theta0+basis[:,:2]@np.array([RADIUS*np.cos(float(a)),RADIUS*np.sin(float(a))],dtype=np.float32)
                if k:theta+=basis[:,2:2+k]@z.astype(np.float32)
                choices.append((k,(a,z),_norm_mse(xval[t],yval[t],theta)[1]))
            k,(a,z),_=next((q for q in choices if q[2]<=THRESHOLD),choices[-1])
            dims[t]=k;codes.append(np.concatenate((np.array([a],dtype=np.float16),z)))
    cursor=sum(len(c) for c in codes)
    flat=np.concatenate(codes).astype(np.float16) if cursor else np.empty(0,dtype=np.float16)
    if method=='adaptive_direct':
        maxd=int(dims.max());state={'theta0':theta0,'basis':basis[:,:maxd],'dims':dims,'codes':flat}
    else:
        maxk=int(dims.max());state={'theta0':theta0,'basis':basis[:,:2+maxk],'radius':np.array([RADIUS],dtype=np.float32),'dims':dims,'codes':flat}
    return state,ops


def make_state(method,theta0,basis,xtr,ytr,xval,yval):
    if method=='tied':return {'theta0':theta0},0
    if method=='independent':
        w=np.stack([np.linalg.lstsq(xtr[t],ytr[t],rcond=None)[0] for t in range(T)]).astype(np.float32)
        return {'weights':w},T*(NS*D*D+D**3)
    if method=='fixed_said8':
        c=np.stack([np.linalg.lstsq(xtr[t]@basis,ytr[t]-xtr[t]@theta0,rcond=None)[0] for t in range(T)]).astype(np.float32)
        return {'theta0':theta0,'basis':basis,'codes':c},T*(NS*64+512)
    return _fit_adaptive(method,theta0,basis,xtr,ytr,xval,yval)


def evaluate(method,state,x,y,fitops):
    ts=time.perf_counter();vals=[]
    for t in range(T):vals.append(_norm_mse(x[t],y[t],decode(method,state,t))[1])
    wall=time.perf_counter()-ts
    if method=='adaptive_direct':dims=state['dims'];ops=float(np.mean([D+D*int(d)+int(d) for d in dims]))
    elif method=='mirror_sparse':dims=state['dims'];ops=float(np.mean([D+D*(2+int(k))+(2+int(k))+6 for k in dims]))
    elif method=='fixed_said8':dims=np.full(T,8);ops=D+D*K+K
    elif method=='independent':dims=np.full(T,D);ops=D
    else:dims=np.zeros(T);ops=D
    n=T*x.shape[1]
    return {'normalized_mse':float(np.mean(vals)),'max_task_normalized_mse':float(np.max(vals)),
            'normalized_mse_private':None,'test_examples':n,'optimizer_updates':0,'fit_compute_proxy':int(fitops),
            'active_ops_per_example':ops,'active_ops_total':int(ops*n),'wall_time_s':wall,
            'examples_per_s':n/max(wall,1e-12),'mean_selected_residual_dim':float(np.mean(dims))}


def run_world(seed,fraction,outdir):
    theta0,basis,private,true_angles,true_coeff,targets,xtr,xval,xte,ytr,yval,yte=make_world(seed,fraction)
    rows=[];events=[]
    for method in METHODS:
        start=time.perf_counter();state,fitops=make_state(method,theta0,basis,xtr,ytr,xval,yval);enc=time.perf_counter()-start
        path=outdir/f'{seed}_p{int(fraction*100):03d}_{method}.npz';nbytes=save_payload(path,state);loaded=load_payload(path)
        m=evaluate(method,loaded,xte,yte,fitops)
        mpriv=[];malign=[]
        for t in range(T):
            val=_norm_mse(xte[t],yte[t],decode(method,loaded,t))[1]
            (mpriv if private[t] else malign).append(val)
            if method=='adaptive_direct':choice=int(loaded['dims'][t])
            elif method=='mirror_sparse':choice=2+int(loaded['dims'][t])
            elif method=='fixed_said8':choice=8
            else:choice=0
            events.append({'seed':seed,'private_fraction':fraction,'method':method,'task':t,'is_private':int(private[t]),'selected_intrinsic_dimension':choice,'test_normalized_mse':val})
        if mpriv:m['private_task_normalized_mse']=float(np.mean(mpriv))
        else:m['private_task_normalized_mse']='NA'
        if malign:m['aligned_task_normalized_mse']=float(np.mean(malign))
        else:m['aligned_task_normalized_mse']='NA'
        rows.append({'seed':seed,'private_fraction':fraction,'method':method,'serialized_bytes':nbytes,
                     'tensor_bytes':sum(a.nbytes for a in state.values()),'support_examples':T*NS,
                     'validation_examples':T*NV,'test_examples':T*NT,'optimizer_updates':0,
                     'fit_compute_proxy':m['fit_compute_proxy'],'active_ops_per_example':m['active_ops_per_example'],
                     'active_ops_total':m['active_ops_total'],'encode_wall_time_s':enc,'wall_time_s':m['wall_time_s'],
                     'examples_per_s':m['examples_per_s'],'normalized_mse':m['normalized_mse'],
                     'max_task_normalized_mse':m['max_task_normalized_mse'],'mean_selected_residual_dim':m['mean_selected_residual_dim'],
                     'private_task_normalized_mse':m['private_task_normalized_mse'],'aligned_task_normalized_mse':m['aligned_task_normalized_mse'],
                     'payload_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows,events
