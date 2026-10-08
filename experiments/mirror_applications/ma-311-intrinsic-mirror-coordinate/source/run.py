import argparse,csv,json,struct,time
from pathlib import Path
import numpy as np
T,D,K,ALIGNED=8,128,32,5
SCALE=.2/127
METHODS=['tied','intrinsic','mirror','independent']

def basis(seed):
    rng=np.random.default_rng(seed);q,_=np.linalg.qr(rng.normal(size=(D,K)));return q[:,:K]

def rotate(z,theta):
    out=np.empty_like(z);omega=np.arange(1,K//2+1);a=theta*omega;c=np.cos(a);s=np.sin(a)
    x=z[0::2];y=z[1::2];out[0::2]=c*x-s*y;out[1::2]=s*x+c*y;return out

def world(seed):
    rng=np.random.default_rng(seed);b=basis(seed+99);base=rng.normal(size=D)/np.sqrt(D);u=rng.normal(size=K)/np.sqrt(K)
    theta=rng.uniform(-.12,.12,ALIGNED-1);zs=[np.zeros(K)]+[rotate(u,a) for a in theta]+[rng.normal(size=K)/np.sqrt(K) for _ in range(T-ALIGNED)]
    targets=[base+b@z for z in zs];samples=[]
    for i,w in enumerate(targets):
        task=[]
        for n,salt in [(512,100),(256,200),(512,300)]:
            x=np.random.default_rng(seed+salt+i*31).normal(size=(n,D));y=x@w;task.append((x,y))
        samples.append(task)
    return b,base,samples

def fit_intrinsic(b,base,x,y):return np.linalg.lstsq(x@b,y-x@base,rcond=None)[0]
def fit_full(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def nmse(x,y,w):return float(np.mean((x@w-y)**2)/(np.mean(y*y)+1e-12))

def encode(method,seed,base,u,records,private,full):
    cfg={'method':method,'tasks':T,'dim':D,'intrinsic_dim':K,'projection_seed':seed+99,'projection_recipe':'numpy_default_rng_normal_qr_v1','records':records,'has_base':method!='independent','has_u':method=='mirror'}
    arr=bytearray()
    if method!='independent':arr.extend(np.asarray(base,dtype='<f4').tobytes())
    if method=='mirror':arr.extend(np.asarray(u,dtype='<f2').tobytes())
    for i,r in enumerate(records):
        if r['kind']=='angle':arr.extend(np.asarray([r['code']],dtype='i1').tobytes())
        elif r['kind']=='intrinsic':arr.extend(np.asarray(private[i],dtype='<f2').tobytes())
        elif r['kind']=='full':arr.extend(np.asarray(full[i],dtype='<f4').tobytes())
    head=json.dumps(cfg,separators=(',',':'),sort_keys=True).encode();return struct.pack('<H',len(head))+head+bytes(arr)

def decode(blob):
    n=struct.unpack('<H',blob[:2])[0];cfg=json.loads(blob[2:2+n]);buf=memoryview(blob)[2+n:];off=0
    def take(dtype,count):
        nonlocal off
        a=np.frombuffer(buf,dtype=dtype,count=count,offset=off).copy();off+=a.nbytes;return a
    b=basis(cfg['projection_seed'])
    base=take('<f4',D) if cfg['has_base'] else None
    u=take('<f2',K).astype(np.float32) if cfg['has_u'] else None
    vec=[];zrows=[]
    for i,r in enumerate(cfg['records']):
        if r['kind']=='base':v=base;z=np.zeros(K)
        elif r['kind']=='anchor':z=u.astype(np.float64);v=base+b@z
        elif r['kind']=='angle':z=rotate(u.astype(np.float64),float(take('i1',1)[0])*SCALE);v=base+b@z
        elif r['kind']=='intrinsic':z=take('<f2',K).astype(np.float32);v=base+b@z
        else:v=take('<f4',D);z=None
        vec.append(v);zrows.append(z)
    return np.stack(vec),cfg,zrows

def fit_angle(u,z):
    grid=np.linspace(-.2,.2,801);best=(float('inf'),0.)
    for a in grid:
        err=np.mean((rotate(u,a)-z)**2)
        if err<best[0]:best=(err,a)
    code=int(np.clip(np.rint(best[1]/SCALE),-127,127));return code,rotate(u,code*SCALE),len(grid)*K

def run(seed,threshold):
    b,_,data=world(seed);summaries=[];events=[]
    for method in METHODS:
        st=time.time();records=[];private={};fullmap={};search_ops=0;accepted=0
        base=fit_full(*data[0][0]) if method!='independent' else None
        if method in ('intrinsic','mirror'):
            intr=[np.zeros(K)]+[fit_intrinsic(b,base,*data[i][0]) for i in range(1,T)]
            u=intr[1].copy()
        else:intr=None;u=None
        for i in range(T):
            xval,yval=data[i][1]
            if method=='independent':
                fullmap[i]=fit_full(*data[i][0]);records.append({'kind':'full'});choice='full_private';alloc=True;val=nmse(xval,yval,fullmap[i]);ops=512*D*D
            elif i==0:records.append({'kind':'base'});choice='shared_base';alloc=False;val=nmse(xval,yval,base);ops=512*D*D
            elif method=='tied':records.append({'kind':'base'});choice='hard_tie';alloc=False;val=nmse(xval,yval,base);ops=0
            elif method=='intrinsic':records.append({'kind':'intrinsic'});private[i]=intr[i];choice='task_intrinsic_vector';alloc=True;val=nmse(xval,yval,base+b@intr[i]);ops=512*K*K
            else:
                if i==1:code=0;zview=u.copy();ops=0
                else:code,zview,ops=fit_angle(u,intr[i]);search_ops+=ops
                val=nmse(xval,yval,base+b@zview)
                if val<=threshold:
                    if i==1:records.append({'kind':'anchor'});choice='shared_intrinsic_anchor';accepted+=1
                    else:records.append({'kind':'angle','code':code});choice='mirror_angle';accepted+=1
                    alloc=False
                else:records.append({'kind':'intrinsic'});private[i]=intr[i];choice='private_intrinsic_fallback';alloc=True;ops+=512*K*K
            blob=encode(method,seed,base,u,records,private,fullmap);vecs,_,_=decode(blob)
            events.append({'condition':'fresh' if seed>=31122 else 'development','world':seed,'method':method,'task_index':i,'task_kind':'aligned' if i<ALIGNED else 'unrelated','choice':choice,'allocated_private':alloc,'validation_n_mse':val,'cumulative_payload_bytes':len(blob),'search_ops_proxy':ops})
        blob=encode(method,seed,base,u,records,private,fullmap);vecs,_,_=decode(blob)
        errs=[nmse(*data[i][2],vecs[i]) for i in range(T)];mse=float(np.mean(errs));t0=time.time()
        for _ in range(5):
            v,_,_=decode(blob)
            for i in range(T):data[i][2][0]@v[i]
        throughput=5*sum(len(d[2][1]) for d in data)/(time.time()-t0)
        intrinsic_fit_ops=(T-1)*512*K*K if method in ('intrinsic','mirror') else 0
        basis_ops=D*K*K if method in ('intrinsic','mirror') else 0
        opsproxy=(T*512*D*D if method=='independent' else (512*D*D+intrinsic_fit_ops+basis_ops+search_ops))
        summaries.append({'condition':'fresh' if seed>=31122 else 'development','world_or_seed':seed,'method':method,'serialized_bytes':len(blob),'train_tokens_or_examples':T*512,'optimizer_updates':0,'active_compute_proxy':opsproxy,'wall_time_s':time.time()-st,'primary_metric':'mean_normalized_heldout_mse','primary_value':mse,'secondary_metric':'payload_decode_and_inference_examples_per_s','secondary_value':throughput,'status_note':f'accepted_views={accepted}; private_intrinsic={sum(r["kind"]=="intrinsic" for r in records)}; angle_search_ops={search_ops}; projection_basis_bytes=0 (paid seed+recipe; deterministic QR reconstruction); per_task_mse={errs}'})
    return summaries,events

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--world',type=int,required=True);p.add_argument('--threshold',type=float,required=True);p.add_argument('--intrinsic-dim',type=int,default=K);p.add_argument('--out',required=True);p.add_argument('--events',required=True);a=p.parse_args();K=a.intrinsic_dim;ss,ee=run(a.world,a.threshold);Path(a.out).write_text(json.dumps(ss,indent=2));Path(a.events).write_text(json.dumps(ee,indent=2));
 for r in ss:print(r['world_or_seed'],r['method'],f"nMSE={r['primary_value']:.6g}",f"bytes={r['serialized_bytes']}")
