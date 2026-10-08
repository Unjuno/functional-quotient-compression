import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
D,K,T=32,16,128;R=.25;GRID=np.linspace(-np.pi,np.pi,720,endpoint=False,dtype=np.float32);BUDGETS=(0,2,4,8)
METHODS=['tied','fixed_d2','fixed_d4','fixed_d8','fixed_d16','adaptive_dense','shared_sparse_direct','mirror_sparse','independent']

def basis(seed):
 q,_=np.linalg.qr(np.random.default_rng(seed+1501).normal(size=(D,K)));return q.astype(np.float32)

def world(seed):
 rng=np.random.default_rng(seed);P=basis(seed);base=rng.normal(0,.12,D).astype(np.float32);counts=np.array([0]*48+[2]*32+[4]*32+[8]*16);rng.shuffle(counts);z=np.zeros((T,K),np.float32);groups=[]
 for i,k in enumerate(counts):
  a=rng.uniform(-np.pi,np.pi);z[i,:2]=R*np.array([np.cos(a),np.sin(a)]);idx=rng.choice(np.arange(2,K),int(k),replace=False) if k else np.empty(0,int);z[i,idx]=rng.normal(0,.07,len(idx));groups.append(f'private_{k}')
 targets=base[None,:]+z@P.T;sets=[]
 for i,w in enumerate(targets):
  task=[]
  for n,salt in [(96,1000),(64,2000),(128,3000)]:
   x=np.random.default_rng(seed+salt+i*41).normal(size=(n,D)).astype(np.float32);y=x@w;task.append((x,y))
  sets.append(task)
 return P,base,z,targets,sets,groups

def nmse(x,y,w):return float(np.mean((x@w-y)**2)/(np.mean(y*y)+1e-12))
def fit_full(P,base,x,y,d):return np.linalg.lstsq(x@P[:,:d],y-x@base,rcond=None)[0]
def pick_indices(z,k):return np.argsort(np.abs(z[2:]))[::-1][:k]+2

def fit_mirror(P,base,x,y,idx):
 f=x@P;resid=y-x@base
 if len(idx):
  A=f[:,idx];pinv=np.linalg.pinv(A);ry=resid-A@(pinv@resid);rc=f[:,0]-A@(pinv@f[:,0]);rs=f[:,1]-A@(pinv@f[:,1]);rem=ry[:,None]-R*(rc[:,None]*np.cos(GRID)[None,:]+rs[:,None]*np.sin(GRID)[None,:]);errs=np.mean(rem*rem,axis=0)
 else:
  rem=resid[:,None]-R*(f[:,0,None]*np.cos(GRID)[None,:]+f[:,1,None]*np.sin(GRID)[None,:]);errs=np.mean(rem*rem,axis=0);pinv=A=None
 j=int(np.argmin(errs));a=np.float16(GRID[j]);pair=R*(f[:,0]*np.cos(float(a))+f[:,1]*np.sin(float(a)))
 vals=np.linalg.lstsq(f[:,idx],resid-pair,rcond=None)[0] if len(idx) else np.empty(0)
 return a,vals.astype(np.float16),len(GRID)*len(y)*8+len(y)*len(idx)**2

def save(arrays,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as zf:
  for n in sorted(arrays):
   b=io.BytesIO();np.lib.format.write_array(b,np.ascontiguousarray(arrays[n]),allow_pickle=False);info=zipfile.ZipInfo(n+'.npy',date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED;info.external_attr=0o600<<16;zf.writestr(info,b.getvalue())
 return path.stat().st_size,hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):
 with zipfile.ZipFile(path) as zf:return {n[:-4]:np.load(io.BytesIO(zf.read(n)),allow_pickle=False) for n in sorted(zf.namelist())}

def decode(method,s,i):
 if method=='independent':return s['weights'][i].astype(np.float32)
 base=s['base'].astype(np.float32)
 if method=='tied':return base
 P=s['basis'].astype(np.float32)
 if method in ('shared_sparse_direct','mirror_sparse'):
  co=s['coff'];ioff=s['ioff'];k=int(s['counts'][i]);ids=s['indices'][int(ioff[i]):int(ioff[i+1])].astype(int);v=s['codes'][int(co[i]):int(co[i+1])].astype(np.float32)
  if method=='shared_sparse_direct':z=np.zeros(K,np.float32);z[:2]=v[:2];z[ids]=v[2:]
  else:z=np.zeros(K,np.float32);a=float(s['angles'][i]);z[:2]=float(s['radius'][0])*np.array([np.cos(a),np.sin(a)]);z[ids]=v
  return base+P@z
 d=int(s['dims'][i]);z=s['codes'][int(s['offsets'][i]):int(s['offsets'][i+1])].astype(np.float32);return base+P[:,:d]@z

def encode(method,P,base,targets,sets,groups,threshold,outdir,seed,split):
 events=[];fitops=0;Tn=len(sets);decisions=[]
 if method=='tied':return {'base':base.astype('<f4')},events,0
 if method=='independent':
  w=np.stack([np.linalg.lstsq(x,y,rcond=None)[0] for x,y in [task[0] for task in sets]]).astype('<f4');return {'weights':w},events,Tn*(96*D*D+D**3)
 if method.startswith('fixed_d') or method=='adaptive_dense':
  dims=np.zeros(Tn,np.uint8);offsets=[0];chunks=[]
  for i,task in enumerate(sets):
   x,y=task[0];xv,yv=task[1]
   if method.startswith('fixed_d'):ds=[int(method.split('_d')[1])]
   else:ds=[2,4,8,16]
   chosen=None;v=None;val=float('inf')
   for d in ds:
    zz=fit_full(P,base,x,y,d);score=nmse(xv,yv,base+P[:,:d]@zz);fitops+=96*d*d
    q=zz.astype(np.float16).astype(np.float32);score=nmse(xv,yv,base+P[:,:d]@q)
    if chosen is None:chosen=d;v=zz;val=score
    if method.startswith('fixed_d') or score<=threshold:chosen=d;v=zz;val=score;break
   dims[i]=chosen;chunk=np.asarray(v,dtype='<f2');chunks.append(chunk);offsets.append(offsets[-1]+len(chunk));decisions.append((chosen,val))
  md=int(dims.max());arrays={'base':base.astype('<f4'),'basis':P[:,:md].astype('<f4'),'dims':dims,'offsets':np.asarray(offsets,dtype='<u2'),'codes':np.concatenate(chunks).astype('<f2')}
  return arrays,events,fitops
 # shared/private methods: support fit full coefficients to rank sparse residual coordinates, validate each budget, select smallest.
 counts=np.zeros(Tn,np.uint8);coff=[0];ioff=[0];codes=[];indices=[];angles=np.zeros(Tn,dtype='<f2');modes=np.zeros(Tn,np.uint8);fitops=0
 for i,task in enumerate(sets):
  x,y=task[0];xv,yv=task[1];zfit=fit_full(P,base,x,y,K);fitops+=96*K*K;chosen=None;best=None;bestpair=None;val=float('inf');bestidx=None;bestangle=None
  for k in BUDGETS:
   idx=pick_indices(zfit,k)
   if method=='shared_sparse_direct':
    dims=np.concatenate(([0,1],idx));zz=np.linalg.lstsq(x@P[:,dims],y-x@base,rcond=None)[0];fitops+=96*len(dims)**2
    q=zz.astype(np.float16).astype(np.float32);w=base+P[:,dims]@q;score=nmse(xv,yv,w);residual_fit=zz[2:]
   else:
    a,extra,o=fit_mirror(P,base,x,y,idx);fitops+=o
    w=base+P[:,:K]@np.pad(np.concatenate((R*np.array([np.cos(float(a)),np.sin(float(a))]),extra.astype(np.float32))), (0,K-2-len(extra)))
    if len(idx):
     # decode correctly with selected sparse coordinates only
     z=np.zeros(K,np.float32);z[:2]=R*np.array([np.cos(float(a)),np.sin(float(a))]);z[idx]=extra.astype(np.float32);w=base+P@z
    score=nmse(xv,yv,w);zz=extra
   if score<=threshold:chosen=k;best=residual_fit if method=='shared_sparse_direct' else zz;bestpair=zz[:2] if method=='shared_sparse_direct' else None;val=score;bestidx=idx;bestangle=a if method=='mirror_sparse' else None;break
   if chosen is None:chosen=k;best=residual_fit if method=='shared_sparse_direct' else zz;bestpair=zz[:2] if method=='shared_sparse_direct' else None;val=score;bestidx=idx;bestangle=a if method=='mirror_sparse' else None
  counts[i]=chosen
  if method=='shared_sparse_direct':codes.extend(np.asarray(bestpair,np.float16));codes.extend(np.asarray(best,np.float16))
  else:angles[i]=bestangle;codes.extend(np.asarray(best,np.float16))
  indices.extend(np.asarray(bestidx,dtype=np.uint8));coff.append(len(codes));ioff.append(len(indices));events.append({'task_index':i,'group':groups[i],'choice':'shared_coefficients_plus_sparse_private' if method=='shared_sparse_direct' else 'mirror_angle_plus_sparse_private','private_count':int(chosen),'validation_n_mse':val,'private_fallback':False})
 arrays={'base':base.astype('<f4'),'basis':P.astype('<f4'),'counts':counts,'coff':np.asarray(coff,dtype='<u2'),'ioff':np.asarray(ioff,dtype='<u2'),'codes':np.asarray(codes,dtype='<f2'),'indices':np.asarray(indices,dtype='u1')}
 if method=='mirror_sparse':arrays['angles']=angles;arrays['radius']=np.asarray([R],dtype='<f2')
 return arrays,events,fitops

def run(seed,split,threshold,outdir):
 P,base,z,targets,sets,groups=world(seed);rows=[];allevents=[]
 for method in METHODS:
  t0=time.perf_counter();a,e,fitops=encode(method,P,base,targets,sets,groups,threshold,outdir,seed,split);path=outdir/f'{split}_{seed}_{method}.npz';n,h=save(a,path);s=load(path);per=[];by={}
  for i,task in enumerate(sets):
   err=nmse(task[2][0],task[2][1],decode(method,s,i));per.append(err);by.setdefault(groups[i],[]).append(err)
  st=time.perf_counter()
  for _ in range(3):
   for i,task in enumerate(sets):task[2][0]@decode(method,s,i)
  wall=(time.perf_counter()-st)/3
  avgdim=float(np.mean(s['counts'])) if method in ('shared_sparse_direct','mirror_sparse') else float(np.mean(s['dims'])) if 'dims' in s else (K if method=='independent' else 1)
  rows.append({'condition':split,'world_or_seed':seed,'method':method,'serialized_bytes':n,'support_examples':T*96,'validation_examples':T*64,'test_examples':T*128,'optimizer_updates':0,'fit_compute_proxy':fitops,'active_ops_proxy_per_example':D*(2+avgdim)+(5 if method=='mirror_sparse' else 0),'wall_time_s':wall,'examples_per_s':3*T*128/max(time.perf_counter()-st,1e-12),'primary_metric':'mean_task_normalized_test_mse','primary_value':float(np.mean(per)),'group_test_n_mse':{k:float(np.mean(v)) for k,v in by.items()},'payload_sha256':h,'private_count_mean':avgdim,'allocation_counts':{str(k):int(np.sum(s['counts']==k)) for k in np.unique(s['counts'])} if 'counts' in s else {}})
  allevents.extend({'condition':split,'world':seed,'method':method,**x,'final_payload_bytes':n,'fit_compute_proxy':fitops} for x in e)
 return rows,allevents

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--split',choices=['development','fresh'],required=True);ap.add_argument('--threshold',type=float,default=1e-4);ap.add_argument('--outdir',required=True);ap.add_argument('--json',required=True);a=ap.parse_args();s,e=run(a.seed,a.split,a.threshold,Path(a.outdir));Path(a.json).write_text(json.dumps({'summaries':s,'events':e},indent=2)+'\n')
