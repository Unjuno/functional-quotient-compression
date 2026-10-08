import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
D,K,T,NG=32,16,160,4
RADIUS=.25
GRID=np.linspace(-np.pi,np.pi,720,endpoint=False,dtype=np.float32)
METHODS=['tied','fixed_d2','fixed_d4','fixed_d8','fixed_d16','adaptive_direct','adaptive_mirror','independent']

def basis(seed):
 rng=np.random.default_rng(seed+991);q,_=np.linalg.qr(rng.normal(size=(D,K)));return q[:,:K].astype(np.float32)

def make_world(seed):
 rng=np.random.default_rng(seed);P=basis(seed);base=rng.normal(0,.15,D).astype(np.float32);z=np.zeros((T,K),np.float32);groups=[]
 for i in range(128):
  g=i//32;dim=(2,4,8,16)[g];angle=rng.uniform(-np.pi,np.pi);z[i,0]=RADIUS*np.cos(angle);z[i,1]=RADIUS*np.sin(angle)
  if dim>2:z[i,2:dim]=rng.normal(0,.04,dim-2)
  groups.append(f'aligned_d{dim}')
 z[128:]=rng.normal(0,.12,(32,K));groups += ['unrelated']*32
 targets=base[None,:]+z@P.T; sets=[]
 for i,w in enumerate(targets):
  task=[]
  for n,salt in [(64,1000),(64,2000),(128,3000)]:
   x=np.random.default_rng(seed+salt+i*37).normal(size=(n,D)).astype(np.float32);y=x@w;task.append((x,y))
  sets.append(task)
 return P,base,z,targets,sets,groups

def nmse(x,y,w):
 return float(np.mean((x@w-y)**2)/(np.mean(y*y)+1e-12))

def fit_direct(P,base,x,y,d):
 q=P[:,:d];z=np.linalg.lstsq(x@q,y-x@base,rcond=None)[0];return z

def fit_mirror(P,base,x,y,d):
 f=x@P[:,:d];resid=y-x@base;cos=np.cos(GRID);sin=np.sin(GRID)
 if d>2:
  A=f[:,2:d];pinv=np.linalg.pinv(A);ry=resid-A@(pinv@resid);rc=f[:,0]-A@(pinv@f[:,0]);rs=f[:,1]-A@(pinv@f[:,1])
  remainder=ry[:,None]-RADIUS*(rc[:,None]*cos[None,:]+rs[:,None]*sin[None,:]);err=np.mean(remainder*remainder,axis=0)
 else:
  remainder=resid[:,None]-RADIUS*(f[:,0,None]*cos[None,:]+f[:,1,None]*sin[None,:]);err=np.mean(remainder*remainder,axis=0);pinv=None;A=None
 j=int(np.argmin(err));angle=np.float16(GRID[j])
 if d>2:extra=pinv@(resid-RADIUS*(f[:,0]*cos[j]+f[:,1]*sin[j]));ops=len(y)*(d-2)**2
 else:extra=np.empty(0);ops=0
 return angle,np.asarray(extra,dtype=np.float16),len(GRID)*len(y)*8+ops

def pack_arrays(arrays,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as zf:
  for name in sorted(arrays):
   b=io.BytesIO();np.lib.format.write_array(b,np.ascontiguousarray(arrays[name]),allow_pickle=False)
   info=zipfile.ZipInfo(name+'.npy',date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED;info.external_attr=0o600<<16;zf.writestr(info,b.getvalue())
 return path.stat().st_size,hashlib.sha256(path.read_bytes()).hexdigest()

def load_arrays(path):
 with zipfile.ZipFile(path) as zf:return {n[:-4]:np.load(io.BytesIO(zf.read(n)),allow_pickle=False) for n in sorted(zf.namelist())}

def decode(method,s,task):
 if method=='independent':return s['weights'][task].astype(np.float32)
 base=s['base'].astype(np.float32)
 if method=='tied':return base
 P=s['basis'].astype(np.float32);d=int(s['dims'][task]);off=s['offsets'];v=s['codes'][int(off[task]):int(off[task+1])].astype(np.float32)
 if method=='adaptive_mirror' and int(s['modes'][task])==0:
  a=float(v[0]);z=np.empty(d,np.float32);z[:2]=float(s['radius'][0])*np.array([np.cos(a),np.sin(a)],np.float32);z[2:]=v[1:]
 else:z=v
 return base+P[:,:d]@z

def encode_state(method,P,base,targets,sets,groups,threshold):
 dims=np.zeros(T,np.uint8);modes=np.zeros(T,np.uint8);chunks=[];offsets=[0];events=[];fitops=0;selected=[]
 if method=='tied':arrays={'base':base.astype('<f4')};return arrays,events,0
 if method=='independent':
  weights=[]
  for i,task in enumerate(sets):weights.append(np.linalg.lstsq(task[0][0],task[0][1],rcond=None)[0])
  arrays={'weights':np.asarray(weights,dtype='<f4')};return arrays,events,T*(64*D*D+D**3)
 for i,(task,_) in enumerate(zip(sets,targets)):
  xtr,ytr=task[0];xv,yv=task[1]
  if method.startswith('fixed_d'):
   d=int(method.split('_d')[1]);v=fit_direct(P,base,xtr,ytr,d);chunk=np.asarray(v,dtype=np.float16);dims[i]=d;fitops+=len(ytr)*d*d
  elif method=='adaptive_direct':
   found=False;d=K;v=None;val=float('inf');ops=0
   for cand in (2,4,8,16):
    z=fit_direct(P,base,xtr,ytr,cand);w=base+P[:,:cand]@z;score=nmse(xv,yv,w);ops+=len(ytr)*cand*cand
    if score<=threshold:d=cand;v=z;val=score;found=True;break
   if not found:v=fit_direct(P,base,xtr,ytr,K);val=nmse(xv,yv,base+P@v);ops+=len(ytr)*K*K
   chunk=np.asarray(v,dtype=np.float16);dims[i]=d;fitops+=ops
   events.append({'task_index':i,'group':groups[i],'choice':'intrinsic_coefficients','dimension':d,'validation_n_mse':val,'private_fallback':not found})
  elif method=='adaptive_mirror':
   found=False;d=K;angle=None;extra=None;val=float('inf');ops=0
   for cand in (2,4,8,16):
    a,r,o=fit_mirror(P,base,xtr,ytr,cand);z=np.empty(cand,np.float32);z[:2]=RADIUS*np.array([np.cos(float(a)),np.sin(float(a))]);z[2:]=r.astype(np.float32);score=nmse(xv,yv,base+P[:,:cand]@z);ops+=o
    if score<=threshold:d=cand;angle=a;extra=r;val=score;found=True;break
   if found:
    chunk=np.concatenate(([angle],extra)).astype(np.float16);dims[i]=d;modes[i]=0;choice='mirror_view_plus_residual'
   else:
    v=fit_direct(P,base,xtr,ytr,K);chunk=np.asarray(v,dtype=np.float16);dims[i]=K;modes[i]=1;val=nmse(xv,yv,base+P@v);choice='private_full_intrinsic_fallback';ops+=len(ytr)*K*K
   fitops+=ops;events.append({'task_index':i,'group':groups[i],'choice':choice,'dimension':int(dims[i]),'validation_n_mse':val,'private_fallback':not found})
  else:raise ValueError(method)
  chunks.append(chunk);offsets.append(offsets[-1]+len(chunk))
 maxd=int(dims.max());arrays={'base':base.astype('<f4'),'basis':P[:,:maxd].astype('<f4'),'dims':dims,'offsets':np.asarray(offsets,dtype='<u2'),'codes':np.concatenate(chunks).astype('<f2')}
 if method=='adaptive_mirror':arrays['modes']=modes;arrays['radius']=np.asarray([RADIUS],dtype='<f2')
 return arrays,events,fitops

def run(seed,split,threshold,outdir):
 P,base,z,targets,sets,groups=make_world(seed);summaries=[];all_events=[]
 for method in METHODS:
  start=time.perf_counter();arrays,events,fitops=encode_state(method,P,base,targets,sets,groups,threshold);path=outdir/f'{split}_{seed}_{threshold}_{method}.npz';nbytes,digest=pack_arrays(arrays,path);loaded=load_arrays(path)
  errs=[];group_err={}
  for i,task in enumerate(sets):
   xt,yt=task[2];w=decode(method,loaded,i);err=nmse(xt,yt,w);errs.append(err);group_err.setdefault(groups[i],[]).append(err)
  inference_start=time.perf_counter()
  for rep in range(3):
   for i,task in enumerate(sets):task[2][0]@decode(method,loaded,i)
  wall=time.perf_counter()-inference_start;throughput=3*T*128/max(wall,1e-12)
  counts={}
  for e in events:counts[e['choice']]=counts.get(e['choice'],0)+1
  summaries.append({'condition':f'{split}_t{threshold:g}','world_or_seed':seed,'method':method,'serialized_bytes':nbytes,'support_examples':T*64,'validation_examples':T*64,'test_examples':T*128,'optimizer_updates':0,'fit_compute_proxy':fitops,'active_ops_proxy_per_example':(D*D if method=='independent' else D if method=='tied' else D*(float(np.mean(arrays['dims']))+1)+(6 if method=='adaptive_mirror' else 0)),'wall_time_s':float(wall),'examples_per_s':throughput,'primary_metric':'mean_task_normalized_test_mse','primary_value':float(np.mean(errs)),'group_test_n_mse':{k:float(np.mean(v)) for k,v in group_err.items()},'payload_sha256':digest,'allocation_counts':counts,'max_payload_dimension':int(arrays.get('dims',np.zeros(1)).max())})
  for e in events:all_events.append({'condition':f'{split}_t{threshold:g}','world':seed,'method':method,**e,'final_payload_bytes':nbytes,'fit_compute_proxy':fitops})
 return summaries,all_events

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--split',choices=['development','fresh'],required=True);ap.add_argument('--threshold',type=float,required=True);ap.add_argument('--outdir',required=True);ap.add_argument('--json',required=True);ap.add_argument('--events',required=True);a=ap.parse_args();s,e=run(a.seed,a.split,a.threshold,Path(a.outdir));Path(a.json).write_text(json.dumps({'summaries':s,'events':e},indent=2)+'\n');print('\n'.join(f"{x['condition']} {x['method']} bytes={x['serialized_bytes']} nMSE={x['primary_value']:.5g} fitops={x['fit_compute_proxy']} alloc={x['allocation_counts']}" for x in s))
