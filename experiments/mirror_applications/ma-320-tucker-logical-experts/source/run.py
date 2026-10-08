import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
D,O,K,T=32,16,16,64;N_ALIGNED,N_UNRELATED=48,16;H=8;GRID=1024;THRESHOLD=1e-4
METHODS=['tied','tucker1','tucker2','tucker4','tucker8','tucker16','tucker16_private','mirror','independent']
AMPLITUDES=np.array([.25/np.sqrt(h) for h in range(1,H+1)],dtype=np.float32)

def world(seed):
 rng=np.random.default_rng(seed);P=D*O
 q,_=np.linalg.qr(rng.normal(size=(P,K)));basis=q.T.astype(np.float32).reshape(K,O,D)
 base=rng.normal(0,.05,size=(O,D)).astype(np.float32);phases=rng.uniform(0,2*np.pi,N_ALIGNED).astype(np.float32);weights=[];groups=[]
 for i in range(N_ALIGNED):
  c=np.array([v for h,a in enumerate(AMPLITUDES,1) for v in (a*np.cos(h*phases[i]),a*np.sin(h*phases[i]))],dtype=np.float32)
  weights.append(base+np.einsum('k,kod->od',c,basis));groups.append('aligned_orbit')
 # Construct orthonormal off-bank directions for unrelated experts.
 raw=rng.normal(size=(P,N_UNRELATED));raw-=q@(q.T@raw);u,_=np.linalg.qr(raw)
 for i in range(N_UNRELATED):weights.append(base+.28*u[:,i].reshape(O,D).astype(np.float32));groups.append('unrelated')
 weights=np.stack(weights);sets=[]
 for i,w in enumerate(weights):
  task=[]
  for n,salt in [(96,1100),(64,2200),(128,3300)]:
   x=np.random.default_rng(seed+salt+i*37).normal(size=(n,D)).astype(np.float32);y=x@w.T;task.append((x,y))
  sets.append(task)
 return basis,base,weights,sets,np.asarray(groups)

def nrmse(x,y,w):
 err=x@w.T-y
 return float(np.mean(err*err)/(np.mean(y*y)+1e-12))

def features(x,basis):return np.einsum('nd,kod->nok',x,basis,optimize=True)

def fit_coeff(x,y,base,basis,r):
 f=features(x,basis[:r]);res=y-x@base.T
 return np.linalg.lstsq(f.reshape(-1,r),res.reshape(-1),rcond=None)[0].astype(np.float32)

def mirror_angle(x,y,base,basis):
 f=features(x,basis);res=(y-x@base.T).reshape(-1);grid=np.linspace(0,2*np.pi,GRID,endpoint=False,dtype=np.float32)
 coeff=np.empty((GRID,K),dtype=np.float32)
 for h,a in enumerate(AMPLITUDES,1):coeff[:,2*h-2]=a*np.cos(h*grid);coeff[:,2*h-1]=a*np.sin(h*grid)
 pred=np.einsum('nok,gk->nog',f,coeff,optimize=True);err=np.mean((pred-res.reshape(len(x),O,1))**2,axis=(0,1));j=int(np.argmin(err));return float(grid[j]),coeff[j],GRID*len(x)*O*K

def pack(arrays,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
  for name in sorted(arrays):
   b=io.BytesIO();np.lib.format.write_array(b,np.ascontiguousarray(arrays[name]),allow_pickle=False);info=zipfile.ZipInfo(name+'.npy',date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED;info.external_attr=0o600<<16;z.writestr(info,b.getvalue())
 raw=path.read_bytes();return len(raw),hashlib.sha256(raw).hexdigest()

def unpack(path):
 with zipfile.ZipFile(path) as z:return {n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in sorted(z.namelist())}

def encode(method,basis,base,weights,sets,groups):
 start=time.perf_counter();ids=np.arange(T,dtype=np.uint8);events=[];fitops=0
 if method=='independent': arr={'weights':weights.astype('<f2'),'task_ids':ids};return arr,events,T*96*D*O, time.perf_counter()-start
 if method=='tied':arr={'base':base.astype('<f2'),'task_ids':ids};return arr,events,0,time.perf_counter()-start
 rank={'tucker1':1,'tucker2':2,'tucker4':4,'tucker8':8,'tucker16':16,'tucker16_private':16}.get(method)
 if rank:
  codes=[];private_ids=[];private_weights=[];rows=[];vals=[]
  for i,task in enumerate(sets):
   x,y=task[0];c=fit_coeff(x,y,base,basis,rank);fitops+=96*O*D*rank
   w=base+np.einsum('k,kod->od',c,basis[:rank]);score=nrmse(x[1],y[1],w)
   if method=='tucker16_private' and score>THRESHOLD:
    private_ids.append(i);private_weights.append(weights[i].astype('<f2'));events.append({'expert_id':i,'group':str(groups[i]),'path':'private_full','validation_n_mse':score})
   else:
    rows.append(i);codes.append(c.astype('<f2'));vals.append(score);events.append({'expert_id':i,'group':str(groups[i]),'path':'tucker_coefficients','validation_n_mse':score})
  arr={'base':base.astype('<f2'),'basis':basis[:rank].astype('<f2'),'task_ids':ids,'code_ids':np.asarray(rows,dtype=np.uint8),'codes':np.asarray(codes,dtype='<f2').reshape(-1,rank),'private_ids':np.asarray(private_ids,dtype=np.uint8),'private_weights':np.asarray(private_weights,dtype='<f2').reshape(-1,O,D)}
  return arr,events,fitops,time.perf_counter()-start
 if method=='mirror':
  angles=[];angle_ids=[];private_ids=[];private_weights=[]
  for i,task in enumerate(sets):
   x,y=task[0];a,c,ops=mirror_angle(x,y,base,basis);fitops+=ops
   xval,yval=task[1];w=base+np.einsum('k,kod->od',c,basis);score=nrmse(xval,yval,w)
   if score>THRESHOLD:
    private_ids.append(i);private_weights.append(weights[i].astype('<f2'));events.append({'expert_id':i,'group':str(groups[i]),'path':'private_full','validation_n_mse':score})
   else:
    angle_ids.append(i);angles.append(a);events.append({'expert_id':i,'group':str(groups[i]),'path':'shared_phase','validation_n_mse':score})
  arr={'base':base.astype('<f2'),'basis':basis.astype('<f2'),'amplitudes':AMPLITUDES.astype('<f2'),'task_ids':ids,'angle_ids':np.asarray(angle_ids,dtype=np.uint8),'angles':np.asarray(angles,dtype='<f2'),'private_ids':np.asarray(private_ids,dtype=np.uint8),'private_weights':np.asarray(private_weights,dtype='<f2').reshape(-1,O,D)}
  return arr,events,fitops,time.perf_counter()-start
 raise ValueError(method)

def decode(method,s,i):
 if method=='independent':return s['weights'][i].astype(np.float32)
 if method=='tied':return s['base'].astype(np.float32)
 if i in s['private_ids'].astype(int):return s['private_weights'][int(np.where(s['private_ids']==i)[0][0])].astype(np.float32)
 base=s['base'].astype(np.float32);basis=s['basis'].astype(np.float32)
 if method=='mirror':
  j=int(np.where(s['angle_ids']==i)[0][0]);a=float(s['angles'][j]);c=np.empty(K,np.float32)
  for h,v in enumerate(s['amplitudes'].astype(np.float32),1):c[2*h-2]=v*np.cos(h*a);c[2*h-1]=v*np.sin(h*a)
 else:
  j=int(np.where(s['code_ids']==i)[0][0]);c=s['codes'][j].astype(np.float32)
 return base+np.einsum('k,kod->od',c,basis,optimize=True)

def run(seed,split,outdir,jsonpath):
 basis,base,weights,sets,groups=world(seed);rows=[];events_all=[];outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
 for method in METHODS:
  arrays,events,fitops,wall=encode(method,basis,base,weights,sets,groups);path=outdir/f'{split}_{seed}_{method}.npz';n,digest=pack(arrays,path);s=unpack(path);per=[];group_err={};t0=time.perf_counter();active=0
  for i,task in enumerate(sets):
   x,y=task[2];w=decode(method,s,i);score=nrmse(x,y,w);per.append(score);group_err.setdefault(str(groups[i]),[]).append(score);active+=D*O+(0 if method=='independent' else s.get('basis',np.empty((0,))).shape[0]*O*D)
  infer_wall=time.perf_counter()-t0
  rows.append({'condition':split,'seed':seed,'method':method,'serialized_bytes':n,'payload_sha256':digest,'support_examples':T*96,'validation_examples':T*64,'test_examples':T*128,'optimizer_updates':0,'fit_compute_proxy':fitops,'active_ops_proxy_per_request':active/T,'private_experts':len(arrays.get('private_ids',[])),'shared_phase_experts':len(arrays.get('angle_ids',[])),'mean_test_n_mse':float(np.mean(per)),'max_test_n_mse':float(np.max(per)),'group_test_n_mse':{k:float(np.mean(v)) for k,v in group_err.items()},'fit_wall_s':wall,'decode_and_infer_wall_s':infer_wall,'requests_per_s':T/infer_wall,'payload_format':'deterministic fixed-timestamp ZIP/NPY'})
  events_all.extend({'condition':split,'seed':seed,'method':method,**e,'serialized_bytes':n,'payload_sha256':digest} for e in events)
 result={'seed':seed,'condition':split,'basis_seed':seed+300,'summary':rows,'allocation_events':events_all};Path(jsonpath).parent.mkdir(parents=True,exist_ok=True);Path(jsonpath).write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--split',choices=['development','fresh'],required=True);ap.add_argument('--outdir',required=True);ap.add_argument('--json',required=True);a=ap.parse_args();run(a.seed,a.split,a.outdir,a.json)
