import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
D=64;DIM=8;R=4;TASK_ALIGNED=128;TASK_MIXED=128;RHO=.25;GRID=1024;THRESHOLD=1e-4
METHODS=['hard_tied','loretta_independent','shared_pair_private','mirror_phase_private','independent_full']

def tt_matrix(g1,g2,g3,g4):
 return np.einsum('ia,ajb,bkc,cl->ijkl',g1,g2,g3,g4,optimize=True).reshape(D,D).astype(np.float32)

def random_cores(rng):
 g1=rng.normal(0,.45,(8,R)).astype(np.float32);g2=rng.normal(0,.45,(R,8,R)).astype(np.float32);g3=rng.normal(0,.45,(R,8,R)).astype(np.float32);g4=rng.normal(0,.45,(R,8)).astype(np.float32)
 return [g1,g2,g3,g4]

def world(seed,bank):
 rng=np.random.default_rng(seed);shared=random_cores(rng);g1,g2,g3,g4=shared;u=rng.normal(0,.3,g2.shape).astype(np.float32);v=rng.normal(0,.3,g2.shape).astype(np.float32);angles=rng.uniform(0,2*np.pi,TASK_ALIGNED if bank=='aligned' else 96).astype(np.float32)
 n_aligned=TASK_ALIGNED if bank=='aligned' else 96;n_tasks=TASK_ALIGNED if bank=='aligned' else TASK_MIXED
 all_cores=[[] for _ in range(4)];weights=[];groups=[]
 for i in range(n_tasks):
  if i<n_aligned:
   a=angles[i];taskcores=[g1,g2+RHO*(np.cos(a)*u+np.sin(a)*v),g3,g4];groups.append('aligned_orbit')
  else:taskcores=random_cores(rng);groups.append('unrelated')
  for j,c in enumerate(taskcores):all_cores[j].append(c.copy())
  weights.append(tt_matrix(*taskcores))
 weights=np.stack(weights);all_cores=[np.stack(x) for x in all_cores];basew=tt_matrix(*shared);uw=tt_matrix(g1,u,g3,g4);vw=tt_matrix(g1,v,g3,g4);sets=[]
 for i,w in enumerate(weights):
  task=[]
  for n,salt in [(64,1100),(32,2200),(64,3300)]:
   x=np.random.default_rng(seed+salt+i*29).normal(size=(n,D)).astype(np.float32);y=x@w.T;task.append((x,y))
  sets.append(task)
 return {'shared':shared,'u':u,'v':v,'basew':basew,'uw':uw,'vw':vw,'weights':weights,'cores':all_cores,'sets':sets,'groups':np.asarray(groups),'n_aligned':n_aligned,'bank':bank}

def nrmse(x,y,w):return float(np.mean((x@w.T-y)**2)/(np.mean(y*y)+1e-12))

def fit_pair(x,y,basew,uw,vw):
 f0=x@uw.T;f1=x@vw.T;res=(y-x@basew.T).reshape(-1);a=np.stack([f0.reshape(-1),f1.reshape(-1)],axis=1);return np.linalg.lstsq(a,res,rcond=None)[0].astype(np.float32)

def fit_phase(x,y,basew,uw,vw):
 g=np.linspace(0,2*np.pi,GRID,endpoint=False,dtype=np.float32);bp=x@basew.T;fu=x@uw.T;fv=x@vw.T;pred=bp[:,:,None]+RHO*(fu[:,:,None]*np.cos(g)[None,None,:]+fv[:,:,None]*np.sin(g)[None,None,:]);err=np.mean((pred-y[:,:,None])**2,axis=(0,1));idx=int(np.argmin(err));return float(g[idx]),GRID*len(x)*D*3

def pack(arrays,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
  for name in sorted(arrays):
   b=io.BytesIO();np.lib.format.write_array(b,np.ascontiguousarray(arrays[name]),allow_pickle=False);info=zipfile.ZipInfo(name+'.npy',date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED;info.external_attr=0o600<<16;z.writestr(info,b.getvalue())
 raw=path.read_bytes();return len(raw),hashlib.sha256(raw).hexdigest()

def unpack(path):
 with zipfile.ZipFile(path) as z:return {n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in sorted(z.namelist())}

def encode(method,w):
 start=time.perf_counter();t=w['weights'];sets=w['sets'];groups=w['groups'];ids=np.arange(len(t),dtype=np.uint8);events=[];fitops=0
 if method=='hard_tied':return {'g1':w['shared'][0].astype('<f2'),'g2':w['shared'][1].astype('<f2'),'g3':w['shared'][2].astype('<f2'),'g4':w['shared'][3].astype('<f2'),'task_ids':ids},events,0,time.perf_counter()-start
 if method=='loretta_independent':
  a={f'g{i+1}':w['cores'][i].astype('<f2') for i in range(4)};a['task_ids']=ids;return a,events,len(t)*64*D*D,time.perf_counter()-start
 if method=='independent_full':return {'weights':t.astype('<f2'),'task_ids':ids},events,len(t)*64*D*D,time.perf_counter()-start
 codes=[];code_ids=[];angles=[];angle_ids=[];private_ids=[];private=[]
 for i,task in enumerate(sets):
  x,y=task[0];xv,yv=task[1]
  if method=='shared_pair_private':
   c=fit_pair(x,y,w['basew'],w['uw'],w['vw']);fitops+=64*D*2;pred=w['basew']+c[0]*w['uw']+c[1]*w['vw'];score=nrmse(xv,yv,pred)
   if score>THRESHOLD:private_ids.append(i);private.append([w['cores'][k][i] for k in range(4)]);events.append({'task_id':i,'group':str(groups[i]),'path':'private_loretta_tt','validation_n_mse':score})
   else:code_ids.append(i);codes.append(c);events.append({'task_id':i,'group':str(groups[i]),'path':'free_core_pair','validation_n_mse':score})
  else:
   a,ops=fit_phase(x,y,w['basew'],w['uw'],w['vw']);fitops+=ops;xv,yv=task[1];pred=w['basew']+RHO*(np.cos(a)*w['uw']+np.sin(a)*w['vw']);score=nrmse(xv,yv,pred)
   if score>THRESHOLD:private_ids.append(i);private.append([w['cores'][k][i] for k in range(4)]);events.append({'task_id':i,'group':str(groups[i]),'path':'private_loretta_tt','validation_n_mse':score})
   else:angle_ids.append(i);angles.append(a);events.append({'task_id':i,'group':str(groups[i]),'path':'mirror_phase','validation_n_mse':score})
 arr={'g1':w['shared'][0].astype('<f2'),'g2':w['shared'][1].astype('<f2'),'g3':w['shared'][2].astype('<f2'),'g4':w['shared'][3].astype('<f2'),'u':w['u'].astype('<f2'),'v':w['v'].astype('<f2'),'task_ids':ids,'private_ids':np.asarray(private_ids,dtype=np.uint8)}
 for j in range(4):arr[f'private_g{j+1}']=np.asarray([p[j] for p in private],dtype='<f2')
 if method=='shared_pair_private':arr['code_ids']=np.asarray(code_ids,dtype=np.uint8);arr['codes']=np.asarray(codes,dtype='<f2').reshape(-1,2)
 else:arr['angle_ids']=np.asarray(angle_ids,dtype=np.uint8);arr['angles']=np.asarray(angles,dtype='<f2');arr['radius']=np.asarray([RHO],dtype='<f2')
 return arr,events,fitops,time.perf_counter()-start

def decode(method,s,i):
 if method=='independent_full':return s['weights'][i].astype(np.float32)
 if method=='loretta_independent':return tt_matrix(*(s[f'g{k}'][i].astype(np.float32) for k in range(1,5)))
 if method=='hard_tied':return tt_matrix(*(s[f'g{k}'].astype(np.float32) for k in range(1,5)))
 if i in s['private_ids'].astype(int):
  ix=int(np.where(s['private_ids']==i)[0][0]);return tt_matrix(*(s[f'private_g{k}'][ix].astype(np.float32) for k in range(1,5)))
 shared=[s[f'g{k}'].astype(np.float32) for k in range(1,5)];u=s['u'].astype(np.float32);v=s['v'].astype(np.float32)
 base=tt_matrix(*shared);wu=tt_matrix(shared[0],u,shared[2],shared[3]);wv=tt_matrix(shared[0],v,shared[2],shared[3])
 if method=='shared_pair_private':j=int(np.where(s['code_ids']==i)[0][0]);a,b=s['codes'][j].astype(np.float32);return base+a*wu+b*wv
 j=int(np.where(s['angle_ids']==i)[0][0]);a=float(s['angles'][j]);rho=float(s['radius'][0]);return base+rho*(np.cos(a)*wu+np.sin(a)*wv)

def run(seed,split,outdir,jsonpath):
 summaries=[];events_all=[];outdir=Path(outdir)
 for bank in ['aligned','mixed']:
  w=world(seed,bank)
  for method in METHODS:
   arr,events,fitops,fitwall=encode(method,w);path=outdir/f'{split}_{seed}_{bank}_{method}.npz';n,digest=pack(arr,path);s=unpack(path);per=[];by={};active=0;t0=time.perf_counter()
   for i,task in enumerate(w['sets']):
    x,y=task[2];pred=decode(method,s,i);score=nrmse(x,y,pred);per.append(score);by.setdefault(str(w['groups'][i]),[]).append(score);active+=262144+4096 if method!='independent_full' else 4096
   inferwall=time.perf_counter()-t0
   summaries.append({'condition':split,'bank':bank,'seed':seed,'method':method,'serialized_bytes':n,'payload_sha256':digest,'support_examples':len(w['sets'])*64,'validation_examples':len(w['sets'])*32,'test_examples':len(w['sets'])*64,'optimizer_updates':0,'fit_compute_proxy':fitops,'active_ops_proxy_per_request':active/len(w['sets']),'private_tasks':len(arr.get('private_ids',[])),'view_tasks':len(arr.get('angle_ids',[])),'mean_test_n_mse':float(np.mean(per)),'max_test_n_mse':float(np.max(per)),'group_test_n_mse':{k:float(np.mean(v)) for k,v in by.items()},'fit_wall_s':fitwall,'decode_infer_wall_s':inferwall,'requests_per_s':len(w['sets'])/inferwall})
   events_all.extend({'condition':split,'bank':bank,'seed':seed,'method':method,**e,'serialized_bytes':n,'payload_sha256':digest} for e in events)
 result={'seed':seed,'condition':split,'summaries':summaries,'allocation_events':events_all};p=Path(jsonpath);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--split',choices=['development','fresh'],required=True);ap.add_argument('--outdir',required=True);ap.add_argument('--json',required=True);a=ap.parse_args();run(a.seed,a.split,a.outdir,a.json)
