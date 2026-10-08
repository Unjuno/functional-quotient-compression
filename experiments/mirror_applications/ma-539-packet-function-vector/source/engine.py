"""MA-539 synthetic packet function-vector vs generic packet latent screen."""
from __future__ import annotations
import json,struct,time
import numpy as np
T,P,D,QTRUE=8,4,8,2
ALIGNED=6
QS=(1,2,4)
RIDGES=(0.0,1e-4,1e-2)
METHODS=('hard_shared','ptp_shared_packet_latent','mirror_function_vector','independent_task_slot_upper')

def world(seed):
 rng=np.random.default_rng(seed)
 w0=rng.normal(size=(P,D)).astype(np.float64)/np.sqrt(D)
 basis=rng.normal(size=(4,P,D)).astype(np.float64)/np.sqrt(D)
 z=rng.normal(size=(ALIGNED,QTRUE)).astype(np.float64)
 private=rng.normal(size=(T-ALIGNED,P,D)).astype(np.float64)/np.sqrt(D)
 return w0,basis,z,private

def data(seed,task,n):
 rng=np.random.default_rng(seed*100+task)
 return rng.normal(size=(n,D)).astype(np.float64)

def target(x,task,w0,basis,z,private):
 if task<ALIGNED: w=w0+np.einsum('q,qpd->pd',z[task],basis[:QTRUE])
 else: w=private[task-ALIGNED]
 return x@w.T

def fit_code(x,y,w0,basis,q,ridge):
 # One task code explains all P packet slots using the same latent coordinate.
 residual=(y-x@w0.T).reshape(-1)
 phi=np.stack([(x@basis[k].T).reshape(-1) for k in range(q)],axis=1)
 return np.linalg.solve(phi.T@phi+ridge*np.eye(q),phi.T@residual)

def fit_private(x,y,ridge):
 # Slotwise ordinary least-squares private function.
 return np.stack([np.linalg.solve(x.T@x+ridge*np.eye(D),x.T@y[:,p]) for p in range(P)])

def build(method,q,ridge,seed):
 w0,basis,z,private=world(seed)
 codes=[]; priv=[]
 for task in range(T):
  x=data(seed+11,task,64); y=target(x,task,w0,basis,z,private)
  if task<ALIGNED and method in ('ptp_shared_packet_latent','mirror_function_vector'):
   codes.append(fit_code(x,y,w0,basis[:q],q,ridge))
  elif task>=ALIGNED and method in ('ptp_shared_packet_latent','mirror_function_vector','independent_task_slot_upper'):
   priv.append((task,fit_private(x,y,ridge)))
  elif method=='independent_task_slot_upper': priv.append((task,fit_private(x,y,ridge)))
 arr={}
 meta={'method':method,'q':q,'tasks':T,'slots':P,'input_dim':D,'aligned_tasks':ALIGNED}
 if method=='hard_shared': arr={'w0':w0.astype('<f4')}
 elif method in ('ptp_shared_packet_latent','mirror_function_vector'):
  arr={'w0':w0.astype('<f4'),'basis':basis[:q].astype('<f4'),'codes':np.stack(codes).astype('<f4')}
  for task,w in priv: arr[f'private_{task:02}']=w.astype('<f4')
  if method=='mirror_function_vector': arr['fv_to_decoder']=np.eye(q,dtype='<f4')
 elif method=='independent_task_slot_upper':
  for task in range(T):
   x=data(seed+11,task,64); y=target(x,task,w0,basis,z,private)
   arr[f'weights_{task:02}']=fit_private(x,y,ridge).astype('<f4')
 else: raise ValueError(method)
 return meta,arr

def serialize(meta,arrays):
 spec=[]; chunks=[]
 for n in sorted(arrays):
  a=np.ascontiguousarray(arrays[n],dtype='<f4');b=a.tobytes();spec.append({'name':n,'shape':list(a.shape),'nbytes':len(b)});chunks.append(b)
 h=json.dumps({'meta':meta,'arrays':spec},sort_keys=True,separators=(',',':')).encode()
 return b'MA539\x01'+struct.pack('<I',len(h))+h+b''.join(chunks)

def deserialize(blob):
 assert blob[:6]==b'MA539\x01'; n=struct.unpack('<I',blob[6:10])[0];h=json.loads(blob[10:10+n]);pos=10+n;arr={}
 for x in h['arrays']:
  b=blob[pos:pos+x['nbytes']];pos+=x['nbytes'];arr[x['name']]=np.frombuffer(b,dtype='<f4').copy().reshape(x['shape'])
 assert pos==len(blob)
 return h['meta'],arr

def predict(method,meta,a,x,task):
 if method=='hard_shared': return x@a['w0'].T
 if method=='independent_task_slot_upper': return x@a[f'weights_{task:02}'].T
 if task>=ALIGNED: return x@a[f'private_{task:02}'].T
 code=a['codes'][task]
 if method=='mirror_function_vector': code=a['fv_to_decoder']@code
 return _predict_shared(x,a['w0'],a['basis'],code)

def _predict_shared(x,w0,basis,code):
 out=x@w0.T
 for k in range(len(code)): out+=code[k]*(x@basis[k].T)
 return out

def metrics(seed,method,q,ridge):
 w0,basis,z,private=world(seed);meta,arr=build(method,q,ridge,seed);blob=serialize(meta,arr)
 mse=[]; exact=[]; nfit=0
 t0=time.perf_counter();meta,arr=deserialize(blob)
 for task in range(T):
  x=data(seed+97,task,256); y=target(x,task,w0,basis,z,private); yh=predict(method,meta,arr,x,task)
  mse.extend(np.mean((yh-y)**2,axis=1).tolist())
  exact.extend(np.all((yh>=0)==(y>=0),axis=1).tolist())
 wall=time.perf_counter()-t0
 infer= T*256*P*D*(1+(q if method in ('ptp_shared_packet_latent','mirror_function_vector') else 0))
 if method=='mirror_function_vector': infer+=ALIGNED*q*q*256*P
 fit_flops=T*64*P*q*q if method in ('ptp_shared_packet_latent','mirror_function_vector') else T*64*D*D
 return {'seed':seed,'method':method,'q':q,'ridge':ridge,'serialized_bytes':len(blob),'normalized_packet_mse':float(np.mean(mse)/max(np.mean([np.mean(target(data(seed+97,t,256),t,w0,basis,z,private)**2) for t in range(T)]),1e-30)),'exact_sign_packet_accuracy':float(np.mean(exact)),'support_examples':T*64,'fit_flops_proxy':int(fit_flops),'inference_macs_proxy':int(infer),'wall_clock_test_s':wall,'payload_hash':__import__('hashlib').sha256(blob).hexdigest()}

if __name__=='__main__':
 import csv,pathlib,sys
 root=pathlib.Path(__file__).resolve().parents[1]; mode=sys.argv[1] if len(sys.argv)>1 else 'dev'
 if mode=='dev':
  rows=[]
  for seed in (53901,53902):
   for q in QS:
    for ridge in RIDGES:
     for method in METHODS: rows.append(metrics(seed,method,q,ridge))
  with (root/'source/development.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
  by={}
  for q in QS:
   rr=[r for r in rows if r['method']=='mirror_function_vector' and r['q']==q]
   by[str(q)]=min(sum(r['normalized_packet_mse'] for r in rr if r['ridge']==ridge)/len([r for r in rr if r['ridge']==ridge]) for ridge in RIDGES)
  qualifies=[q for q in QS if by[str(q)]<=1e-6]; q=min(qualifies) if qualifies else min(QS,key=lambda x:by[str(x)])
  rr=[r for r in rows if r['method']=='mirror_function_vector' and r['q']==q]
  ridge=min(RIDGES,key=lambda l:sum(r['normalized_packet_mse'] for r in rr if r['ridge']==l))
  cfg={'selected_q':q,'selected_ridge':ridge,'dev_rank_mse':by,'fresh_seeds':[53911,53912,53913]}
  (root/'source/frozen_config.json').write_text(json.dumps(cfg,indent=2)+'\n');print(cfg)
 elif mode=='fresh':
  cfg=json.load(open(root/'source/frozen_config.json'));rows=[]
  for seed in cfg['fresh_seeds']:
   for method in METHODS: rows.append(metrics(seed,method,cfg['selected_q'],cfg['selected_ridge']))
  with (root/'source/fresh.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
  for r in rows: print(r)
 else: raise SystemExit('usage: engine.py dev|fresh')
