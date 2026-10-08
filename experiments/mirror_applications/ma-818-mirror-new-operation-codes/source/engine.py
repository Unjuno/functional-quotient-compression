"""MA-818 operation-only code screen on frozen content vectors."""
from __future__ import annotations
import json,struct,time,hashlib
import numpy as np
D,PAIRS,OLD=8,((0,1),(2,3),(4,5),(6,7)),3
SUPPORTS=(4,8,16)
METHODS=('structured_mirror_angles','ordinary_structured_operation_embedding','ordinary_linearized_operation_code','independent_full_operator')

def givens(theta):
 m=np.eye(D)
 for i,t in enumerate(theta):
  c,s=np.cos(t),np.sin(t);m[2*i:2*i+2,2*i:2*i+2]=((c,-s),(s,c))
 return m

def generators():
 out=[]
 for i in range(len(PAIRS)):
  g=np.zeros((D,D));g[2*i,2*i+1]=-1;g[2*i+1,2*i]=1;out.append(g)
 return out

def world(seed,relation):
 rng=np.random.default_rng(seed);old=rng.uniform(-0.6,0.6,size=(OLD,len(PAIRS)))
 aligned=rng.uniform(-0.6,0.6,size=len(PAIRS));private=np.linalg.qr(rng.normal(size=(D,D)))[0]
 new=givens(aligned) if relation=='aligned' else private
 return old,aligned,new

def fit_angle(x,y):
 theta=[]
 for i in range(len(PAIRS)):
  a,b=x[:,2*i],x[:,2*i+1];c,d=y[:,2*i],y[:,2*i+1]
  cs=float(np.sum(a*c+b*d));sn=float(np.sum(a*d-b*c));theta.append(np.arctan2(sn,cs))
 return np.asarray(theta)

def fit_linearized(x,y):
 delta=y-x;phi=np.stack([(x@g.T).reshape(-1) for g in generators()],axis=1);return np.linalg.lstsq(phi,delta.reshape(-1),rcond=None)[0]

def fit_full(x,y):
 return np.linalg.lstsq(x,y,rcond=None)[0].T

def serialize(meta,arr):
 specs=[];chunks=[]
 for name in sorted(arr):
  a=np.ascontiguousarray(arr[name],dtype='<f4');raw=a.tobytes();specs.append({'name':name,'shape':list(a.shape),'nbytes':len(raw)});chunks.append(raw)
 h=json.dumps({'meta':meta,'arrays':specs},sort_keys=True,separators=(',',':')).encode();return b'MA818\x01'+struct.pack('<I',len(h))+h+b''.join(chunks)

def deserialize(blob):
 assert blob[:6]==b'MA818\x01';n=struct.unpack('<I',blob[6:10])[0];h=json.loads(blob[10:10+n]);pos=10+n;arr={}
 for s in h['arrays']:
  raw=blob[pos:pos+s['nbytes']];pos+=s['nbytes'];arr[s['name']]=np.frombuffer(raw,dtype='<f4').copy().reshape(s['shape'])
 assert pos==len(blob);return h['meta'],arr

def predicted_matrix(method,arr,op):
 if op<OLD:return givens(arr['old_angles'][op])
 if method in ('structured_mirror_angles','ordinary_structured_operation_embedding'):return givens(arr['new_angles'])
 if method=='ordinary_linearized_operation_code':return np.eye(D)+sum(float(a)*g for a,g in zip(arr['coeff'],generators()))
 return arr['new_matrix']

def run_sequence(method,arr,sequence,x0):
 x=np.asarray(x0,dtype=np.float64);ys=[]
 for op in sequence:x=np.tanh(predicted_matrix(method,arr,op)@x);ys.append(x.copy())
 return np.stack(ys)

def evaluate(seed,method,support,relation):
 old,angles,new=world(seed,relation);rng=np.random.default_rng(seed+support*17+(0 if relation=='aligned' else 99));xfit=rng.normal(size=(support,D));yfit=xfit@new.T
 t0=time.perf_counter()
 if method in ('structured_mirror_angles','ordinary_structured_operation_embedding'):params={'new_angles':fit_angle(xfit,yfit)}
 elif method=='ordinary_linearized_operation_code':params={'coeff':fit_linearized(xfit,yfit)}
 else:params={'new_matrix':fit_full(xfit,yfit)}
 fit_wall=time.perf_counter()-t0
 arr={'old_angles':old.astype('<f4'),**{k:np.asarray(v,dtype='<f4') for k,v in params.items()}}
 serialized_method='structured_givens_operation' if method in ('structured_mirror_angles','ordinary_structured_operation_embedding') else method
 meta={'method':serialized_method,'support':support,'relation':relation,'dim':D,'sequence':[0,3,1,3]};blob=serialize(meta,arr);meta,arr=deserialize(blob)
 xtest=np.random.default_rng(seed+1234).normal(size=(64,D));seq=[0,3,1,3];errs=[];norms=[];old_err=[];t0=time.perf_counter()
 for x0 in xtest:
  true=[];x=x0.copy()
  for op in seq:
   mat=givens(old[op]) if op<OLD else new;x=np.tanh(mat@x);true.append(x.copy())
  true=np.stack(true);pred=run_sequence(method,arr,seq,x0);errs.append(np.mean((pred-true)**2));norms.append(np.mean(true**2))
  # retention: existing operations run without the new operation
  x=x0.copy();yt=[]
  for op in (0,1,2):x=np.tanh(givens(old[op])@x);yt.append(x.copy())
  old_err.append(np.mean((np.stack(yt)-run_sequence(method,arr,(0,1,2),x0))**2))
 wall=time.perf_counter()-t0
 qmac={'structured_mirror_angles':0,'ordinary_structured_operation_embedding':0,'ordinary_linearized_operation_code':D*len(PAIRS),'independent_full_operator':D*D}[method]*len(seq)*len(xtest)
 givens_mults=2*D*len(seq)*len(xtest) if method in ('structured_mirror_angles','ordinary_structured_operation_embedding') else 0
 trig_evals=2*len(PAIRS)*len(seq)*len(xtest) if method in ('structured_mirror_angles','ordinary_structured_operation_embedding') else 0
 return {'seed':seed,'method':method,'support':support,'relation':relation,'serialized_bytes':len(blob),'normalized_sequence_mse':float(np.mean(errs)/max(np.mean(norms),1e-30)),'old_operation_retention_mse':float(np.mean(old_err)),'support_fit_flops_proxy':int(support*D*len(PAIRS) if method!='independent_full_operator' else support*D*D),'query_macs_proxy':int(qmac),'givens_scalar_multiplications':int(givens_mults),'trig_evaluations':int(trig_evals),'fit_wall_s':fit_wall,'query_wall_s':wall,'payload_sha256':hashlib.sha256(blob).hexdigest()}

if __name__=='__main__':
 import csv,pathlib,sys
 root=pathlib.Path(__file__).resolve().parents[1];mode=sys.argv[1] if len(sys.argv)>1 else 'dev';seeds=(81801,81802) if mode=='dev' else (81811,81812,81813)
 rows=[evaluate(seed,m,sup,rel) for seed in seeds for sup in SUPPORTS for rel in ('aligned','unrelated') for m in METHODS]
 p=root/('source/development.csv' if mode=='dev' else 'source/fresh.csv')
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 for r in rows:print(r)
