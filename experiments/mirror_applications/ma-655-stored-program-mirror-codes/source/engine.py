"""MA-655 program-memory interpolation mechanism screen (NumPy only)."""
from __future__ import annotations
import json,struct,time,hashlib
import numpy as np
D,TASKS,ALIGNED,STEPS=12,24,23,8
METHODS=('nspm_corner_memory','mirror_shared_program_code','ordinary_shared_matrix_basis','independent_full_programs')
CORNERS=((-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.))

def world(seed):
 rng=np.random.default_rng(seed);a0=rng.normal(size=(D,D))/np.sqrt(D);b1=rng.normal(size=(D,D))/np.sqrt(D);b2=rng.normal(size=(D,D))/np.sqrt(D)
 codes=rng.uniform(-1,1,size=(ALIGNED,2));private=rng.normal(size=(D,D))/np.sqrt(D)
 return a0,b1,b2,codes,private

def affine(a0,b1,b2,z):return a0+z[0]*b1+z[1]*b2

def bilinear_weights(z):
 x,y=z;return np.asarray([(1-x)*(1-y),(1-x)*(1+y),(1+x)*(1-y),(1+x)*(1+y)],dtype=np.float64)/4

def serialize(meta,arrays):
 spec=[];chunks=[]
 for name in sorted(arrays):
  a=np.ascontiguousarray(arrays[name],dtype='<f4');raw=a.tobytes();spec.append({'name':name,'shape':list(a.shape),'nbytes':len(raw)});chunks.append(raw)
 h=json.dumps({'meta':meta,'arrays':spec},sort_keys=True,separators=(',',':')).encode()
 return b'MA655\x01'+struct.pack('<I',len(h))+h+b''.join(chunks)

def deserialize(blob):
 assert blob[:6]==b'MA655\x01';n=struct.unpack('<I',blob[6:10])[0];h=json.loads(blob[10:10+n]);pos=10+n;arr={}
 for spec in h['arrays']:
  raw=blob[pos:pos+spec['nbytes']];pos+=spec['nbytes'];arr[spec['name']]=np.frombuffer(raw,dtype='<f4').copy().reshape(spec['shape'])
 assert pos==len(blob)
 return h['meta'],arr

def state(method,seed):
 a0,b1,b2,codes,private=world(seed);meta={'method':method,'hidden':D,'aligned_tasks':ALIGNED,'steps':STEPS};arr={'task_addresses':np.vstack([codes,np.zeros((1,2))]).astype('<f4'),'private':private.astype('<f4')}
 if method=='nspm_corner_memory':arr['anchors']=np.stack([affine(a0,b1,b2,np.asarray(z)) for z in CORNERS]).astype('<f4')
 elif method in ('mirror_shared_program_code','ordinary_shared_matrix_basis'):
  arr.update({'a0':a0.astype('<f4'),'b1':b1.astype('<f4'),'b2':b2.astype('<f4')})
 elif method=='independent_full_programs':arr['programs']=np.stack([affine(a0,b1,b2,z) for z in codes]).astype('<f4')
 else:raise ValueError(method)
 return meta,arr

def controller(method,meta,arr,z,task_index):
 if task_index>=ALIGNED:return arr['private']
 if method=='nspm_corner_memory':return np.einsum('i,ijk->jk',bilinear_weights(z),arr['anchors'])
 if method in ('mirror_shared_program_code','ordinary_shared_matrix_basis'):return arr['a0']+z[0]*arr['b1']+z[1]*arr['b2']
 nearest=int(np.argmin(np.sum((arr['task_addresses'][:ALIGNED]-z)**2,axis=1)));return arr['programs'][nearest]

def trajectory(a,x0,steps=STEPS):
 x=np.asarray(x0,dtype=np.float64);ys=[]
 for _ in range(steps):x=np.tanh(a@x);ys.append(x.copy())
 return np.stack(ys)

def eval_one(seed,method,nheld=32):
 a0,b1,b2,codes,private=world(seed);meta,arr=state(method,seed);blob=serialize(meta,arr);meta,arr=deserialize(blob)
 rng=np.random.default_rng(seed+901);held=rng.uniform(-1,1,size=(nheld,2));all_z=list(held)+[np.zeros(2)];task_ids=[i%ALIGNED for i in range(nheld)]+[TASKS-1];inputs=rng.normal(size=(len(all_z),D));errs=[];macs=0;switch_ns=[];t_all=time.perf_counter()
 for z,task_id,x0 in zip(all_z,task_ids,inputs):
  target=private if task_id==TASKS-1 else affine(a0,b1,b2,z)
  ref=trajectory(target,x0)
  t0=time.perf_counter();a=controller(method,meta,arr,z,task_id);pred=trajectory(a,x0);switch_ns.append(time.perf_counter()-t0)
  errs.append(float(np.mean((pred-ref)**2)))
  if task_id==TASKS-1:macs+=STEPS*D*D
  elif method=='nspm_corner_memory':macs+=STEPS*4*D*D+STEPS*8
  elif method in ('mirror_shared_program_code','ordinary_shared_matrix_basis'):macs+=STEPS*3*D*D+STEPS*2*D
  else:macs+=STEPS*D*D
 wall=time.perf_counter()-t_all
 target_energy=np.mean([np.mean(trajectory(private,x)**2) if i==len(all_z)-1 else np.mean(trajectory(affine(a0,b1,b2,z),x)**2) for i,(z,x) in enumerate(zip(all_z,inputs))])
 return {'seed':seed,'method':method,'serialized_bytes':len(blob),'normalized_trajectory_mse':float(np.mean(errs)/max(target_energy,1e-30)),'controller_macs_proxy':int(macs),'mean_switch_wall_s':float(np.mean(switch_ns)),'total_eval_wall_s':wall,'heldout_interpolations':nheld,'private_checks':1,'payload_sha256':hashlib.sha256(blob).hexdigest()}

if __name__=='__main__':
 import csv,pathlib,sys
 root=pathlib.Path(__file__).resolve().parents[1];mode=sys.argv[1] if len(sys.argv)>1 else 'dev'
 if mode=='dev':
  rows=[eval_one(seed,m) for seed in (65501,65502) for m in METHODS]
  with (root/'source/development.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
  print(rows)
 elif mode=='fresh':
  rows=[eval_one(seed,m) for seed in (65511,65512,65513) for m in METHODS]
  with (root/'source/fresh.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
  for row in rows:print(row)
 else:raise SystemExit('usage: engine.py dev|fresh')
