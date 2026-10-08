"""NumPy Fast Weight Programmer / Mirror context screen for MA-288."""
from __future__ import annotations
import hashlib,struct,time
import numpy as np
D=8; CONTEXTS=4; WRITES=8; N_QUERY=512
METHODS=('shared_fwp','hard_tie','mirror','film_scalar','lowrank_residual','fwp_independent')

def pack(records):
 out=bytearray(b'MA288\x01');out.extend(struct.pack('<H',len(records)))
 for name,value in records:
  n=name.encode();out.extend(struct.pack('<H',len(n)));out.extend(n)
  if isinstance(value,np.ndarray):
   a=np.asarray(value)
   if a.dtype.kind=='f':a=np.asarray(a,dtype=np.dtype('<f4') if a.dtype.itemsize>2 else np.dtype('<f2'),order='C')
   elif a.dtype.kind in 'ui':a=np.asarray(a,dtype=np.dtype(a.dtype).newbyteorder('<'),order='C')
   else:raise TypeError(a.dtype)
   dt=a.dtype.str.encode();out.extend(struct.pack('<B',len(dt)));out.extend(dt);out.extend(struct.pack('<B',a.ndim))
   for d in a.shape:out.extend(struct.pack('<I',int(d)))
   data=a.tobytes(order='C')
  else:
   data=value if isinstance(value,bytes) else value.encode();out.extend(b'\x00\x01');out.extend(struct.pack('<I',len(data)))
  out.extend(struct.pack('<I',len(data)));out.extend(data)
 return bytes(out)

def unpack(payload):
 if payload[:6]!=b'MA288\x01':raise ValueError('invalid payload magic')
 p=6;n=struct.unpack_from('<H',payload,p)[0];p+=2;records=[]
 for _ in range(n):
  ln=struct.unpack_from('<H',payload,p)[0];p+=2;name=payload[p:p+ln].decode();p+=ln
  dn=payload[p];p+=1;dt=payload[p:p+dn].decode() if dn else '';p+=dn;nd=payload[p];p+=1;shape=[]
  for _ in range(nd):shape.append(struct.unpack_from('<I',payload,p)[0]);p+=4
  nb=struct.unpack_from('<I',payload,p)[0];p+=4;data=payload[p:p+nb];p+=nb
  value=np.frombuffer(data,dtype=np.dtype(dt)).reshape(shape).copy() if dt else data;records.append((name,value))
 if p!=len(payload):raise ValueError('trailing data')
 return records

def givens(angle):
 r=np.eye(D);c,s=np.cos(angle),np.sin(angle);r[:2,:2]=[[c,-s],[s,c]];return r

def delta_write(memory,x,y,eta=1.0):
 return memory+eta*np.outer(x,y-x@memory)

def make_world(seed,condition):
 rng=np.random.default_rng(seed);base=rng.normal(scale=.15,size=(D,D));angles=np.array([0.,*rng.uniform(-1.1,1.1,size=CONTEXTS-1)])
 if condition=='aligned':teachers=[givens(a)@base@givens(a).T for a in angles]
 elif condition=='independent':
  teachers=[rng.normal(scale=.15,size=(D,D)) for _ in range(CONTEXTS)];base=teachers[0].copy()
 else:raise ValueError(condition)
 queries=[rng.normal(size=(N_QUERY,D)) for _ in range(CONTEXTS)]
 return base,angles,teachers,queries

def learn_fwp(target):
 """Eight online delta-rule writes using a one-hot key basis."""
 memory=np.zeros((D,D));eye=np.eye(D)
 for i,x in enumerate(eye):memory=delta_write(memory,x,target[i])
 return memory

def fit_lowrank(base,target,rank):
 u,s,vh=np.linalg.svd(target-base,full_matrices=False);root=np.sqrt(s[:rank]);return u[:,:rank]*root,root[:,None]*vh[:rank]

def serialize(method,base,angles,teachers,rank):
 ids=np.arange(CONTEXTS,dtype=np.uint8);records=[('method',method)]
 if method in ('hard_tie','mirror','film_scalar','lowrank_residual'):records.append(('base',base))
 if method=='shared_fwp':records.append(('memory',learn_fwp(teachers[-1])))
 elif method=='hard_tie':pass
 elif method=='mirror':records += [('context_ids',ids),('angles',angles)]
 elif method=='film_scalar':
  scales=np.array([np.sum(t*base)/(np.sum(base*base)+1e-12) for t in teachers]);records += [('context_ids',ids),('scales',scales)]
 elif method=='lowrank_residual':
  records.append(('context_ids',ids))
  for i,t in enumerate(teachers):
   u,v=fit_lowrank(base,t,rank);records.extend([(f'u_{i}',u),(f'v_{i}',v)])
 elif method=='fwp_independent':
  records.append(('context_ids',ids))
  for i,t in enumerate(teachers):records.append((f'memory_{i}',learn_fwp(t)))
 return pack(records)

def restore(payload,rank=2):
 r=dict(unpack(payload));method=r['method'].decode();out=[]
 if method=='shared_fwp':return [r['memory'] for _ in range(CONTEXTS)]
 if method=='hard_tie':return [r['base'] for _ in range(CONTEXTS)]
 if method=='mirror':return [givens(float(a))@r['base']@givens(float(a)).T for a in r['angles']]
 if method=='film_scalar':return [float(a)*r['base'] for a in r['scales']]
 if method=='lowrank_residual':return [r['base']+r[f'u_{i}']@r[f'v_{i}'] for i in range(CONTEXTS)]
 if method=='fwp_independent':return [r[f'memory_{i}'] for i in range(CONTEXTS)]
 raise ValueError(method)

def _rotate_rows(x,angle,transpose=False):
 c,s=np.cos(angle),np.sin(angle);z=np.array(x,copy=True)
 if not transpose:z[...,0]=c*x[...,0]+s*x[...,1];z[...,1]=-s*x[...,0]+c*x[...,1]
 else:z[...,0]=c*x[...,0]-s*x[...,1];z[...,1]=s*x[...,0]+c*x[...,1]
 return z

def infer(method,x,records,context):
 if method=='shared_fwp':return x@records['memory']
 if method=='hard_tie':return x@records['base']
 if method=='mirror':
  angle=float(records['angles'][context]);return _rotate_rows(_rotate_rows(x,angle)@records['base'],angle,transpose=True)
 if method=='film_scalar':return float(records['scales'][context])*(x@records['base'])
 if method=='lowrank_residual':return x@records['base']+(x@records[f'u_{context}'])@records[f'v_{context}']
 if method=='fwp_independent':return x@records[f'memory_{context}']
 raise ValueError(method)

def run_world(seed,condition,rank,split):
 base,angles,teachers,queries=make_world(seed,condition);base_state=learn_fwp(teachers[0]);rows=[]
 for method in METHODS:
  start=time.perf_counter();payload=serialize(method,base_state,angles,teachers,rank);states=restore(payload,rank);elapsed=time.perf_counter()-start
  assert pack(unpack(payload))==payload
  errors=[float(np.mean((x@t-x@w)**2)) for x,t,w in zip(queries,teachers,states)]
  ops={'shared_fwp':WRITES*CONTEXTS*D*D*2,'hard_tie':WRITES*D*D*2,'mirror':WRITES*D*D*2+CONTEXTS*D*8,'film_scalar':WRITES*D*D*2+CONTEXTS*D*D,'lowrank_residual':WRITES*D*D*2+CONTEXTS*D*D*rank*2,'fwp_independent':WRITES*CONTEXTS*D*D*2}[method]
  # Time inference through the implementation path for each session.
  decoded=dict(unpack(payload))
  runtime_path_diff=max(float(np.max(np.abs(infer(method,queries[c],decoded,c)-queries[c]@states[c]))) for c in range(CONTEXTS))
  t0=time.perf_counter()
  for _ in range(100):
   for c in range(CONTEXTS): _=infer(method,queries[c],decoded,c)
  infer_s=time.perf_counter()-t0
  ips=100*CONTEXTS*N_QUERY/max(infer_s,1e-12)
  rows.append(dict(split=split,seed=seed,condition=condition,method=method,rank=rank,contexts=CONTEXTS,writes=WRITES*(CONTEXTS if method in ('shared_fwp','film_scalar','lowrank_residual','fwp_independent') else 1),optimizer_updates=0,task_output_mse=float(np.mean(errors)),max_context_mse=max(errors),serialized_bytes=len(payload),context_state_bytes=(0 if method in ('shared_fwp','hard_tie') else len(payload)-len(pack([('method',method)] if method=='fwp_independent' else [('method',method),('base',base)]))),active_compute_proxy=ops,wall_time_s=elapsed,inference_examples_per_s=ips,payload_sha256=hashlib.sha256(payload).hexdigest(),reconstruction_max_abs_diff=max(float(np.max(np.abs(t-w))) for t,w in zip(teachers,states)),runtime_path_max_abs_diff=runtime_path_diff))
 return rows

def run(seed,split,rank):
 return run_world(seed,'aligned',rank,split)+run_world(seed,'independent',rank,split)

def deterministic_rows(seed,split,rank):
 a=run(seed,split,rank);b=run(seed,split,rank);keys=('task_output_mse','serialized_bytes','payload_sha256')
 if [[r[k] for k in keys] for r in a]!=[[r[k] for k in keys] for r in b]:raise AssertionError('non-deterministic replay')
 return a
