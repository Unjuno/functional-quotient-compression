"""Synthetic BOFT-style depth-view screen for MA-276."""
from __future__ import annotations
import hashlib,struct,time
import numpy as np
D=8;DEPTH=4;N=512
METHODS=('hard_tie','mirror_boft','scalar_gate','static_lora_rank2','generated_basis','boft_per_depth','untied_full')
STAGES=(1.,-.7,.5)

def pack(records):
 out=bytearray(b'MA276\x01');out.extend(struct.pack('<H',len(records)))
 for name,value in records:
  n=name.encode();out.extend(struct.pack('<H',len(n)));out.extend(n)
  if isinstance(value,np.ndarray):
   a=np.asarray(value)
   if a.dtype.kind=='f':a=np.asarray(a,dtype=np.dtype('<f4'),order='C')
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
 if payload[:6]!=b'MA276\x01':raise ValueError('bad magic')
 p=6;n=struct.unpack_from('<H',payload,p)[0];p+=2;out=[]
 for _ in range(n):
  ln=struct.unpack_from('<H',payload,p)[0];p+=2;name=payload[p:p+ln].decode();p+=ln;dn=payload[p];p+=1;dt=payload[p:p+dn].decode() if dn else '';p+=dn;nd=payload[p];p+=1;shape=[]
  for _ in range(nd):shape.append(struct.unpack_from('<I',payload,p)[0]);p+=4
  nb=struct.unpack_from('<I',payload,p)[0];p+=4;data=payload[p:p+nb];p+=nb
  out.append((name,np.frombuffer(data,dtype=np.dtype(dt)).reshape(shape).copy() if dt else data))
 if p!=len(payload):raise ValueError('trailing data')
 return out

def _rotate_pair(m,i,j,angle):
 c,s=np.cos(angle),np.sin(angle);g=np.eye(D);g[i,i]=c;g[j,j]=c;g[i,j]=-s;g[j,i]=s
 return m@g

def boft(angle_or_angles):
 m=np.eye(D);idx=0
 if np.ndim(angle_or_angles)==0:
  angles=np.concatenate([np.full(4,float(angle_or_angles)*k) for k in STAGES])
 else:angles=np.asarray(angle_or_angles).reshape(-1)
 for stage,stride in enumerate((1,2,4)):
  for base in range(0,D,2*stride):
   for off in range(stride):
    i,j=base+off,base+off+stride;m=_rotate_pair(m,i,j,float(angles[idx]));idx+=1
 return m

def make_world(seed,condition):
 rng=np.random.default_rng(seed);q,_=np.linalg.qr(rng.normal(size=(D,D)));w0=.72*np.eye(D)+.18*q
 angles=np.array([0.,*rng.uniform(-.65,.65,size=DEPTH-1)])
 if condition=='aligned':teachers=[boft(a)@w0@boft(a).T for a in angles]
 elif condition=='independent':teachers=[.72*np.eye(D)+.18*np.linalg.qr(rng.normal(size=(D,D)))[0] for _ in range(DEPTH)];w0=teachers[0].copy()
 else:raise ValueError(condition)
 x=rng.normal(size=(N,D));return w0,angles,teachers,x

def _lowrank(r,rank):
 u,s,vh=np.linalg.svd(r,full_matrices=False);root=np.sqrt(s[:rank]);return u[:,:rank]*root,root[:,None]*vh[:rank]

def _generated_basis(residuals,rank):
 flat=np.stack([r.reshape(-1) for r in residuals]);_,_,vt=np.linalg.svd(flat,full_matrices=False);basis=vt[:rank].reshape(rank,D,D);codes=flat@vt[:rank].T;return basis,codes

def serialize(method,w0,angles,teachers,rank):
 rec=[('method',method)]
 if method!='untied_full':rec.append(('shared_block',w0))
 if method!='hard_tie':rec.append(('depth_ids',np.arange(DEPTH,dtype=np.uint8)))
 if method=='mirror_boft':rec.append(('mirror_angles',angles))
 elif method=='scalar_gate':rec.append(('gates',np.array([np.sum((t-w0)*w0)/(np.sum(w0*w0)+1e-12) for t in teachers])))
 elif method=='static_lora_rank2':
  for i,t in enumerate(teachers):
   u,v=_lowrank(t-w0,2);rec += [(f'u_{i}',u),(f'v_{i}',v)]
 elif method=='generated_basis':
  b,c=_generated_basis([t-w0 for t in teachers],rank);rec += [('residual_basis',b),('depth_codes',c)]
 elif method=='boft_per_depth':
  # Strong independent BOFT control: one angle for each sparse butterfly pair per depth.
  rec.append(('boft_angles',np.stack([np.concatenate([np.full(4,a*k) for k in STAGES]) for a in angles])))
 elif method=='untied_full':
  for i,t in enumerate(teachers):rec.append((f'block_{i}',t))
 return pack(rec)

def restore(payload,rank=2):
 r=dict(unpack(payload));method=r['method'].decode();w0=r.get('shared_block');out=[]
 if method=='hard_tie':return [w0.copy() for _ in range(DEPTH)]
 if method=='mirror_boft':return [boft(a)@w0@boft(a).T for a in r['mirror_angles']]
 if method=='scalar_gate':return [(1+float(g))*w0 for g in r['gates']]
 if method=='static_lora_rank2':return [w0+r[f'u_{i}']@r[f'v_{i}'] for i in range(DEPTH)]
 if method=='generated_basis':return [w0+np.tensordot(c,r['residual_basis'],axes=(0,0)) for c in r['depth_codes']]
 if method=='boft_per_depth':return [boft(a)@w0@boft(a).T for a in r['boft_angles']]
 if method=='untied_full':return [r[f'block_{i}'] for i in range(DEPTH)]
 raise ValueError(method)

def prepare_runtime(method,records):
 if method=='mirror_boft':return [boft(a) for a in records['mirror_angles']]
 if method=='boft_per_depth':return [boft(a) for a in records['boft_angles']]
 return None

def apply_runtime(method,x,records,layer,prepared=None):
 r=records;w0=r.get('shared_block')
 if method=='hard_tie':return x@w0
 if method=='mirror_boft':
  b=prepared[layer] if prepared is not None else boft(float(r['mirror_angles'][layer]));return (x@b@w0)@b.T
 if method=='scalar_gate':return x@((1+float(r['gates'][layer]))*w0)
 if method=='static_lora_rank2':return x@w0+(x@r[f'u_{layer}'])@r[f'v_{layer}']
 if method=='generated_basis':return x@w0+sum(float(c) *(x@mat) for c,mat in zip(r['depth_codes'][layer],r['residual_basis']))
 if method=='boft_per_depth':
  b=prepared[layer] if prepared is not None else boft(r['boft_angles'][layer]);return (x@b@w0)@b.T
 if method=='untied_full':return x@r[f'block_{layer}']
 raise ValueError(method)

def _composition(mats,x):
 y=x
 for w in mats:y=y@w
 return y

def run_world(seed,condition,rank,split):
 w0,angles,teachers,x=make_world(seed,condition);rows=[]
 for method in METHODS:
  t0=time.perf_counter();payload=serialize(method,w0,angles,teachers,rank);states=restore(payload,rank);elapsed=time.perf_counter()-t0
  assert pack(unpack(payload))==payload
  decoded=dict(unpack(payload));prepare_start=time.perf_counter();prepared=prepare_runtime(method,decoded);prepare_s=time.perf_counter()-prepare_start;layer_mse=float(np.mean([np.mean((x@a-x@b)**2) for a,b in zip(teachers,states)]))
  comp_mse=float(np.mean((_composition(teachers,x)-_composition(states,x))**2))
  pathdiff=max(float(np.max(np.abs(apply_runtime(method,x,decoded,l,prepared)-x@states[l]))) for l in range(DEPTH))
  workspace_bytes=(len(prepared)*D*D*8 if prepared is not None else 0)
  # A compact MAC estimate; butterfly rotation overhead is charged in mirror/BOFT paths.
  ops={'hard_tie':DEPTH*D*D,'mirror_boft':DEPTH*(D*D+24*D),'scalar_gate':DEPTH*D*D,'static_lora_rank2':DEPTH*(D*D+4*D),'generated_basis':DEPTH*(D*D+2*rank*D),'boft_per_depth':DEPTH*(D*D+24*D),'untied_full':DEPTH*D*D}[method]
  it=time.perf_counter()
  for _ in range(100):
   y=x
   for l in range(DEPTH):y=apply_runtime(method,y,decoded,l,prepared)
  infer_s=time.perf_counter()-it;ips=100*N*DEPTH/max(infer_s,1e-12)
  rows.append(dict(split=split,seed=seed,condition=condition,method=method,rank=rank,depth=DEPTH,layer_output_mse=layer_mse,composed_output_mse=comp_mse,serialized_bytes=len(payload),optimizer_updates=0,active_compute_proxy=ops,operator_prepare_s=prepare_s,operator_workspace_bytes=workspace_bytes,wall_time_s=elapsed,inference_examples_per_s=ips,payload_sha256=hashlib.sha256(payload).hexdigest(),reconstruction_max_abs_diff=max(float(np.max(np.abs(a-b))) for a,b in zip(teachers,states)),runtime_path_max_abs_diff=pathdiff))
 return rows

def run(seed,split,rank):return run_world(seed,'aligned',rank,split)+run_world(seed,'independent',rank,split)
def deterministic_rows(seed,split,rank):
 a=run(seed,split,rank);b=run(seed,split,rank);keys=('layer_output_mse','composed_output_mse','serialized_bytes','payload_sha256')
 if [[r[k] for k in keys] for r in a]!=[[r[k] for k in keys] for r in b]:raise AssertionError('replay mismatch')
 return a
