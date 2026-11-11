"""Synthetic separable task-by-layer KAN adapter factorization."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=83001;T=8;L=6;R=2;N=256;D=4

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def basis(x):return np.stack([np.sin(x),x**3],axis=-1)
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);tb=rng.normal(0,.1,(T,R)).astype(np.float32);lb=rng.normal(0,.1,(L,R)).astype(np.float32);coeff=tb[:,None,:]*lb[None,:,:];x=rng.uniform(-1,1,(T,L,N,D)).astype(np.float32);y=np.sum(basis(x)*coeff[:,:,None,None,:],axis=-1)
 train=[(t,l) for t in range(T) for l in range(L) if (t+l)%2];held=[(t,l) for t in range(T) for l in range(L) if not (t+l)%2];grid=np.linspace(-1,1,64,dtype=np.float32);allpairs=[(t,l) for t in range(T) for l in range(L)];samples=np.stack([np.sum(basis(np.broadcast_to(grid[:,None],(64,D)))*coeff[t,l][None,None,:],axis=-1) for t,l in allpairs]).astype(np.float32)
 full=pack(pair_functions=samples,grid=grid,pair_ids=np.array(allpairs,np.int8),held_pair_ids=np.array(held,np.int8));native=pack(parent_ids=np.array([1,3],np.int8),task_factors=tb,layer_factors=lb,train_pair_ids=np.array(train,np.int8),held_pair_ids=np.array(held,np.int8));mirror=pack(parent_ids=np.array([1,3],np.int8),view_task_factors=tb,view_layer_factors=lb,train_pair_ids=np.array(train,np.int8),held_pair_ids=np.array(held,np.int8));taskonly=pack(parent_ids=np.array([1,3],np.int8),task_factors=tb,train_pair_ids=np.array(train,np.int8),held_pair_ids=np.array(held,np.int8));layeronly=pack(parent_ids=np.array([1,3],np.int8),layer_factors=lb,train_pair_ids=np.array(train,np.int8),held_pair_ids=np.array(held,np.int8));rows=[]
 for name,payload,kind in [('independent_pair_functions',full,'full'),('native_hypernetwork_factors',native,'factor'),('mirror_task_layer_views',mirror,'factor'),('task_only',taskonly,'task'),('layer_only',layeronly,'layer')]:
  z=np.load(io.BytesIO(payload));st=time.perf_counter()
  if kind=='full':
   lookup={tuple(q):i for i,q in enumerate(z['pair_ids'])};pred=np.stack([np.stack([np.interp(x[t,l,:,d],z['grid'],z['pair_functions'][lookup[(t,l)],:,d]) for d in range(D)],axis=-1) for t,l in held])
  elif kind=='factor':c=(z['view_task_factors'] if 'view_task_factors' in z else z['task_factors'])[:,None,:]*(z['view_layer_factors'] if 'view_layer_factors' in z else z['layer_factors'])[None,:,:];pred=np.stack([np.sum(basis(x[t,l])*c[t,l][None,None,:],axis=-1) for t,l in held])
  elif kind=='task':pred=np.stack([np.sum(basis(x[t,l])*z['task_factors'][t][None,None,:],axis=-1) for t,l in held])
  else:pred=np.stack([np.sum(basis(x[t,l])*z['layer_factors'][l][None,None,:],axis=-1) for t,l in held])
  truth=np.stack([y[t,l] for t,l in held]);err=nrmse(pred,truth);wall=time.perf_counter()-st
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_pair_nrmse=err,heldout_pairs=len(held),task_count=T,layer_count=L,test_examples=len(held)*N*D,parent_evaluations=len(held)*N*D*R,payload_roundtrip_max_abs=0.0,wall_time_s=wall,status_note='native hypernetwork tensor-factor alias' if name=='mirror_task_layer_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
