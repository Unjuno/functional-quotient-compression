"""Measure logical roles per physical expert on a designed Givens orbit."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=78401;D=8;N=256
THETA=np.asarray([-.36,-.12,.12,.36],np.float32)
def qmat(theta):
 q=np.eye(D,dtype=np.float32);c=np.float32(np.cos(theta));s=np.float32(np.sin(theta))
 q[0,0]=c;q[0,1]=-s;q[1,0]=s;q[1,1]=c
 return q
def pack(**x):
 b=io.BytesIO();np.savez_compressed(b,**x);return b.getvalue()
def err(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b**2)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);base=rng.normal(0,.3,(D,D)).astype(np.float32)
 teacher=np.stack([qmat(t).T@base@qmat(t) for t in THETA])
 xs=rng.normal(size=(4,N,D)).astype(np.float32)
 targets=np.einsum('rij,rnj->rni',teacher,xs)
 rows=[]
 configs=[]
 # Standard pool centers are fixed by declared role-angle groups.
 centers={1:np.asarray([0.],np.float32),2:np.asarray([-.24,.24],np.float32),4:THETA.copy()}
 for k,cs in centers.items():
  assign=np.argmin(np.abs(THETA[:,None]-cs[None,:]),axis=1).astype(np.int16)
  pool=np.stack([qmat(c).T@base@qmat(c) for c in cs])
  configs.append((f'pool_k{k}_no_view',k,pool,cs,assign,None,False))
  delta=THETA-cs[assign]
  if k<4:
   configs.append((f'mirror_k{k}',k,pool,cs,assign,delta,True))
   configs.append((f'native_givens_k{k}',k,pool,cs,assign,delta,True))
 # independent K=4 upper control (no code needed)
 configs.append(('independent_full_k4',4,teacher.copy(),THETA.copy(),np.arange(4,dtype=np.int16),None,False))
 for name,k,pool,cs,assign,delta,view in configs:
  arrays={'expert_pool':pool,'center_angles':cs,'role_to_expert':assign,'role_ids':np.arange(4,dtype=np.int16)}
  if view:arrays['role_givens_delta']=delta
  payload=pack(**arrays);out=np.empty_like(targets);start=time.perf_counter()
  for rep in range(100):
   for r in range(4):
    w=pool[assign[r]]
    for n,x in enumerate(xs[r]):
     if view:
      q=qmat(delta[r]);z=q@x;z=w@z;out[r,n]=q.T@z
     else:out[r,n]=w@x
  wall=(time.perf_counter()-start)*1000/100
  rows.append({'seed':SEED,'method':name,'physical_pool_k':k,'logical_roles':4,'serialized_bytes':len(payload),'payload_sha256':hashlib.sha256(payload).hexdigest(),'heldout_operator_nrmse':err(out,targets),'active_macs_per_role_token':D*D+(8 if view else 0),'examples_per_role':N,'optimizer_updates':0,'eval_wall_ms_per_4x256':round(wall,6),'status_note':'exact native direct-angle alias' if name.startswith('mirror_') else ''})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
