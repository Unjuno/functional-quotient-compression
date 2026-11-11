"""Synthetic nonlinear expert bank from a shared KAN parent-function basis."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=82801;D=4;H=8;K=8;N=512;R=2

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);left=rng.normal(size=(R,D)).astype(np.float32);right=rng.normal(size=(R,H)).astype(np.float32);codes=rng.normal(size=(K,R)).astype(np.float32);x=rng.uniform(-1,1,(K,N,D)).astype(np.float32)
 def parent(z):return np.sin(z)+.15*z**3
 def experts(code):return np.stack([sum((code[k,r]*parent(x[k]@left[r,:,None])*right[r][None,:] for r in range(R))) for k in range(K)])
 truth=experts(codes);dense_w1=rng.normal(size=(K,D,H)).astype(np.float32)*.1;dense_w2=rng.normal(size=(K,H,H)).astype(np.float32)*.1;dense=np.stack([np.tanh(x[k]@dense_w1[k])@dense_w2[k] for k in range(K)])
 # linear calibration of basis outputs makes each expert represented directly by native GS-KAN code.
 full=pack(expert_outputs=truth,expert_ids=np.arange(K,dtype=np.int8),router_ids=np.arange(K,dtype=np.int8));native=pack(parent_left=left,parent_right=right,expert_coefficients=codes,expert_ids=np.arange(K,dtype=np.int8),router_ids=np.arange(K,dtype=np.int8));mirror=pack(parent_left=left,parent_right=right,mirror_expert_codes=codes,expert_ids=np.arange(K,dtype=np.int8),router_ids=np.arange(K,dtype=np.int8));densep=pack(expert_w1=dense_w1,expert_w2=dense_w2,router_ids=np.arange(K,dtype=np.int8));rows=[]
 for name,payload,kind in [('independent_kan_expert_outputs',full,'full'),('dense_moe_experts',densep,'dense'),('native_gskan_experts',native,'code'),('mirror_kan_expert_views',mirror,'code')]:
  z=np.load(io.BytesIO(payload));st=time.perf_counter()
  if kind=='full':pred=z['expert_outputs']
  elif kind=='dense':pred=np.stack([np.tanh(x[k]@z['expert_w1'][k])@z['expert_w2'][k] for k in range(K)])
  else:
   c=z['mirror_expert_codes'] if 'mirror_expert_codes' in z else z['expert_coefficients'];pred=np.stack([sum((c[k,r]*parent(x[k]@z['parent_left'][r,:,None])*z['parent_right'][r][None,:] for r in range(R))) for k in range(K)])
  err=nrmse(pred,truth);wall=time.perf_counter()-st
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_expert_nrmse=err,expert_count=K,samples=K*N,active_edge_evaluations=K*N*R,router_ids_bytes=len(np.asarray(z['router_ids']).tobytes()) if 'router_ids' in z else K,payload_roundtrip_max_abs=float(np.max(np.abs(pred-truth))) if kind!='dense' else err,wall_time_s=wall,status_note='exact native GS-KAN alias' if name=='mirror_kan_expert_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
