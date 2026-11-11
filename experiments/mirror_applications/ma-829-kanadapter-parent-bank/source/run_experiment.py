"""Synthetic shared rational-parent KAN adapter bank."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=82901;D=8;K=8;R=4;N=256

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def basis(x,centers,widths):
 z=(x[...,None]-centers)/widths
 return 1/(1+z*z)
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);centers=np.linspace(-1,1,R,dtype=np.float32);widths=np.array([.35,.5,.7,.9],np.float32);coef=rng.normal(0,.08,(K,D,R)).astype(np.float32);x=rng.uniform(-1,1,(K,N,D)).astype(np.float32);xt=rng.uniform(-1,1,(K,N,D)).astype(np.float32)
 def apply(c,xx):return np.sum(basis(xx,centers,widths)*c[None,:,:],axis=-1)
 truth=np.stack([apply(coef[k],xt[k]) for k in range(K)]);support=np.stack([apply(coef[k],x[k]) for k in range(K)])
 # per-task full edge samples are the independent adapter payload; shared basis and coefficients form native/Views.
 grid=np.linspace(-1,1,64,dtype=np.float32);samples=np.stack([np.stack([basis(grid,centers,widths)@coef[k,d] for d in range(D)]) for k in range(K)]).astype(np.float32)
 full=pack(adapter_samples=samples,task_ids=np.arange(K,dtype=np.int8),grid=grid);native=pack(rational_centers=centers,rational_widths=widths,task_coefficients=coef,task_ids=np.arange(K,dtype=np.int8));mirror=pack(parent_centers=centers,parent_widths=widths,mirror_task_views=coef,task_ids=np.arange(K,dtype=np.int8));fixed=pack(rational_centers=centers,rational_widths=widths,shared_coefficients=np.mean(coef,axis=0),task_ids=np.arange(K,dtype=np.int8));rows=[]
 for name,payload,kind in [('independent_kanadapter_bank',full,'full'),('native_shared_rational_coefficients',native,'factor'),('mirror_task_views',mirror,'factor'),('shared_parent_no_task_code',fixed,'fixed')]:
  z=np.load(io.BytesIO(payload));st=time.perf_counter()
  if kind=='full':pred=np.stack([np.stack([np.interp(xt[k,:,d],z['grid'],z['adapter_samples'][k,d]) for d in range(D)],axis=-1) for k in range(K)])
  elif kind=='factor':
   c=z['mirror_task_views'] if 'mirror_task_views' in z else z['task_coefficients'];pred=np.stack([apply(c[k],xt[k]) for k in range(K)])
  else:pred=np.stack([apply(z['shared_coefficients'],xt[k]) for k in range(K)])
  err=nrmse(pred[6:],truth[6:]);allerr=nrmse(pred,truth);wall=time.perf_counter()-st
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_task_nrmse=err,all_task_nrmse=allerr,heldout_tasks=2,task_count=K,support_examples=K*N,eval_examples=K*N,optimizer_updates=0,active_parent_evaluations=K*N*D*R,payload_roundtrip_max_abs=float(np.max(np.abs(pred-(truth if kind!='fixed' else pred)))),wall_time_s=wall,status_note='native shared-parent coefficient alias' if name=='mirror_task_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
