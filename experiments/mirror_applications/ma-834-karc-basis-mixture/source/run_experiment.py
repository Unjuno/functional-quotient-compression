"""Synthetic task-adaptive mixture of explicit KARC basis families."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=83401;T=9;N=256;NS=64;F=3

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def fourier(x):return np.stack([np.ones_like(x),np.sin(np.pi*x),np.cos(np.pi*x),np.sin(2*np.pi*x),np.cos(2*np.pi*x)],axis=-1)
def cheb(x):return np.stack([np.ones_like(x),x,2*x*x-1,4*x**3-3*x,8*x**4-8*x*x+1],axis=-1)
def spline(x):return np.stack([np.ones_like(x),np.maximum(0,x+.5)**3,np.maximum(0,x)**3,np.maximum(0,x-.5)**3,x],axis=-1)
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);funcs=[fourier,spline,cheb];x=rng.uniform(-1,1,(T,N)).astype(np.float32);xs=rng.uniform(-1,1,(T,NS)).astype(np.float32);family=np.repeat(np.arange(F),3);coef=rng.normal(0,.2,(T,5)).astype(np.float32);truth=np.stack([funcs[family[t]](x[t])@coef[t] for t in range(T)]);support=np.stack([funcs[family[t]](xs[t])@coef[t] for t in range(T)])
 coeff_all=np.zeros((T,F,5),np.float32)
 for t in range(T):coeff_all[t,family[t]]=coef[t]
 selected=np.zeros(T,np.int8);selected_coef=np.zeros((T,5),np.float32)
 for t in range(T):
  errors=[];fits=[]
  for fi,fun in enumerate(funcs):
   ff=fun(xs[t]);fit=np.linalg.lstsq(ff,support[t],rcond=None)[0].astype(np.float32);errors.append(float(np.mean((ff@fit-support[t])**2)));fits.append(fit)
  selected[t]=int(np.argmin(errors));selected_coef[t]=fits[selected[t]]
 grid=np.linspace(-1,1,128,dtype=np.float32);tables=np.stack([funcs[family[t]](grid)@coef[t] for t in range(T)]).astype(np.float32)
 independent=pack(function_tables=tables,grid=grid,task_ids=np.arange(T,dtype=np.int8));full=pack(fourier_coefficients=coeff_all[:,0],spline_coefficients=coeff_all[:,1],chebyshev_coefficients=coeff_all[:,2],family_ids=family.astype(np.int8),task_ids=np.arange(T,dtype=np.int8));mirror=pack(basis_ids=np.array([0,1,2],np.int8),task_view_codes=selected,task_coefficients=selected_coef,task_ids=np.arange(T,dtype=np.int8));native=pack(basis_ids=np.array([0,1,2],np.int8),family_ids=selected,selected_coefficients=selected_coef,task_ids=np.arange(T,dtype=np.int8));fixed=pack(fourier_coefficients=coef,task_ids=np.arange(T,dtype=np.int8));rows=[]
 for name,payload,kind in [('independent_sampled_tables',independent,'ind'),('full_heterogeneous_karc_bank',full,'full'),('native_karc_family_selection',native,'native'),('mirror_sparse_basis_views',mirror,'view'),('fixed_fourier_only',fixed,'fixed')]:
  z=np.load(io.BytesIO(payload));st=time.perf_counter()
  if kind=='ind':pred=np.stack([np.interp(x[t],z['grid'],z['function_tables'][t]) for t in range(T)]);active=T*N*128
  elif kind=='full':
   bank=[z['fourier_coefficients'],z['spline_coefficients'],z['chebyshev_coefficients']];pred=np.stack([funcs[family[t]](x[t])@bank[family[t]][t] for t in range(T)]);active=T*N*5
  elif kind in ('view','native'):
   ids=z['task_view_codes'] if 'task_view_codes' in z else z['family_ids'];cc=z['task_coefficients'] if 'task_coefficients' in z else z['selected_coefficients'];pred=np.stack([funcs[int(ids[t])](x[t])@cc[t] for t in range(T)]);active=T*N*5
  else:pred=np.stack([fourier(x[t])@z['fourier_coefficients'][t] for t in range(T)]);active=T*N*5
  err=nrmse(pred,truth);sel=float(np.mean(selected==family)) if kind in ('view','native') else float('nan');wall=time.perf_counter()-st
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_function_nrmse=err,task_count=T,support_points=T*NS,test_points=T*N,active_feature_evaluations=active,selection_accuracy=sel,payload_roundtrip_max_abs=0.0,wall_time_s=wall,status_note='native family-ID selection alias' if name=='mirror_sparse_basis_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
