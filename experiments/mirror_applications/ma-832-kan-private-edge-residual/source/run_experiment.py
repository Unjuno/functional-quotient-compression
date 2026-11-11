"""Synthetic shared KAN parent with sparse private rational edge residuals."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=83201;E=32;NT=256

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def parent(x):return np.sin(x)+.2*x**3
def bump(x,c,w,a):return a/(1+((x-c)/w)**2)
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);exc=np.array([3,9,17,26],np.int8);c=rng.uniform(-.5,.5,len(exc)).astype(np.float32);w=np.full(len(exc),.12,np.float32);a=rng.uniform(.15,.25,len(exc)).astype(np.float32);x=rng.uniform(-1,1,(E,NT)).astype(np.float32);truth=np.stack([parent(x[e])+(bump(x[e],c[list(exc).index(e)],w[list(exc).index(e)],a[list(exc).index(e)]) if e in exc else 0) for e in range(E)]);held=np.arange(E-8,E);grid=np.linspace(-1,1,128,dtype=np.float32);samples=np.stack([parent(grid)+(bump(grid,c[list(exc).index(e)],w[list(exc).index(e)],a[list(exc).index(e)]) if e in exc else 0) for e in range(E)])
 independent=pack(edge_functions=samples,grid=grid,edge_ids=np.arange(E,dtype=np.int8));shared=pack(parent_code=np.array([1],np.int8),edge_ids=np.arange(E,dtype=np.int8));mirror=pack(parent_code=np.array([1],np.int8),view_codes=np.ones(E,np.float32),private_edge_ids=exc,private_centers=c,private_widths=w,private_amplitudes=a,edge_ids=np.arange(E,dtype=np.int8));native=pack(parent_code=np.array([1],np.int8),edge_scales=np.ones(E,np.float32),residual_edge_ids=exc,rational_centers=c,rational_widths=w,rational_amplitudes=a,edge_ids=np.arange(E,dtype=np.int8));rows=[]
 for name,payload,kind in [('independent_edge_tables',independent,'ind'),('shared_parent_only',shared,'shared'),('mirror_sparse_private_residual',mirror,'res'),('native_gskan_plus_rational_residual',native,'res')]:
  z=np.load(io.BytesIO(payload));st=time.perf_counter()
  if kind=='ind':pred=np.stack([np.interp(x[e],z['grid'],z['edge_functions'][e]) for e in range(E)])
  elif kind=='shared':pred=np.stack([parent(x[e]) for e in range(E)])
  else:
   ids=z['private_edge_ids'] if 'private_edge_ids' in z else z['residual_edge_ids'];cc=z['private_centers'] if 'private_centers' in z else z['rational_centers'];ww=z['private_widths'] if 'private_widths' in z else z['rational_widths'];aa=z['private_amplitudes'] if 'private_amplitudes' in z else z['rational_amplitudes'];pred=[]
   for e in range(E):
    out=parent(x[e]);
    if e in ids: j=list(ids).index(e);out=out+bump(x[e],cc[j],ww[j],aa[j])
    pred.append(out)
   pred=np.stack(pred)
  err=nrmse(pred[held],truth[held]);allerr=nrmse(pred,truth);wall=time.perf_counter()-st
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_edge_nrmse=err,all_edge_nrmse=allerr,heldout_edges=len(held),edge_count=E,private_edge_count=(len(exc) if kind=='res' else 0),parent_evaluations=E*NT,private_function_evaluations=(len(exc)*NT if kind=='res' else 0),payload_roundtrip_max_abs=float(np.max(np.abs(pred-truth))) if kind!='ind' else 0.0,wall_time_s=wall,status_note='native rational residual alias' if name=='mirror_sparse_private_residual' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
