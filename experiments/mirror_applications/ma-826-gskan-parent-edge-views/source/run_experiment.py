"""Synthetic GS-KAN parent function and edge transform codes."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=82601;E=32;N=128;NT=256

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def parent(x):return np.sin(x)+.2*x**3
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);a=rng.uniform(.65,1.35,E).astype(np.float32);b=rng.uniform(-.3,.3,E).astype(np.float32);x=rng.uniform(-1,1,(E,NT)).astype(np.float32);truth=np.stack([parent(a[e]*x[e]+b[e]) for e in range(E)])
 train_idx=np.arange(E-8);held=np.arange(E-8,E);grid=np.linspace(-1.5,1.5,64,dtype=np.float32);samples=np.stack([parent(a[e]*grid+b[e]) for e in range(E)]).astype(np.float32)
 full=pack(edge_samples=samples,edge_ids=np.arange(E,dtype=np.int8),grid=grid);native=pack(parent_coeff=np.array([.2],np.float32),edge_scale=a,edge_shift=b,edge_ids=np.arange(E,dtype=np.int8));mirror=pack(parent_coeff=np.array([.2],np.float32),view_scale=a,view_shift=b,edge_ids=np.arange(E,dtype=np.int8));mod=pack(parent_coeff=np.array([.2],np.float32),view_scale=a,view_shift=b,modulation=np.ones(E,np.float32),edge_ids=np.arange(E,dtype=np.int8));rows=[]
 for name,payload,kind in [('independent_kan_edge_samples',full,'full'),('native_gskan_affine_parent',native,'affine'),('mirror_affine_edge_views',mirror,'affine'),('mirror_affine_modulation_views',mod,'mod')]:
  z=np.load(io.BytesIO(payload))
  if kind=='full':rec=np.stack([np.interp(x[e],z['grid'],z['edge_samples'][e]) for e in range(E)])
  else:
   sc=z['edge_scale'] if 'edge_scale' in z else z['view_scale'];sh=z['edge_shift'] if 'edge_shift' in z else z['view_shift'];m=z['modulation'] if 'modulation' in z else np.ones(E,np.float32);rec=np.stack([m[e]*parent(sc[e]*x[e]+sh[e]) for e in range(E)])
  err=nrmse(rec[held],truth[held]);rt=0.0;evals=E*NT;st=time.perf_counter();_=np.stack([parent(a[e]*x[e]+b[e]) for e in range(E)]);wall=time.perf_counter()-st
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_edge_nrmse=err,all_edge_nrmse=nrmse(rec,truth),heldout_edges=len(held),edge_count=E,test_samples=E*NT,parent_function_evaluations=evals,payload_roundtrip_max_abs=rt,wall_time_s=wall,status_note='native GS-KAN affine alias' if name=='mirror_affine_edge_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
