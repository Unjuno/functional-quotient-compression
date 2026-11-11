"""Synthetic source-target factorized edge-function bank."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=82701;NODES=10;E=32;R=2;NT=256

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def parent(x):return np.sin(x)+.2*x**3
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);src=rng.integers(NODES,size=E,dtype=np.int8);dst=rng.integers(NODES,size=E,dtype=np.int8);sc=rng.normal(1,.08,(NODES,R)).astype(np.float32);tc=rng.normal(1,.08,(NODES,R)).astype(np.float32);sb=rng.normal(0,.05,NODES).astype(np.float32);tb=rng.normal(0,.05,NODES).astype(np.float32);a=np.sum(sc[src]*tc[dst],axis=1)/R;b=sb[src]+tb[dst];x=rng.uniform(-1,1,(E,NT)).astype(np.float32);truth=np.stack([parent(a[e]*x[e]+b[e]) for e in range(E)]);held=np.arange(E-8,E);grid=np.linspace(-1.5,1.5,64,dtype=np.float32);ind_samples=np.stack([parent(a[e]*grid+b[e]) for e in range(E)]).astype(np.float32)
 native=pack(parent_scale=np.array([.2],np.float32),edge_scale=a,edge_shift=b,source_ids=src,target_ids=dst);mirror=pack(parent_scale=np.array([.2],np.float32),source_scale=sc,target_scale=tc,source_bias=sb,target_bias=tb,source_ids=src,target_ids=dst);ind=pack(edge_samples=ind_samples,grid=grid,source_ids=src,target_ids=dst);sourceonly=pack(parent_scale=np.array([.2],np.float32),source_scale=sc,source_bias=sb,source_ids=src,target_ids=dst);rows=[]
 for name,payload,kind in [('independent_edge_samples',ind,'ind'),('native_edge_coefficients',native,'native'),('mirror_source_target_codes',mirror,'mirror'),('source_only_codes',sourceonly,'source')]:
  z=np.load(io.BytesIO(payload))
  if kind=='ind':rec=np.stack([np.interp(x[e],z['grid'],z['edge_samples'][e]) for e in range(E)])
  elif kind=='native':rec=np.stack([parent(z['edge_scale'][e]*x[e]+z['edge_shift'][e]) for e in range(E)])
  elif kind=='mirror':aa=np.sum(z['source_scale'][z['source_ids']]*z['target_scale'][z['target_ids']],axis=1)/R;bb=z['source_bias'][z['source_ids']]+z['target_bias'][z['target_ids']];rec=np.stack([parent(aa[e]*x[e]+bb[e]) for e in range(E)])
  else:rec=np.stack([parent(z['source_scale'][z['source_ids'][e],0]*x[e]+z['source_bias'][z['source_ids'][e]]) for e in range(E)])
  err=nrmse(rec[held],truth[held]);st=time.perf_counter();_=np.stack([parent(a[e]*x[e]+b[e]) for e in range(E)]);wall=time.perf_counter()-st
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_edge_nrmse=err,all_edge_nrmse=nrmse(rec,truth),heldout_edges=len(held),edge_count=E,parent_evaluations=E*NT,payload_roundtrip_max_abs=0.0,wall_time_s=wall,status_note='native coefficient factorization alias' if name=='mirror_source_target_codes' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
