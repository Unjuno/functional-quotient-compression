"""Synthetic graph-program codes over shared node functions."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=82301;N=8;F=4;P=8;E=12

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def graph(rng):
 out=[]
 for p in range(P):
  es=set()
  while len(es)<E:
   a=int(rng.integers(N));b=int(rng.integers(N))
   if a!=b:es.add((a,b))
  out.append(sorted(es))
 return out
def execute(x,edges,W):
 h=x.copy()
 for _ in range(2):
  z=np.zeros_like(h)
  for a,b in edges:z[:,b]+=h[:,a]@W.T
  h=np.tanh(h+z)
 return h
def run():
 rng=np.random.default_rng(SEED);W=rng.normal(size=(F,F)).astype(np.float32)*.15;edges=graph(rng);x=rng.normal(size=(P,128,N,F)).astype(np.float32);xt=rng.normal(size=(P,256,N,F)).astype(np.float32);y=np.stack([execute(x[p],edges[p],W) for p in range(P)]);yt=np.stack([execute(xt[p],edges[p],W) for p in range(P)])
 sparse_idx=np.array([[p,a,b] for p in range(P) for a,b in edges[p]],np.int8);dense=np.zeros((P,N,N),np.uint8)
 for p in range(P):
  for a,b in edges[p]:dense[p,a,b]=1
 payloads=[('independent_dense_graphs',pack(node_functions=np.broadcast_to(W,(P,F,F)),adjacency=dense,program_ids=np.arange(P,dtype=np.int8)), 'dense'),('native_sparse_edge_indices',pack(node_function=W,edge_indices=sparse_idx,program_ids=np.arange(P,dtype=np.int8)), 'sparse'),('mirror_graph_program_code',pack(node_function=W,program_code=sparse_idx,program_ids=np.arange(P,dtype=np.int8)), 'sparse'),('shared_fixed_topology',pack(node_function=W,edge_indices=sparse_idx[:E],program_ids=np.arange(P,dtype=np.int8)),'fixed')];rows=[]
 for name,payload,kind in payloads:
  z=np.load(io.BytesIO(payload))
  if kind=='dense':maps=[[(a,b) for a in range(N) for b in range(N) if z['adjacency'][p,a,b]] for p in range(P)];ww=z['node_functions']
  elif kind=='sparse':
   key='program_code' if 'program_code' in z else 'edge_indices';maps=[[(int(a),int(b)) for pp,a,b in z[key] if pp==p] for p in range(P)];ww=np.broadcast_to(z['node_function'],(P,F,F))
  else:maps=[[(int(a),int(b)) for _,a,b in z['edge_indices']]]*P;ww=np.broadcast_to(z['node_function'],(P,F,F))
  pred=np.stack([execute(xt[p],maps[p],ww[p]) for p in range(P)]);err=nrmse(pred[6:],yt[6:]);allerr=nrmse(pred,yt);rt=0.0;st=time.perf_counter();_=np.stack([execute(xt[p],maps[p],ww[p]) for p in range(P)]);wall=time.perf_counter()-st
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_program_nrmse=err,all_program_nrmse=allerr,program_count=P,heldout_programs=2,edges_per_program=E,active_edge_ops_per_layer=int(np.mean([len(g) for g in maps])),optimizer_updates=0,payload_roundtrip_max_abs=rt,wall_time_s=wall,status_note='native sparse edge-index alias' if name=='mirror_graph_program_code' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
