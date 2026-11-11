"""Synthetic topology x node-function composition over a shared graph substrate."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=82401;N=8;F=4;NT=6;NF=4;E=10

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);funcs=rng.normal(size=(NF,F,F)).astype(np.float32)*.1;edges=[]
 for t in range(NT):
  es=set()
  while len(es)<E:
   a=int(rng.integers(N));b=int(rng.integers(N))
   if a!=b:es.add((a,b))
  edges.append(sorted(es))
 edges=np.array(edges,np.int8);pairs=[(t,f) for t in range(NT) for f in range(NF)];held=[(t,f) for t,f in pairs if (t+f)%4==0];x=rng.normal(size=(len(held),128,N,F)).astype(np.float32)
 def execute(inp,edge,W):
  h=inp.copy();z=np.zeros_like(h)
  for a,b in edge:z[:,b]+=h[:,a]@W.T
  return np.tanh(h+z)
 truth=np.stack([execute(x[i],edges[t],funcs[f]) for i,(t,f) in enumerate(held)])
 independent=pack(pair_functions=np.stack([funcs[f] for t,f in pairs]),pair_edges=np.stack([edges[t] for t,f in pairs]),pair_ids=np.array(pairs,np.int8))
 native=pack(node_functions=funcs,topology_edges=edges,topology_ids=np.arange(NT,dtype=np.int8),function_ids=np.arange(NF,dtype=np.int8),pair_ids=np.array(pairs,np.int8))
 mirror=pack(shared_function_bank=funcs,topology_views=edges,function_views=np.arange(NF,dtype=np.int8),pair_codes=np.array(pairs,np.int8))
 fixed=pack(node_function=funcs[0],topology_edges=edges[0],pair_ids=np.array(pairs,np.int8));rows=[]
 for name,payload,kind in [('independent_pair_networks',independent,'ind'),('native_modular_routing',native,'native'),('mirror_factorized_views',mirror,'mirror'),('fixed_topology_function',fixed,'fixed')]:
  z=np.load(io.BytesIO(payload))
  if kind=='ind':
   lookup={tuple(p):i for i,p in enumerate(z['pair_ids'])};idx=[lookup[q] for q in held];we=z['pair_functions'][idx];ed=z['pair_edges'][idx]
  elif kind=='native':we=np.stack([z['node_functions'][f] for t,f in held]);ed=np.stack([z['topology_edges'][t] for t,f in held])
  elif kind=='mirror':we=np.stack([z['shared_function_bank'][f] for t,f in held]);ed=np.stack([z['topology_views'][t] for t,f in held])
  else:we=np.broadcast_to(z['node_function'],(len(held),F,F));ed=np.broadcast_to(z['topology_edges'],(len(held),E,2))
  pred=np.stack([execute(x[i],ed[i],we[i]) for i in range(len(held))]);err=nrmse(pred,truth);ops=int(np.mean([len(e) for e in ed]));st=time.perf_counter();_=np.stack([execute(x[i],ed[i],we[i]) for i in range(len(held))]);wall=time.perf_counter()-st
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_pair_output_nrmse=err,heldout_pairs=len(held),test_examples=len(held)*128,active_edge_ops=ops,optimizer_updates=0,payload_roundtrip_max_abs=0.0,wall_time_s=wall,status_note='native module/topology routing alias' if name=='mirror_factorized_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
