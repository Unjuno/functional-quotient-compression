#!/usr/bin/env python3
"""DARE sparse deltas and native parameter-superposition bind/unbind controls."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D,MAXN,P=1024,32,128
SEEDS={'dev-71201':71201,'audit-71211':71211,'audit-71212':71212,'audit-71213':71213}
COUNTS=(2,4,8,16,32);RHOS=(0.0,0.1);KEEP=.1
def pack(**a):
 b=io.BytesIO();np.savez(b,**a);return b.getvalue()
def context_random(seed,task):return np.random.default_rng(seed+task).integers(0,2,size=D,dtype=np.int8)*2-1
def context_hadamard(task):
 j=np.arange(D,dtype=np.uint32);return np.where((np.bitwise_count(j & np.uint32(task))%2)==0,1,-1).astype(np.int8)
def dare_rows(deltas,rng):
 masks=rng.random(deltas.shape)<KEEP;return deltas*masks/KEEP,masks
def sparse_pack(base,deltas,masks):
 offsets=[0];idx=[];vals=[]
 for row,m in zip(deltas,masks):
  ix=np.flatnonzero(m);idx.extend(ix.tolist());vals.extend(row[ix].tolist());offsets.append(len(idx))
 return pack(base=base.astype(np.float32),offsets=np.asarray(offsets,np.int32),indices=np.asarray(idx,np.uint16),values=np.asarray(vals,np.float32),metadata=np.asarray([D,len(deltas)],np.int32))
def sparse_decode(p):
 n=len(p['offsets'])-1;out=np.zeros((n,D),np.float32)
 for i in range(n):
  a,b=int(p['offsets'][i]),int(p['offsets'][i+1]);out[i,p['indices'][a:b].astype(int)]=p['values'][a:b]
 return out
def ps_pack(base,bound,n,seed,context_kind):
 return pack(base=base.astype(np.float32),bound_delta=bound.astype(np.float32),context_seed=np.asarray([seed],np.int64),task_ids=np.arange(n,dtype=np.int16),context_kind=np.asarray([1 if context_kind=='hadamard' else 0],np.int8),metadata=np.asarray([D,n],np.int32))
def ps_decode(p,context_kind):
 n=len(p['task_ids']);seed=int(p['context_seed'][0]);bound=p['bound_delta'];out=[]
 for t in p['task_ids'].astype(int):
  c=context_hadamard(t) if context_kind=='hadamard' else context_random(seed,t)
  out.append(c*bound)
 return np.stack(out)
def nrmse(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
def run_one(label,seed,outdir,counts=COUNTS,rhos=RHOS):
 rng=np.random.default_rng(seed);outdir.mkdir(parents=True,exist_ok=True);rows=[]
 base=rng.normal(size=D).astype(np.float32)*.02
 # Prefix nesting lets task-count quality change without changing earlier task deltas.
 raw_all=rng.normal(size=(MAXN,D)).astype(np.float32)*.05
 probes=rng.normal(size=(MAXN,P,D)).astype(np.float32)
 for rho in rhos:
  raw=raw_all.copy()
  if rho:
   private=rng.normal(size=raw.shape).astype(np.float32);private*=rho*np.linalg.norm(raw,axis=1,keepdims=True)/np.maximum(np.linalg.norm(private,axis=1,keepdims=True),1e-30);raw+=private
  dare_all,mask_all=dare_rows(raw,rng)
  for n in counts:
   d=raw[:n];dd=dare_all[:n];mm=mask_all[:n];xs=probes[:n]
   target=[x@v for x,v in zip(xs,d)]
   dare_payload=sparse_pack(base,dd,mm)
   models={'independent_dense':(pack(base=base,deltas=d.astype(np.float32),metadata=np.asarray([D,n],np.int32)),d,'dense'),
    'independent_DARE':(dare_payload,dd,'sparse')}
   for ctx in ('random','hadamard'):
    C=np.stack([context_hadamard(t) if ctx=='hadamard' else context_random(seed,t) for t in range(n)]).astype(np.float32)
    bound=np.sum(C*(dd if ctx else d),axis=0)
    # Context types are identical across DARE/no-DARE methods; arrays are regenerated from paid seed/IDs.
    if ctx=='random':
     bound_raw=np.sum(C*d,axis=0);bound_dare=np.sum(C*dd,axis=0)
     models['PS_random_noDARE']=(ps_pack(base,bound_raw,n,seed,'random'),None,'ps_random')
     models['PS_random_DARE']=(ps_pack(base,bound_dare,n,seed,'random'),None,'ps_random')
    else:
     bound_dare=np.sum(C*dd,axis=0)
     models['Mirror_Hadamard_DARE']=(ps_pack(base,bound_dare,n,seed,'hadamard'),None,'ps_hadamard')
     models['native_PS_Hadamard_DARE']=(ps_pack(base,bound_dare,n,seed,'hadamard'),None,'ps_hadamard')
   for method,(payload,recon,kind) in models.items():
    (outdir/f'{label}_n{n}_rho{rho:g}.{method}.npz').write_bytes(payload)
    if kind=='sparse':
     with np.load(io.BytesIO(payload),allow_pickle=False) as q: replay=sparse_decode(q)
    elif kind=='dense':
     with np.load(io.BytesIO(payload),allow_pickle=False) as q: replay=q['deltas']
    elif kind=='ps_random':
     with np.load(io.BytesIO(payload),allow_pickle=False) as q: replay=ps_decode(q,'random')
    else:
     with np.load(io.BytesIO(payload),allow_pickle=False) as q: replay=ps_decode(q,'hadamard')
    errs=[nrmse(x@v,y) for x,v,y in zip(xs,replay,target)]
    dare_rel=nrmse(replay,dd)
    nnz=int(mm.sum());ops=(nnz//n if method=='independent_DARE' else D)
    rows.append(dict(condition=label,seed=seed,task_count=n,private_delta_fraction=rho,method=method,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),mean_output_nrmse=float(np.mean(errs)),max_output_nrmse=float(np.max(errs)),nrmse_vs_independent_DARE=dare_rel,dare_nonzeros=nnz,examples=n*P,optimizer_updates=0,bind_wall_s=0.0,active_compute_proxy=ops,context_kind=kind,status_note='synthetic linear deltas; base and context metadata charged; sparse indices uint16'))
 return rows
def main():
 rows=[]
 for label,seed in SEEDS.items():rows.extend(run_one(label,seed,ROOT/'source'/'payloads'))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('wrote',len(rows),'model-level rows')
if __name__=='__main__':main()
