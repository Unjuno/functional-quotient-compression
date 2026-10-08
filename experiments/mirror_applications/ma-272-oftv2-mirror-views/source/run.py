#!/usr/bin/env python3
import argparse,io,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import solve
D,T=16,6;TRI=np.tril_indices(D,-1)
def pack_skew(a):return a[...,TRI[0],TRI[1]]
def unpack_skew(v):
 a=np.zeros((D,D));a[TRI]=v;a[(TRI[1],TRI[0])]=-v;return a
def skew(g,scale=.015):
 a=g.normal(size=(D,D))*scale;return a-a.T
def cayley(a):
 I=np.eye(D);return solve(I-a,I+a)
def neumann(a,k):
 I=np.eye(D);v=I.copy();term=I.copy()
 for _ in range(k):term=term@a;v+=term
 return v@(I+a)
def apply_neumann(x,a,k):
 acc=x.copy();term=x.copy()
 for _ in range(k):term=term@a;acc+=term
 return acc@(np.eye(D)+a)
def world(seed,cond):
 g=np.random.default_rng(seed);W=g.normal(size=(D,D))*.3;m=np.linspace(-1,1,T)
 if cond=='shared_generator_orbit':S=skew(g);ss=np.tile(S,(T,1,1));sc=m
 else:ss=np.array([skew(g) for _ in range(T)]);sc=np.ones(T)
 Q=np.array([cayley(sc[i]*ss[i]) for i in range(T)]);xs=[g.normal(size=(2048,D)) for _ in range(T)];ys=[(x@q)@W for x,q in zip(xs,Q)]
 return W,ss,sc,Q,xs,ys
def payload(st):
 b=io.BytesIO();np.savez(b,**{f'p{i}':np.asarray(x) for i,x in enumerate(st)});return len(b.getvalue())
def bench(fn,reps=60):
 t=time.perf_counter()
 for _ in range(reps):fn()
 return (time.perf_counter()-t)/reps
def run(seed,cond):
 W,ss,sc,Q,xs,ys=world(seed,cond);shared=ss.mean(0);mfit=np.array([np.sum(si*shared)/(np.sum(shared*shared)+1e-12) for si in ss])
 if cond=='shared_generator_orbit':shared=ss[0];mfit=sc
 qshared=np.array([cayley(v*shared) for v in mfit]);mat=np.array([q@W for q in Q]);methods=[]
 methods.append(('independent_oftv2',Q,[W,pack_skew(ss*sc[:,None,None])],None))
 methods.append(('shared_mirror',qshared,[W,pack_skew(shared),mfit],None))
 methods.append(('simple_scalar_generator',qshared,[W,pack_skew(shared),mfit],None))
 methods.append(('materialized_oft',Q,[mat],mat))
 for k in (2,4,8):
  qk=np.array([neumann(v*shared,k) for v in mfit]);methods.append((f'neumann_k{k}',qk,[W,pack_skew(shared),mfit,np.array(k)],k))
 out=[]
 for name,qpred,state,extra in methods:
  if name=='materialized_oft':
   fn=lambda: [x@mm for x,mm in zip(xs,extra)]
   preds=fn();ortho=None;cyc=None; mac=T*2048*D*D
  elif name.startswith('neumann_k'):
   k=extra;As=[v*shared for v in mfit]
   fn=lambda: [apply_neumann(x,a,k)@W for x,a in zip(xs,As)]
   preds=fn();ortho=max(float(np.linalg.norm(q.T@q-np.eye(D))) for q in qpred);cyc=max(float(np.linalg.norm((x@q)@q.T-x)/np.linalg.norm(x)) for x,q in zip(xs,qpred));mac=T*2048*(k+2)*D*D
  else:
   fn=lambda: [(x@q)@W for x,q in zip(xs,qpred)]
   preds=fn();ortho=max(float(np.linalg.norm(q.T@q-np.eye(D))) for q in qpred);cyc=max(float(np.linalg.norm((x@q)@q.T-x)/np.linalg.norm(x)) for x,q in zip(xs,qpred));mac=T*2048*2*D*D
  errs=[float(np.mean((yp-y)**2)) for yp,y in zip(preds,ys)];rel=np.sqrt(np.mean(errs))/(np.sqrt(np.mean([np.mean(y*y) for y in ys]))+1e-12)
  out.append({'seed':seed,'condition':cond,'method':name,'mse':float(np.mean(errs)),'relative_output_rmse':float(rel),'orthogonality_error':ortho,'inverse_cycle_error':cyc,'serialized_bytes':payload(state),'train_examples':T*512,'optimizer_updates':0,'active_compute_proxy':mac,'evaluation_wall_s':bench(fn,20),'transform_wall_s':bench(fn,60)})
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=[]
 for s in a.seeds:
  for c in ('shared_generator_orbit','independent_skew_stress'):r+=run(s,c)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'rows':len(r),'out':str(a.out)}))
if __name__=='__main__':main()
