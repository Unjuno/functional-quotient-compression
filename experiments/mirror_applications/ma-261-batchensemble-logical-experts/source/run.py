#!/usr/bin/env python3
import argparse,io,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
D,O,N=8,6,4

def qmat(a):
 q=np.eye(D)
 for j,v in enumerate(a):
  c,s=np.cos(v),np.sin(v);q[2*j:2*j+2,2*j:2*j+2]=[[c,-s],[s,c]]
 return q
def data(seed,condition):
 g=np.random.default_rng(seed);base=g.normal(size=(D,O))*.6;ang=g.uniform(-.65,.65,(N,D//2))
 w=np.array([qmat(a)@base for a in ang]) if condition=='aligned_givens' else g.normal(size=(N,D,O))*.6
 tr=[];te=[]
 for wi in w:
  x=g.normal(size=(512,D));z=g.normal(size=(1024,D));tr.append((x,x@wi));te.append((z,z@wi))
 return w,tr,te
def fit(tr):return np.array([np.linalg.lstsq(x,y,rcond=None)[0] for x,y in tr])
def befit(w):
 def dec(p):
  i=D*O; a=p[:i].reshape(D,O);r=p[i:i+N*D].reshape(N,D);s=p[i+N*D:].reshape(N,O)
  return np.array([a*np.outer(r[k],s[k]) for k in range(N)])
 p=np.r_[w.mean(0).ravel(),np.ones(N*D+N*O)]
 z=least_squares(lambda v:(dec(v)-w).ravel(),p,max_nfev=120,ftol=1e-9,xtol=1e-9,gtol=1e-9)
 i=D*O;base=z.x[:i].reshape(D,O);r=z.x[i:i+N*D].reshape(N,D);v=z.x[i+N*D:].reshape(N,O)
 return dec(z.x),[base,r,v],z.nfev
def mirfit(w):
 def dec(p):
  a=p[:D*O].reshape(D,O);ang=p[D*O:].reshape(N,D//2)
  return np.array([qmat(v)@a for v in ang])
 p=np.r_[w.mean(0).ravel(),np.zeros(N*D//2)]
 z=least_squares(lambda v:(dec(v)-w).ravel(),p,max_nfev=120,ftol=1e-11,xtol=1e-11,gtol=1e-11)
 return dec(z.x),[z.x[:D*O].reshape(D,O),z.x[D*O:].reshape(N,D//2)],z.nfev
def nbytes(state):
 b=io.BytesIO();np.savez(b,**{f'p{i}':np.asarray(v) for i,v in enumerate(state)});return len(b.getvalue())
def run(seed,cond):
 _,tr,te=data(seed,cond);w=fit(tr);shared=w.mean(0);be,bes,bn=befit(w);mi,mis,mn=mirfit(w)
 meth=[('independent',w,[w]),('hard_tying',np.tile(shared,(N,1,1)),[shared]),('batchensemble_rank1',be,bes),('mirror_givens',mi,mis)];rows=[]
 for name,pred,state in meth:
  t=time.perf_counter();err=[np.mean((x@m-y)**2) for m,(x,y) in zip(pred,te)];elapsed=time.perf_counter()-t
  rows.append({'seed':seed,'condition':cond,'method':name,'mse':float(np.mean(err)),'max_role_mse':float(max(err)),'serialized_bytes':nbytes(state),'train_examples':N*512,'optimizer_updates':0,'active_compute_proxy':N*1024*D*O,'evaluation_wall_s':elapsed,'be_nfev':int(bn),'mirror_nfev':int(mn)})
 return rows
def main():
 a=argparse.ArgumentParser();a.add_argument('--seeds',nargs='+',type=int,required=True);a.add_argument('--out',type=Path,required=True);v=a.parse_args();r=[]
 for seed in v.seeds:
  for c in ('aligned_givens','independent_experts'):r+=run(seed,c)
 v.out.parent.mkdir(parents=True,exist_ok=True);v.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'rows':len(r),'out':str(v.out)}))
if __name__=='__main__':main()
