#!/usr/bin/env python3
import argparse,io,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
D,T,S=8,6,3;K=D//2*S
def boft(a):
 q=np.eye(D);n=0
 for stage in range(S):
  step=1<<stage;z=np.eye(D)
  for i in range(D):
   if i&step:continue
   j=i|step;c,s=np.cos(a[n]),np.sin(a[n]);z[np.ix_([i,j],[i,j])]=[[c,-s],[s,c]];n+=1
  q=z@q
 return q
def world(seed,cond):
 g=np.random.default_rng(seed);W=g.normal(size=(D,D))*.4;m=np.linspace(-1,1,T)
 if cond=='aligned_angle_orbit':phi=g.uniform(-.35,.35,K);a=m[:,None]*phi
 else:a=g.uniform(-.3,.3,(T,K))
 Q=np.array([boft(v) for v in a]);tr=[];te=[]
 for q in Q:
  x=g.normal(size=(512,D));z=g.normal(size=(1024,D));tr.append((x,(x@q)@W));te.append((z,(z@q)@W))
 return W,Q,a,tr,te
def fitmaps(tr):return np.array([np.linalg.lstsq(x,y,rcond=None)[0] for x,y in tr])
def fit_angles(W,maps):
 out=[]
 for M in maps:
  r=least_squares(lambda a:(boft(a)@W-M).ravel(),np.zeros(K),max_nfev=300,ftol=1e-11,xtol=1e-11,gtol=1e-11);out.append(r.x)
 return np.array(out)
def fit_factor(W,maps,angles):
 def dec(p):
  phi=p[:K];m=p[K:];return np.array([boft(v*phi)@W for v in m])
 p=np.r_[angles.mean(0),np.linspace(-.8,.8,T)];r=least_squares(lambda p:(dec(p)-maps).ravel(),p,max_nfev=500,ftol=1e-11,xtol=1e-11,gtol=1e-11);return r.x[:K],r.x[K:],dec(r.x)
def payload(a):
 b=io.BytesIO();np.savez(b,**{f'p{i}':np.asarray(x) for i,x in enumerate(a)});return len(b.getvalue())
def bench(fn,reps=80):
 t=time.perf_counter()
 for _ in range(reps):fn()
 return (time.perf_counter()-t)/reps
def run(seed,cond):
 W,Q,a,tr,te=world(seed,cond);maps=fitmaps(tr);ang=fit_angles(W,maps);phi,m,fac=fit_factor(W,maps,ang)
 shared=ang.mean(0);preds=[('independent_boft',np.array([boft(v)@W for v in ang]),[W,ang]),('shared_boft',np.tile(boft(shared)@W,(T,1,1)),[W,shared]),('factorized_control',fac,[W,np.r_[phi,m]]),('mirror_view',fac,[W,np.r_[phi,m]]),('independent_dense',maps,[maps])]
 rows=[]
 for name,mats,state in preds:
  err=[np.mean((x@M-y)**2) for M,(x,y) in zip(mats,te)];ortho=None;cycle=None
  if name!='independent_dense':
   qarr=np.array([boft(v) for v in ang]) if name=='independent_boft' else (np.array([boft(v) for v in np.tile(shared,(T,1))]) if name=='shared_boft' else np.array([boft(v*phi) for v in m]))
   ortho=max(float(np.linalg.norm(q.T@q-np.eye(D))) for q in qarr);x0=np.ones((8,D));cycle=max(float(np.max(np.abs((x0@q)@q.T-x0))) for q in qarr)
   fn=lambda:[(x@q)@W for x,q in zip([z for z,y in te],qarr)]
  else:fn=lambda:[x@M for M,(x,y) in zip(mats,te)]
  rows.append({'seed':seed,'condition':cond,'method':name,'mse':float(np.mean(err)),'max_task_mse':float(max(err)),'serialized_bytes':payload(state),'orthogonality_error':ortho,'inverse_cycle_error':cycle,'train_examples':T*512,'optimizer_updates':0,'active_compute_proxy':T*1024*D*D*(S if name!='independent_dense' else 1),'transform_wall_s':bench(fn)})
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=[]
 for s in a.seeds:
  for c in ('aligned_angle_orbit','independent_angle_stress'):r+=run(s,c)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'rows':len(r),'out':str(a.out)}))
if __name__=='__main__':main()
