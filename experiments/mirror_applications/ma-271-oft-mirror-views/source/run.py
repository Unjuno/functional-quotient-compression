#!/usr/bin/env python3
import argparse,io,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
D,T,P=8,6,4
def qmat(a):
 q=np.eye(D)
 for j,v in enumerate(a):
  c,s=np.cos(v),np.sin(v);q[2*j:2*j+2,2*j:2*j+2]=[[c,-s],[s,c]]
 return q
def world(seed,aligned):
 g=np.random.default_rng(seed);w=g.normal(size=(D,D))*.5
 if aligned:
  phi=g.uniform(-.6,.6,P);m=np.linspace(-1,1,T);a=m[:,None]*phi
 else:a=g.uniform(-.7,.7,(T,P))
 Q=np.array([qmat(v) for v in a]);tr=[];te=[]
 for q in Q:
  x=g.normal(size=(512,D));z=g.normal(size=(1024,D));tr.append((x,(x@q)@w));te.append((z,(z@q)@w))
 return w,Q,a,tr,te
def fitmaps(tr):return np.array([np.linalg.lstsq(x,y,rcond=None)[0] for x,y in tr])
def fit_block(w,maps):
 arr=[]
 for target in maps:
  r=least_squares(lambda a:(qmat(a)@w-target).ravel(),np.zeros(P),max_nfev=300,ftol=1e-12,xtol=1e-12,gtol=1e-12);arr.append(r.x)
 return np.array(arr)
def fit_rank1(w,maps,angles):
 def dec(p):
  phi=p[:P];m=p[P:];a=m[:,None]*phi;return np.array([qmat(v)@w for v in a])
 p=np.r_[angles.mean(0),np.linspace(-.8,.8,T)]
 r=least_squares(lambda p:(dec(p)-maps).ravel(),p,max_nfev=500,ftol=1e-12,xtol=1e-12,gtol=1e-12);return r.x[:P],r.x[P:],dec(r.x)
def payload(st):
 b=io.BytesIO();np.savez(b,**{f'p{i}':np.asarray(x) for i,x in enumerate(st)});return len(b.getvalue())
def evaluate(pred,te):return [float(np.mean((x@m-y)**2)) for m,(x,y) in zip(pred,te)]
def ortho(qs):return max(float(np.linalg.norm(q.T@q-np.eye(D))) for q in qs)
def timed(qs,w):
 x=np.ones((32,D));start=time.perf_counter()
 for _ in range(400):
  for q in qs:_=(x@q)@w
 return (time.perf_counter()-start)/400

def run(seed,cond):
 w,qtrue,a,tr,te=world(seed,cond=='aligned_orbit');maps=fitmaps(tr);block=fit_block(w,maps);phi,m,mp=fit_rank1(w,maps,block)
 full=[]
 for target in maps:
  u,s,v=np.linalg.svd(target@np.linalg.pinv(w),full_matrices=False);full.append(u@v)
 full=np.array(full);preds=[('oft_dense',np.array([q@w for q in full]),[w,full],full),('oft_blockwise',np.array([qmat(v)@w for v in block]),[w,block],np.array([qmat(v) for v in block])),('simple_rank1_angle',mp,[w,np.r_[phi,m]],np.array([qmat(v) for v in m[:,None]*phi])),('mirror_view',mp,[w,np.r_[phi,m]],[qmat(v) for v in m[:,None]*phi])]
 rows=[]
 for name,pred,state,qs in preds:
  tm=time.perf_counter();err=evaluate(pred,te);wall=time.perf_counter()-tm
  xx=np.ones((8,D));cyc=max(float(np.max(np.abs((xx@q)@q.T-xx))) for q in qs)
  rows.append({'seed':seed,'condition':cond,'method':name,'mse':float(np.mean(err)),'max_task_mse':max(err),'serialized_bytes':payload(state),'orthogonality_error':ortho(qs),'inverse_cycle_error':cyc,'train_examples':T*512,'optimizer_updates':0,'active_compute_proxy':T*1024*D*D,'evaluation_wall_s':wall,'transform_wall_s':timed(qs,w)})
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=[]
 for s in a.seeds:
  for c in ('aligned_orbit','independent_plane_angles'):r+=run(s,c)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'rows':len(r),'out':str(a.out)}))
if __name__=='__main__':main()
