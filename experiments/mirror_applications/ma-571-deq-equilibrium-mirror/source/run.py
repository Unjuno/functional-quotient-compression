#!/usr/bin/env python3
import argparse,csv,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SEEDS=(57101,57102);D=2;TASKS=8;TOL=1e-6;MAXIT=10000
def pack(a):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_STORED) as z:
  for k in sorted(a):
   q=io.BytesIO();np.save(q,a[k],allow_pickle=False);z.writestr(zipfile.ZipInfo(k+'.npy',(1980,1,1,0,0,0)),q.getvalue())
 return b.getvalue()
def rot(t):return np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
def solve(A,b):
 x=np.zeros_like(b)
 for i in range(1,MAXIT+1):
  xn=A@x+b
  if np.linalg.norm(xn-x)<=TOL:return xn,i
  x=xn
 return x,MAXIT
def run(seed):
 rng=np.random.default_rng(seed);Q,_=np.linalg.qr(rng.normal(size=(D,D)));A=Q@np.diag([.55,.32])@Q.T;anchor=np.array([1.,0.]);angles=np.linspace(0,2*np.pi,TASKS,endpoint=False);stars=np.stack([rot(t)@anchor for t in angles]);rows=[]
 states={'hard_tied':{'A':A.astype(np.float32),'anchor':anchor.astype(np.float32),'meta':np.frombuffer(b'hard-tied',dtype=np.uint8)},
 'mirror_angle':{'A':A.astype(np.float32),'anchor':anchor.astype(np.float32),'angles':angles.astype(np.float32),'meta':np.frombuffer(b'view-angle',dtype=np.uint8)},
 'direct_coeff':{'A':A.astype(np.float32),'anchor_basis':np.stack([anchor,np.array([0.,1.])]).astype(np.float32),'coefficients':np.stack([np.cos(angles),np.sin(angles)],axis=1).astype(np.float32),'meta':np.frombuffer(b'direct-coeff',dtype=np.uint8)},
 'independent':{'A_tasks':np.repeat(A[None,:,:],TASKS,axis=0).astype(np.float32),'fixed_points':stars.astype(np.float32),'meta':np.frombuffer(b'independent',dtype=np.uint8)}}
 # Equalize Mirror/direct naming metadata and state for a strict alias test.
 states['direct_coeff']=states['mirror_angle']
 for method,state in states.items():
  errs=[];its=[];stable=0;start=time.perf_counter()
  for t in range(TASKS):
   target_star=stars[t]
   forcing_star=anchor if method=='hard_tied' else target_star
   b=(np.eye(D)-A)@forcing_star;root,n=solve(A,b);its.append(n);stable+=int(n<MAXIT)
   errs.append(np.mean((root-target_star)**2)/max(np.mean(target_star**2),1e-12))
  wall=time.perf_counter()-start;rows.append({'world':seed,'method':method,'serialized_bytes':len(pack(state)),'train_examples':0,'optimizer_updates':0,'iterations_mean':f'{np.mean(its):.6g}','iterations_max':max(its),'operator_mac_proxy':int(sum(its)*D*D),'wall_time_s':f'{wall:.6f}','nmse':f'{np.mean(errs):.9g}','stable_tasks':stable,'status_note':'affine contractive fixed-point iteration'})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'artifacts'/'dev_results.csv'));a=ap.parse_args();r=[]
 for s in SEEDS:r+=run(s)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=r[0].keys());w.writeheader();w.writerows(r)
 print(json.dumps({'rows':len(r)},indent=2))
if __name__=='__main__':main()
