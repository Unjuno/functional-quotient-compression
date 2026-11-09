from __future__ import annotations
import hashlib,io,time
import numpy as np
P,TR,NT,R=64,64,512,16
QS=(2,4,8,12,16)
def pack(a):
 b=io.BytesIO();np.savez_compressed(b,**a);v=b.getvalue();return len(v),hashlib.sha256(v).hexdigest()
def metrics(X,y,mu,cov):
 m=X@mu;v=np.maximum(np.einsum('bi,ij,bj->b',X,cov,X)+.2**2,1e-10);return float(np.mean(.5*np.log(2*np.pi*v)+.5*(y-m)**2/v)),float(np.mean(np.abs(y-m)<=1.64485362695*np.sqrt(v)))
def run(seed=72201):
 rng=np.random.default_rng(seed);Q,_=np.linalg.qr(rng.normal(size=(P,R)));ev=np.geomspace(.3,.015,R);truecov=(Q*ev)@Q.T+np.diag(np.full(P,.008));mu=rng.normal(size=P)*.08
 snaps=rng.multivariate_normal(mu,truecov,size=TR);mhat=snaps.mean(0);cov=np.cov(snaps,rowvar=False,bias=True);vals,U=np.linalg.eigh(cov);ix=np.argsort(vals)[::-1][:R];lam=np.maximum(vals[ix],0);B=U[:,ix];full=(B*lam)@B.T+np.diag(np.maximum(np.diag(cov-(B*lam)@B.T),1e-9))
 X=rng.normal(size=(NT,P));wt=rng.multivariate_normal(mu,truecov,size=NT);y=np.einsum('bi,bi->b',X,wt)+rng.normal(scale=.2,size=NT)
 rows=[]
 states=[('dense_cov',R,{'mean':mhat.astype(np.float32),'cov':cov.astype(np.float32)},cov),('full_SWAG_basis',R,{'mean':mhat.astype(np.float32),'basis':B.astype(np.float32),'eigenvalues':lam.astype(np.float32),'diagonal':np.maximum(np.diag(cov-(B*lam)@B.T),1e-9).astype(np.float32)},full)]
 for q in QS:
  a,s,vt=np.linalg.svd(B,full_matrices=False);A=a[:,:q]*s[:q];C=vt[:q,:];Bq=A@C
  low=(Bq*lam)@Bq.T;diag=np.maximum(np.diag(cov-low),1e-9);approx=low+np.diag(diag)
  state={'mean':mhat.astype(np.float32),'left_factor':A.astype(np.float32),'right_factor':C.astype(np.float32),'eigenvalues':lam.astype(np.float32),'diagonal':diag.astype(np.float32)}
  states.extend([('native_SVD_factor',q,state,approx),('Mirror_factor_code',q,state,approx)])
 for name,q,state,c in states:
  size,digest=pack(state);nll,covg=metrics(X,y,mhat,c);rows.append({'seed':seed,'method':name,'factor_rank':q,'serialized_bytes':size,'payload_sha256':digest,'predictive_nll':nll,'coverage90':covg,'covariance_frobenius_error':float(np.linalg.norm(c-cov)/(np.linalg.norm(cov)+1e-12)),'active_sample_work':int(NT*P*q if name not in ('dense_cov','full_SWAG_basis') else NT*P*P),'wall_s':0.,'status_note':'synthetic Gaussian regression; common generated evaluation set'})
 return rows
def save(rows,path):
 import csv
 with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=72201);p.add_argument('--out',default='RESULTS_CORE.csv');a=p.parse_args();t=time.perf_counter();rows=run(a.seed);dt=time.perf_counter()-t
 for x in rows:x['wall_s']=dt
 save(rows,a.out)
