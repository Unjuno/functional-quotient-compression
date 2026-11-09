from __future__ import annotations
import hashlib,io,time
import numpy as np
P,NTR,NTE=64,128,512
RANKS=(1,2,4,8,16)
def pack(a):
 b=io.BytesIO();np.savez_compressed(b,**a);v=b.getvalue();return len(v),hashlib.sha256(v).hexdigest()
def _metrics(X,y,mu,cov):
 m=X@mu;v=np.einsum('bi,ij,bj->b',X,cov,X)+.2**2;v=np.maximum(v,1e-10)
 nll=float(np.mean(.5*np.log(2*np.pi*v)+.5*(y-m)**2/v));covered=np.abs(y-m)<=1.6448536269514722*np.sqrt(v)
 return nll,float(np.mean(covered))
def run(seed=72101):
 rng=np.random.default_rng(seed);Q,_=np.linalg.qr(rng.normal(size=(P,8)));ev=np.linspace(.3,.03,8);Sigma=(Q*ev)@Q.T+np.diag(np.full(P,.01));mu=rng.normal(size=P)*.1
 # Snapshot estimate mimics a SWAG trajectory around a posterior mean.
 snaps=rng.multivariate_normal(mu,Sigma,size=64);smean=snaps.mean(0);center=snaps-smean;cov=np.cov(snaps,rowvar=False,bias=True)
 Xtr=rng.normal(size=(NTR,P));ytr=Xtr@mu+rng.normal(scale=.2,size=NTR)
 X=rng.normal(size=(NTE,P));wtrue=rng.multivariate_normal(mu,Sigma,size=NTE);y=np.einsum('bi,bi->b',X,wtrue)+rng.normal(scale=.2,size=NTE)
 methods=[]
 methods.append(('dense_posterior',P,{'mean':mu.astype(np.float32),'covariance':Sigma.astype(np.float32)},mu,Sigma))
 for k in RANKS:
  vals,U=np.linalg.eigh(cov);ix=np.argsort(vals)[::-1][:k];lam=np.maximum(vals[ix],0);B=U[:,ix];low=(B*lam)@B.T;diag=np.maximum(np.diag(cov-low),1e-8);approx=low+np.diag(diag)
  # SWAG stores mean, low-rank factor and residual diagonal; Mirror is same posterior address map m~N(0,I).
  state={'mean':smean.astype(np.float32),'basis':B.astype(np.float32),'eigenvalues':lam.astype(np.float32),'diagonal':diag.astype(np.float32)}
  methods.append(('SWAG_lowrank_diag',k,state,smean,approx));methods.append(('Mirror_posterior_coordinate',k,state,smean,approx))
 # Rank-1 Bayesian control: one global covariance direction with diagonal residual.
 vals,U=np.linalg.eigh(cov);ix=np.argsort(vals)[::-1][:1];lam=np.maximum(vals[ix],0);B=U[:,ix];low=(B*lam)@B.T;diag=np.maximum(np.diag(cov-low),1e-8);approx=low+np.diag(diag)
 methods.append(('rank1_BNN',1,{'mean':smean.astype(np.float32),'factor':(B*np.sqrt(lam)).astype(np.float32),'diagonal':diag.astype(np.float32)},smean,approx))
 rows=[]
 for name,k,state,m,c in methods:
  size,digest=pack(state);nll,cov90=_metrics(X,y,m,c);rows.append({'seed':seed,'method':name,'rank':k,'serialized_bytes':size,'payload_sha256':digest,'predictive_nll':nll,'coverage90':cov90,'active_sample_work':int(NTE*(P*k if name!='dense_posterior' else P*P)),'wall_s':0.,'status_note':'synthetic Gaussian regression; 64 posterior snapshots; fixed test set'})
 return rows
def save(rows,path):
 import csv
 with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=72101);p.add_argument('--out',default='RESULTS_CORE.csv');a=p.parse_args();t=time.perf_counter();rows=run(a.seed);dt=time.perf_counter()-t
 for x in rows:x['wall_s']=dt
 save(rows,a.out)
