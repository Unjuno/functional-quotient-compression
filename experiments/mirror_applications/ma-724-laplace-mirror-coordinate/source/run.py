from __future__ import annotations
import hashlib,io,time
import numpy as np
F,TRAIN,TEST=64,256,512
QS=(4,8,16,32);SAMPLES=128
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-40,40)))
def pack(a):
 b=io.BytesIO();np.savez_compressed(b,**a);v=b.getvalue();return len(v),hashlib.sha256(v).hexdigest()
def ece(prob,y,bins=10):
 conf=np.maximum(prob,1-prob);pred=(prob>=.5);acc=(pred==y);out=0.
 for i in range(bins):
  lo=i/bins;hi=(i+1)/bins;m=(conf>=lo)&(conf<(hi if i<bins-1 else hi+1e-8))
  if m.any():out+=m.mean()*abs(float(acc[m].mean())-float(conf[m].mean()))
 return out
def score(Phi,y,mu,cov,seed):
 rng=np.random.default_rng(seed);vals,U=np.linalg.eigh(cov);vals=np.maximum(vals,0);W=mu[:,None]+U@(np.sqrt(vals)[:,None]*rng.normal(size=(len(vals),SAMPLES)))
 ps=sigmoid(Phi@W).mean(axis=1);nll=-np.mean(y*np.log(np.maximum(ps,1e-8))+(1-y)*np.log(np.maximum(1-ps,1e-8)))
 return float(nll),float(np.mean((ps>=.5)==y)),float(ece(ps,y))
def run(seed=72401):
 rng=np.random.default_rng(seed);Xtr=rng.normal(size=(TRAIN,2));Xte=rng.normal(size=(TEST,2));Wf=rng.normal(size=(2,F));b=rng.uniform(0,2*np.pi,size=F);scale=1.5
 ph=lambda x:np.sqrt(2/F)*np.cos(scale*x@Wf+b)
 A=ph(Xtr);Z=ph(Xte);teacher=rng.normal(size=F);ptr=sigmoid(A@teacher);y=(rng.random(TRAIN)<ptr).astype(float);pte=sigmoid(Z@teacher);yt=(rng.random(TEST)<pte).astype(float)
 alpha=1.0;w=np.zeros(F);steps=0
 for _ in range(40):
  p=sigmoid(A@w);g=A.T@(p-y)+alpha*w;H=(A.T*(p*(1-p)))@A+alpha*np.eye(F);d=np.linalg.solve(H,g);w-=d;steps+=1
  if np.linalg.norm(d)<1e-8:break
 p=sigmoid(A@w);H=(A.T*(p*(1-p)))@A+alpha*np.eye(F);C=np.linalg.inv(H)
 states=[('full_Laplace',F,{'mean':w.astype(np.float32),'covariance':C.astype(np.float32)},C),('diagonal_Laplace',F,{'mean':w.astype(np.float32),'variance':np.diag(C).astype(np.float32)},np.diag(np.diag(C)))]
 eig,U=np.linalg.eigh(C);ix=np.argsort(eig)[::-1]
 for q in QS:
  B=U[:,ix[:q]];lam=np.maximum(eig[ix[:q]],1e-10);Cm=np.diag(lam)
  payload={'mean':w.astype(np.float32),'basis':B.astype(np.float32),'code_covariance':Cm.astype(np.float32)};approx=B@Cm@B.T
  states.append(('native_subspace_Laplace',q,payload,approx));states.append(('Mirror_code_Laplace',q,payload,approx))
 B=U[:,ix[:1]];Cm=np.diag(np.maximum(eig[ix[:1]],1e-10));states.append(('rank1_BNN',1,{'mean':w.astype(np.float32),'factor':(B@np.sqrt(Cm)).astype(np.float32)},B@Cm@B.T))
 rows=[]
 for name,q,payload,cov in states:
  size,digest=pack(payload);nll,acc,cal=score(Z,yt,w,cov,seed+100+q);rows.append({'seed':seed,'method':name,'rank':q,'serialized_bytes':size,'payload_sha256':digest,'nll':nll,'accuracy':acc,'ece':cal,'active_sample_work':int(TEST*F*q if name not in ('full_Laplace','diagonal_Laplace') else TEST*F*F),'newton_steps':steps,'wall_s':0.,'status_note':'fixed random Fourier features; synthetic nonlinear binary classification'})
 return rows
def save(rows,path):
 import csv
 with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=72401);p.add_argument('--out',default='RESULTS_CORE.csv');a=p.parse_args();t=time.perf_counter();rows=run(a.seed);dt=time.perf_counter()-t
 for x in rows:x['wall_s']=dt
 save(rows,a.out)
