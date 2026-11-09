from __future__ import annotations
import hashlib,io,time
import numpy as np
D,M,PAIRS=64,4,[(2*i,2*i+1) for i in range(16)]
THRESHOLDS=(.10,.20,.30,.40)
def rotate(w,theta):
 z=w.copy()
 for j,(a,b) in enumerate(PAIRS):
  c=np.cos(theta[j]);s=np.sin(theta[j]);u=w[a];v=w[b];z[a]=c*u-s*v;z[b]=s*u+c*v
 return z
def pack(arr):
 b=io.BytesIO();np.savez_compressed(b,**arr);v=b.getvalue();return len(v),hashlib.sha256(v).hexdigest()
def probs_for(X,weights,k):return 1/(1+np.exp(-np.clip(2*X@weights[:k].T,-40,40)))
def adaptive(X,weights,tau,kind='disagreement'):
 n=len(X);sumprob=np.zeros(n);count=np.zeros(n,int);vals=np.zeros((n,M));active=np.ones(n,bool)
 for k in range(M):
  idx=np.where(active)[0]
  if len(idx)==0:break
  p=1/(1+np.exp(-np.clip(2*X[idx]@weights[k],-40,40)));vals[idx,k]=p;count[idx]+=1;sumprob[idx]+=p;mean=sumprob[idx]/count[idx]
  if kind=='confidence':stop=np.abs(mean-.5)>=tau
  else:
   std=np.array([np.std(vals[i,:count[i]]) for i in idx]);stop=(np.abs(mean-.5)>=tau)&(std<=tau/2)
  active[idx[stop]]=False
 mean=sumprob/np.maximum(count,1);std=np.array([np.std(vals[i,:count[i]]) for i in range(n)])
 selected=np.abs(mean-.5)>=tau
 return mean,count,selected,std
def metrics(p,y,coverage_mask=None):
 pred=p>=.5;acc=float(np.mean(pred==y));nll=float(-np.mean(y*np.log(np.maximum(p,1e-9))+(1-y)*np.log(np.maximum(1-p,1e-9))))
 if coverage_mask is None:coverage_mask=np.ones(len(y),bool)
 coverage=float(coverage_mask.mean());risk=float(np.mean(pred[coverage_mask]!=y[coverage_mask])) if coverage_mask.any() else 1.
 return coverage,risk,acc,nll
def run(seed=72901):
 rng=np.random.default_rng(seed);w=rng.normal(size=D);angles=rng.uniform(-np.pi,np.pi,size=(M,len(PAIRS)));weights=np.stack([rotate(w,a) for a in angles])
 X=rng.normal(size=(6144,D));scores=2*X@weights.T;y=(np.sum(scores>0,axis=1)>=3).astype(int);Xdev,ydev=X[:2048],y[:2048];Xte,yte=X[2048:],y[2048:]
 indep={'weights':weights.astype(np.float32),'metadata':np.array([D,M])};shared={'base':w.astype(np.float32),'angles':angles.astype(np.float32),'plane_pairs':np.array(PAIRS,dtype=np.int16),'metadata':np.array([D,M])}
 rows=[]
 def add(method,tau,state,p,count,selected,div):
  size,digest=pack(state);cov,risk,acc,nll=metrics(p,yte,selected);times=[]
  tX=Xte[:512]
  for x in tX:
   t0=time.perf_counter()
   if method.startswith('fixed'):
    k=int(method[-1]);scores=np.array([x@rotate(w,angles[j]) for j in range(k)]);pr=float(np.mean(1/(1+np.exp(-np.clip(2*scores,-40,40)))));cc=k
   elif method=='independent_full':
    pr=float(np.mean(1/(1+np.exp(-np.clip(2*x@weights.T,-40,40)))));cc=M
   elif method=='native_angle_bank':
    scores=np.array([x@rotate(w,angles[j]) for j in range(M)]);pr=float(np.mean(1/(1+np.exp(-np.clip(2*scores,-40,40)))));cc=M
   else:
    ps=[];total=0.;cc=0
    for j in range(M):
     pp=float(1/(1+np.exp(-np.clip(2*x@rotate(w,angles[j]),-40,40))));ps.append(pp);total+=pp;cc+=1;mean=total/cc
     sd=float(np.std(ps))
     stop=(abs(mean-.5)>=tau) if method=='adaptive_confidence' else ((abs(mean-.5)>=tau) and (sd<=tau/2))
     if stop:break
    pr=total/cc
   times.append((time.perf_counter()-t0)*1000)
  times=np.asarray(times);rows.append({'seed':seed,'method':method,'threshold':tau,'serialized_bytes':size,'payload_sha256':digest,'coverage':cov,'selective_risk':risk,'accuracy':acc,'nll':nll,'mean_active_views':float(np.mean(count)),'active_mac_proxy':int(np.mean(count)*(D+3*len(PAIRS))),'latency_p50_ms':float(np.percentile(times,50)),'latency_p95_ms':float(np.percentile(times,95)),'diversity':div,'status_note':'synthetic majority vote teacher; test set 4096; timings include view reconstruction'})
 # Fixed independent full bank and fixed/adaptive shared angle-coded banks.
 p4=probs_for(Xte,weights,4).mean(axis=1);div=float(np.mean(np.std(probs_for(Xte,weights,4),axis=1)))
 add('independent_full',1.,indep,p4,np.full(len(yte),4),np.ones(len(yte),bool),div)
 add('native_angle_bank',1.,shared,p4,np.full(len(yte),4),np.ones(len(yte),bool),div)
 for k in (1,2,4):
  p=probs_for(Xte,weights,k).mean(axis=1);sel=np.ones(len(yte),bool);state={'base':w.astype(np.float32),'angles':angles[:k].astype(np.float32),'plane_pairs':np.array(PAIRS,dtype=np.int16),'metadata':np.array([D,k])};add(f'fixed{k}',1.,state,p,np.full(len(yte),k),sel,div)
 for tau in THRESHOLDS:
  for kind,name in [('disagreement','adaptive_disagreement'),('confidence','adaptive_confidence')]:
   p,c,sel,_=adaptive(Xte,weights,tau,kind);add(name,tau,{**shared,'policy':np.array([tau],np.float32)},p,c,sel,div)
 return rows
def save(rows,path):
 import csv
 with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=72901);p.add_argument('--out',default='RESULTS_CORE.csv');a=p.parse_args();save(run(a.seed),a.out)
