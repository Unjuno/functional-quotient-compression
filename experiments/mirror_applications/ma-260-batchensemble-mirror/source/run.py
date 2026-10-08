#!/usr/bin/env python3
import argparse,csv,io,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,least_squares

D=8; T=6; ROOT=Path(__file__).resolve().parents[1]
def rotation(a):
 q=np.eye(D); c,s=np.cos(a),np.sin(a); q[:2,:2]=[[c,-s],[s,c]]; return q

def make_world(seed,condition):
 rng=np.random.default_rng(seed)
 base=rng.normal(size=D); base/=np.linalg.norm(base); base*=3.0
 if condition=='rotation-aligned': angles=np.linspace(-1.05,1.05,T); ws=np.array([rotation(a)@base for a in angles])
 else: angles=None; ws=rng.normal(size=(T,D)); ws=3*ws/np.linalg.norm(ws,axis=1)[:,None]
 train=[]; test=[]
 for w in ws:
  sets=[]
  for n in (512,1024):
   x=rng.normal(size=(n,D)); logits=x@w; p=1/(1+np.exp(-logits)); y=rng.binomial(1,p)
   sets.append((x,y))
  train.append(sets[0]);test.append(sets[1])
 return np.array(ws),train,test

def fit_logistic(x,y):
 def fun(p):
  z=x@p[:-1]+p[-1]; prob=1/(1+np.exp(-np.clip(z,-40,40)))
  loss=np.mean(np.logaddexp(0,z)-y*z)+1e-4*np.sum(p[:-1]**2)
  grad=np.r_[x.T@(prob-y)/len(y)+2e-4*p[:-1],np.mean(prob-y)]
  return loss,grad
 t=time.perf_counter();res=minimize(fun,np.zeros(D+1),jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-12});return res.x[:D],float(res.x[-1]),int(res.nit),time.perf_counter()-t

def ece(p,y,bins=10):
 conf=np.maximum(p,1-p); pred=(p>=.5); correct=(pred==y); out=0.
 for lo in np.linspace(0,1,bins+1)[:-1]:
  ix=(conf>=lo)&(conf<lo+1/bins)
  if ix.any():out+=ix.mean()*abs(correct[ix].mean()-conf[ix].mean())
 return float(out)
def metrics(ws,bs,test):
 acc=[]; cal=[]; preds=[]
 for w,b,(x,y) in zip(ws,bs,test):
  p=1/(1+np.exp(-np.clip(x@w+b,-40,40))); acc.append(np.mean((p>=.5)==y));cal.append(ece(p,y));preds.append(p>=.5)
 dis=[np.mean(preds[i]!=preds[j]) for i in range(T) for j in range(i+1,T)]
 return float(np.mean(acc)),float(np.mean(cal)),float(np.mean(dis))
def payload(states):
 f=io.BytesIO();np.savez(f,**{f'x{i}':np.asarray(x) for i,x in enumerate(states)});return len(f.getvalue())
def encode_mirror(fitted,condition):
 W=np.array([x[0] for x in fitted]); b=np.array([x[1] for x in fitted])
 if condition=='rotation-aligned':
  def unpack(p):
   base=p[:D]; angles=p[D:]
   return np.array([rotation(a)@base for a in angles])
  p0=np.r_[W.mean(axis=0),np.linspace(-.7,.7,T)]
  res=least_squares(lambda p:(unpack(p)-W).ravel(),p0,max_nfev=600,xtol=1e-12,ftol=1e-12,gtol=1e-12)
  pred=unpack(res.x); state=[res.x[:D],res.x[D:],b]
 else:
  pred=np.tile(W.mean(axis=0),(T,1)); state=[W.mean(axis=0),np.zeros(T),b]
 return [(w,float(bb)) for w,bb in zip(pred,b)],state

def run(seed,condition):
 teachers,tr,te=make_world(seed,condition); fitted=[];updates=[];trainsec=0.
 for x,y in tr:
  w,b,n,sec=fit_logistic(x,y);fitted.append((w,b));updates.append(n);trainsec+=sec
 W=np.array([a for a,b in fitted]); B=np.array([b for a,b in fitted])
 bebase=W.mean(axis=0); factors=W/(bebase[None,:]+1e-12); bepred=[(bebase*r,float(b)) for r,b in zip(factors,B)]
 mirpred,mirstate=encode_mirror(fitted,condition)
 methods=[('shared',[(W.mean(axis=0),float(B.mean()))]*T,[W.mean(axis=0),np.array([B.mean()])]),('independent',fitted,[W,B]),('batchensemble_rank1',bepred,[bebase,factors,B]),('mirror_rotation',mirpred,mirstate)]
 rows=[]
 for name,model,state in methods:
  a,c,d=metrics([w for w,b in model],[b for w,b in model],te)
  rows.append({'method':name,'accuracy':a,'ece':c,'member_disagreement':d,'serialized_bytes':payload(state),'train_examples':T*512,'optimizer_updates':sum(updates),'training_wall_s':trainsec,'inference_mac_proxy':T*1024*D})
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);ap.add_argument('--split',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();rows=[]
 for seed in a.seeds:
  for condition in ('rotation-aligned','independent-random'):
   for r in run(seed,condition):r.update(seed=seed,condition=condition);rows.append(r)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({'rows':len(rows),'out':str(a.out)}))
if __name__=='__main__':main()
