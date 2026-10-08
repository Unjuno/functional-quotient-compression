#!/usr/bin/env python3
import argparse,io,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
H,O,T=12,6,8
def view(a):
 q=np.eye(H)
 for i,v in enumerate(a):
  c,s=np.cos(v),np.sin(v);q[2*i:2*i+2,2*i:2*i+2]=[[c,-s],[s,c]]
 return q
def world(seed,cond):
 g=np.random.default_rng(seed);W=g.normal(size=(H,O))*.5;base=g.uniform(.4,1.6,H);angs=g.uniform(-1,1,(T,H//2))
 scales=np.array([view(a)@base for a in angs]) if cond=='aligned_pair_rotation' else g.uniform(.3,1.8,(T,H))
 mats=np.array([np.diag(s)@W for s in scales]);tr=[];te=[]
 for m in mats:
  x=g.normal(size=(512,H));z=g.normal(size=(1024,H));tr.append((x,x@m));te.append((z,z@m))
 return W,tr,te
def fitm(tr):return np.array([np.linalg.lstsq(x,y,rcond=None)[0] for x,y in tr])
def gains(mats,W):return np.array([np.sum(m*W,axis=1)/(np.sum(W*W,axis=1)+1e-12) for m in mats])
def mirfit(sc):
 def dec(p):
  b=p[:H];a=p[H:].reshape(T,H//2);return np.array([view(v)@b for v in a])
 p=np.r_[sc.mean(0),np.zeros(T*H//2)];r=least_squares(lambda p:(dec(p)-sc).ravel(),p,max_nfev=500,ftol=1e-12,xtol=1e-12,gtol=1e-12)
 return dec(r.x),np.r_[r.x[:H],r.x[H:]],r.nfev
def payload(a):
 b=io.BytesIO();np.savez(b,**{f'p{i}':np.asarray(x) for i,x in enumerate(a)});return len(b.getvalue())
def run(seed,cond):
 W,tr,te=world(seed,cond);mats=fitm(tr);sc=gains(mats,W);mc,code,nfev=mirfit(sc);mean=sc.mean(0)
 methods=[('independent',mats,[mats]),('ia3',np.array([np.diag(v)@W for v in sc]),[W,sc]),('shared',np.tile(np.diag(mean)@W,(T,1,1)),[W,mean]),('mirror_rotation',np.array([np.diag(v)@W for v in mc]),[W,code])];out=[]
 for name,pred,state in methods:
  tm=time.perf_counter();e=[np.mean((x@m-y)**2) for m,(x,y) in zip(pred,te)];wall=time.perf_counter()-tm
  out.append({'seed':seed,'condition':cond,'method':name,'mse':float(np.mean(e)),'max_task_mse':float(max(e)),'serialized_bytes':payload(state),'train_examples':T*512,'optimizer_updates':0,'active_compute_proxy':T*1024*H*O,'evaluation_wall_s':wall,'mirror_nfev':int(nfev)})
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=[]
 for s in a.seeds:
  for c in ('aligned_pair_rotation','independent_random_gains'):r+=run(s,c)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'rows':len(r),'out':str(a.out)}))
if __name__=='__main__':main()
