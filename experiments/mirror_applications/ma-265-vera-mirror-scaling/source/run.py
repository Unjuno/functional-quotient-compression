#!/usr/bin/env python3
import argparse,io,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
D,O,R,T=16,12,4,8

def world(seed,cond):
 g=np.random.default_rng(seed);base=g.normal(size=(D,O))*.2;B=g.normal(size=(D,R))/np.sqrt(D);A=g.normal(size=(R,O))/np.sqrt(R)
 code=g.normal(size=R)
 if cond=='aligned_scale_orbit':
  angles=np.linspace(-1.1,1.1,T);codes=np.tile(code,(T,1));
  for i,a in enumerate(angles):c,s=np.cos(a),np.sin(a);codes[i,:2]=[c*code[0]-s*code[1],s*code[0]+c*code[1]]
 else:codes=g.normal(size=(T,R))
 mats=np.array([base+B@np.diag(v)@A for v in codes]);tr=[];te=[]
 for w in mats:
  x=g.normal(size=(512,D));z=g.normal(size=(1024,D));tr.append((x,x@w));te.append((z,z@w))
 return base,B,A,codes,tr,te

def fit_mats(tr):return np.array([np.linalg.lstsq(x,y,rcond=None)[0] for x,y in tr])
def fit_codes(mats,base,B,A):
 atoms=np.array([np.outer(B[:,k],A[k,:]).ravel() for k in range(R)]).T
 return np.array([np.linalg.lstsq(atoms,(w-base).ravel(),rcond=None)[0] for w in mats])
def mirror_codes(codes):
 def dec(p):
  b=p[:R];ang=p[R:];out=[]
  for v in ang:
   z=b.copy();c,s=np.cos(v),np.sin(v);z[:2]=[c*b[0]-s*b[1],s*b[0]+c*b[1]];out.append(z)
  return np.array(out)
 p=np.r_[codes.mean(0),np.linspace(-.7,.7,T)]
 z=least_squares(lambda v:(dec(v)-codes).ravel(),p,max_nfev=500,ftol=1e-12,xtol=1e-12,gtol=1e-12)
 return dec(z.x),[z.x[:R],z.x[R:]],int(z.nfev)
def lowrank_delta(mats,base):
 ls=[];rs=[]
 for w in mats:
  u,s,v=np.linalg.svd(w-base,full_matrices=False);ls.append(u[:,:R]*s[:R]);rs.append(v[:R])
 return np.array(ls),np.array(rs)
def payload(st):
 f=io.BytesIO();np.savez(f,**{f'p{i}':np.asarray(v) for i,v in enumerate(st)});return len(f.getvalue())
def mse(mats,te):return [float(np.mean((x@m-y)**2)) for m,(x,y) in zip(mats,te)]
def run(seed,cond):
 base,B,A,true,tr,te=world(seed,cond);mats=fit_mats(tr);codes=fit_codes(mats,base,B,A);mir,ms,nfev=mirror_codes(codes);mean=codes.mean(0);lv,rv=lowrank_delta(mats,base)
 methods=[('independent_dense',mats,[mats]),('vera',np.array([base+B@np.diag(c)@A for c in codes]),[base,B,A,codes]),('hard_tied',np.tile(base+B@np.diag(mean)@A,(T,1,1)),[base,B,A,mean]),('mirror_view',np.array([base+B@np.diag(c)@A for c in mir]),[base,B,A,np.concatenate([ms[0],ms[1]])]),('lora_rank4',np.array([base+lv[i]@rv[i] for i in range(T)]),[base,lv,rv])]
 out=[]
 for name,pred,state in methods:
  t=time.perf_counter();err=mse(pred,te);wall=time.perf_counter()-t
  rankmac=(D*R+R*O);mac=1024*T*(D*O if name=='independent_dense' else rankmac)
  out.append({'seed':seed,'condition':cond,'method':name,'mse':float(np.mean(err)),'max_task_mse':float(max(err)),'serialized_bytes':payload(state),'train_examples':T*512,'optimizer_updates':0,'active_compute_proxy':mac,'evaluation_wall_s':wall,'mirror_nfev':nfev})
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=[]
 for s in a.seeds:
  for c in ('aligned_scale_orbit','independent_scale_codes'):r+=run(s,c)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'rows':len(r),'out':str(a.out)}))
if __name__=='__main__':main()
