import argparse,csv,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
D=12; FIT=[1,4]; HELD=[2,3]; ALL=[1,2,3,4]
def theta(d): return -.18+.09*d
def giv(h,t):
 z=h.copy(); a=z[...,0::2].copy(); b=z[...,1::2].copy(); co=np.cos(t); si=np.sin(t); z[...,0::2]=co*a-si*b; z[...,1::2]=si*a+co*b; return z
def run(seed,split,root):
 rng=np.random.default_rng(seed); A=rng.normal(0,.3,(D,D)); Bs=[np.eye(D)+rng.normal(0,.07,(D,D)) for _ in range(4)]; C=rng.normal(0,.3,(D,D)); X={d:rng.normal(size=(256,D)) for d in ALL}
 def h(x,d):
  z=np.tanh(x@A)
  for b in Bs[:d]: z=np.tanh(.85*z@b+.15*z)
  return z
 Y={d:giv(h(X[d],d),theta(d))@C for d in ALL}
 def mr(q): return np.concatenate([(giv(h(X[d],d),q[0]+q[1]*d)@C-Y[d]).ravel() for d in FIT])
 start=time.perf_counter(); q=least_squares(mr,[0.,0.]).x; fit=time.perf_counter()-start
 # Static rank-one correction: fixed input and output vectors with depth-specific scalar values fitted on training depths.
 u=rng.normal(size=D); v=rng.normal(size=D); u/=np.linalg.norm(u); v/=np.linalg.norm(v)
 amps=[]
 for d in FIT:
  delta=Y[d]-h(X[d],d)@C; amps.append(np.sum(delta*np.outer(X[d]@u,v))/np.sum(np.square(np.outer(X[d]@u,v))))
 ampfit=np.polyfit(FIT,amps,1)
 full={d:giv(np.eye(D),theta(d))@C for d in ALL}
 out=Path(root)/split/str(seed); out.mkdir(parents=True,exist_ok=True)
 def save(name,**kw):
  p=out/f'{name}.npz'; np.savez_compressed(p,**kw); return p.stat().st_size
 sb=save('shared',A=A,B=np.stack(Bs),C=C); mb=save('mirror',q=q); rb=save('rank1',u=u,v=v,amp=ampfit); fb=save('independent',**{f'M{d}':full[d] for d in ALL})
 rows=[]
 for d in HELD:
  x=X[d]; hd=h(x,d); y=Y[d]
  ps={'native':hd@C,'mirror':giv(hd,q[0]+q[1]*d)@C,'rank1':hd@C+(ampfit[0]*d+ampfit[1])*np.outer(x@u,v),'independent':hd@full[d]}
  for m,z in ps.items(): rows.append({'seed':seed,'split':split,'depth':d,'method':m,'nrmse':float(np.sqrt(np.mean((z-y)**2)/(np.mean(y*y)+1e-20))),'payload_bytes':sb+{'native':0,'mirror':mb,'rank1':rb,'independent':fb}[m],'shared_bytes':sb,'method_bytes':{'native':0,'mirror':mb,'rank1':rb,'independent':fb}[m],'fit_seconds':fit})
 hashes={k:hashlib.sha256((out/f'{k}.npz').read_bytes()).hexdigest() for k in ['shared','mirror','rank1','independent']}
 (out/'metrics.json').write_text(json.dumps({'seed':seed,'split':split,'bytes':{'shared':sb,'mirror':mb,'rank1':rb,'independent':fb},'hashes':hashes,'fit_seconds':fit},indent=2)+'\n')
 with (out/'RESULTS_CORE.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 return rows
def main():
 p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--split',choices=['dev','fresh'],required=True); p.add_argument('--seeds',nargs='+',type=int,required=True); a=p.parse_args(); rows=[]
 for seed in a.seeds: rows+=run(seed,a.split,a.out)
 q=Path(a.out)/a.split/'RESULTS_CORE.csv'; q.parent.mkdir(parents=True,exist_ok=True)
 with q.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if __name__=='__main__':main()
