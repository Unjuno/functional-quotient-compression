import argparse,csv,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
TRAIN=[(4,1),(4,2),(8,1),(8,3),(12,2)]; HELD=[(4,3),(8,2),(12,1),(12,3)]; ALL=TRAIN+HELD; D=12
AL=np.array([.13,-.19,.24]); BE=np.array([-.11,.17,.06])
def th(c): return AL[[4,8,12].index(c[0])]+BE[c[1]-1]
def giv(h,t):
 z=h.copy(); a=z[...,0::2].copy(); b=z[...,1::2].copy(); co=np.cos(t); si=np.sin(t); z[...,0::2]=co*a-si*b; z[...,1::2]=si*a+co*b; return z
def run(seed,split,root):
 rng=np.random.default_rng(seed); A=rng.normal(0,.35,(D,D)); B=np.eye(D)+rng.normal(0,.08,(D,D)); C=rng.normal(0,.28,(D,D)); X={c:rng.normal(size=(256,D)) for c in ALL}
 def hidden(x,c):
  w,d=c; h=np.tanh(x@A[:,:w])
  for _ in range(d): h=np.tanh(.9*h@B[:w,:w]+.1*h)
  return h
 def target(x,c): return giv(hidden(x,c),th(c))@C[:c[0],:]
 Y={c:target(X[c],c) for c in ALL}
 def basis(c): return np.eye(5)[[ (4,1),(4,2),(8,1),(8,3),(12,2)].index(c)] if c in TRAIN else np.zeros(5)
 # Fit row and column factors jointly, using train cells only.
 def f(z,c): return z[[4,8,12].index(c[0])]+z[3+c[1]-1]
 def residual(z): return np.concatenate([(giv(hidden(X[c],c),f(z,c))@C[:c[0],:]-Y[c]).ravel() for c in TRAIN])
 t=time.perf_counter(); opt=least_squares(residual,np.zeros(6),max_nfev=100); fit_time=time.perf_counter()-t; q=opt.x
 # Generic direct coefficient control uses the identical additive scalar function class.
 direct=q.copy(); full={c:giv(np.eye(c[0]),th(c))@C[:c[0],:] for c in ALL}
 out=Path(root)/split/str(seed); out.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(out/'shared.npz',A=A,B=B,C=C); sb=(out/'shared.npz').stat().st_size
 def save(name,**kw):
  p=out/f'{name}.npz'; np.savez_compressed(p,**kw); return p.stat().st_size
 mb=save('mirror',alpha=q[:3],beta=q[3:]); db=save('direct',alpha=direct[:3],beta=direct[3:]); fb=save('independent',**{f'M{i}':full[c] for i,c in enumerate(ALL)})
 rows=[]
 for c in HELD:
  w,d=c; x=X[c]; h=hidden(x,c); ym=giv(h,f(q,c))@C[:w,:]; yd=giv(h,f(direct,c))@C[:w,:]; yn=h@C[:w,:]; yi=h@full[c]; y=Y[c]
  for m,p,b in [('native',yn,0),('mirror',ym,mb),('direct',yd,db),('independent',yi,fb)]:
   nrmse=np.sqrt(np.mean((p-y)**2)/(np.mean(y*y)+1e-20)); rows.append({'seed':seed,'split':split,'width':w,'depth':d,'method':m,'nrmse':nrmse,'payload_bytes':sb+b,'shared_bytes':sb,'method_bytes':b,'fit_seconds':fit_time if m in ('mirror','direct') else 0})
 hashes={k:hashlib.sha256((out/f'{k}.npz').read_bytes()).hexdigest() for k in ('shared','mirror','direct','independent')}
 (out/'metrics.json').write_text(json.dumps({'seed':seed,'split':split,'bytes':{'shared':sb,'mirror':mb,'direct':db,'independent':fb},'hashes':hashes,'fit_seconds':fit_time},indent=2)+'\n')
 with (out/'RESULTS_CORE.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 return rows
def main():
 a=argparse.ArgumentParser(); a.add_argument('--out',required=True); a.add_argument('--split',choices=['dev','fresh'],required=True); a.add_argument('--seeds',type=int,nargs='+',required=True); z=a.parse_args(); rows=[]
 for s in z.seeds: rows+=run(s,z.split,z.out)
 p=Path(z.out)/z.split/'RESULTS_CORE.csv'; p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if __name__=='__main__': main()
