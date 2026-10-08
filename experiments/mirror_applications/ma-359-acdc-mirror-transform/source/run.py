#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';SEEDS=(35901,35902,35911,35912,35913);D=32;T=64;N=512

def pack(s,m):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for k,a in sorted(s.items()):
   q=io.BytesIO();np.save(q,np.asarray(a),allow_pickle=False);i=zipfile.ZipInfo(k+'.npy',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,q.getvalue())
  i=zipfile.ZipInfo('metadata.json',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,json.dumps(m,sort_keys=True,separators=(',',':')).encode())
 p=b.getvalue()
 with zipfile.ZipFile(io.BytesIO(p)) as z:s2={n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')};m2=json.loads(z.read('metadata.json'))
 return p,s2,m2

def dct():
 x=np.arange(D);k=x[:,None];q=np.sqrt(2/D)*np.cos(np.pi*(x+.5)*k/D);q[0,:]=np.sqrt(1/D);return q.astype('f4')

def world(seed):
 r=np.random.default_rng(seed);q=dct();d0=(.8+r.random(D)*.4).astype('f4');d1=(r.normal(0,.08,D)).astype('f4');e0=(.8+r.random(D)*.4).astype('f4');e1=r.normal(0,.08,D).astype('f4');alpha=np.linspace(-1,1,T,dtype='f4');beta=np.sin(np.arange(T)*2*np.pi/T).astype('f4');rng=r.normal(size=(N,D)).astype('f4');return q,d0,d1,e0,e1,alpha,beta,rng

def state(method,w):
 q,d0,d1,e0,e1,a,b,x=w
 if method=='dense_independent':
  mats=np.stack([np.diag(d0+a[i]*d1)@q@np.diag(e0+b[i]*e1) for i in range(T)]);return {'maps':mats},{'method':method,'D':D}
 if method=='independent_acdc':
  dl=np.stack([d0+a[i]*d1 for i in range(T)]);dr=np.stack([e0+b[i]*e1 for i in range(T)]);return {'left_diag':dl,'right_diag':dr},{'method':method,'D':D}
 if method=='diagonal_only':
  dl=np.stack([d0+a[i]*d1 for i in range(T)]);return {'diag':dl},{'method':method,'D':D}
 s={'d0':d0,'d1':d1,'e0':e0,'e1':e1,'alpha':a,'beta':b}
 return s,{'method':method,'D':D,'view':'2-scalar'}

def apply(method,s,i,x):
 if method=='dense_independent':return x@s['maps'][i].T
 if method=='diagonal_only':return x*s['diag'][i]
 if method=='independent_acdc':return ((x*s['right_diag'][i])@dct().T)*s['left_diag'][i]
 dl=s['d0']+s['alpha'][i]*s['d1'];dr=s['e0']+s['beta'][i]*s['e1'];return ((x*dr)@dct().T)*dl

def run(seed,method):
 w=world(seed);q,d0,d1,e0,e1,a,b,x=w;train=set(range(0,48));held=set(range(48,64));s,m=state(method,w);p,lm,meta=pack(s,m);errs=[];functions=[];start=time.perf_counter()
 for i in range(T):
  yt=((x*(e0+b[i]*e1))@q.T)*(d0+a[i]*d1);yp=apply(method,lm,i,x);errs.append(float(np.mean((yp-yt)**2)/(np.mean(yt**2)+1e-12)));functions.append(yp)
 uniq=len({np.round(z,5).tobytes() for z in functions});
 if method=='dense_independent':mac=int(T*N*D*D)
 elif method=='diagonal_only':mac=int(T*N*D)
 else:mac=int(T*N*(2*D*int(np.log2(D))+2*D))
 row={'seed':seed,'method':method,'payload_bytes':len(p),'sha256':hashlib.sha256(p).hexdigest(),'heldout_nMSE':float(np.mean([errs[i] for i in held])),'train_nMSE':float(np.mean([errs[i] for i in train])),'distinct_functions':uniq,'transform_MAC_proxy':mac,'examples':N*T,'optimizer_updates':0,'wall_seconds':time.perf_counter()-start};return row

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');a=ap.parse_args();OUT.mkdir(exist_ok=True);rows=[]
 for seed in SEEDS[:2] if a.dev_only else SEEDS:
  for method in ('dense_independent','independent_acdc','mirror_acdc','direct_coefficients','diagonal_only'):
   row=run(seed,method);rows.append(row);print(json.dumps(row,sort_keys=True))
 p=OUT/('development.csv' if a.dev_only else 'results.csv')
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
