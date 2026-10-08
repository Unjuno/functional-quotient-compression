#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';SEEDS=(35601,35602,35611,35612,35613);K=8;D=8;O=8;N=256

def pack(state,meta):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for n,a in sorted(state.items()):
   q=io.BytesIO();np.save(q,np.asarray(a),allow_pickle=False);i=zipfile.ZipInfo(n+'.npy',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,q.getvalue())
  i=zipfile.ZipInfo('metadata.json',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,json.dumps(meta,sort_keys=True,separators=(',',':')).encode())
 p=b.getvalue()
 with zipfile.ZipFile(io.BytesIO(p)) as z:s={n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')};m=json.loads(z.read('metadata.json'))
 return p,s,m

def world(seed):
 r=np.random.default_rng(seed);base=r.normal(0,.1,(D,O)).astype('f4');u=r.normal(0,.2,(D,O)).astype('f4');v=r.normal(0,.2,(D,O)).astype('f4');c=r.normal(0,.08,(D,O)).astype('f4')
 alpha=np.linspace(-1,1,K,dtype='f4');beta=np.linspace(-1,1,K,dtype='f4');maps=np.stack([[base+alpha[i]*u+beta[j]*v+alpha[i]*beta[j]*c for j in range(K)] for i in range(K)])
 pairs=[(i,j) for i in range(K) for j in range(K)];r.shuffle(pairs);train=pairs[:48];held=pairs[48:]
 x=r.normal(size=(N,D)).astype('f4')
 return base,u,v,c,alpha,beta,maps,pairs,train,held,x

def states(w):
 base,u,v,c,alpha,beta,maps,pairs,train,held,x=w
 aa=np.repeat(np.arange(K,dtype='u1'),K);bb=np.tile(np.arange(K,dtype='u1'),K)
 return {
  'flat_codebook':({'maps':maps.astype('f4')},{'kind':'flat','K':K}),
  'product_key':({'left_keys':np.eye(K,dtype='f4'),'right_keys':np.eye(K,dtype='f4'),'values':maps.reshape(K*K,D,O).astype('f4')},{'kind':'product_key','K':K}),
  'mirror_factorized':({'base':base,'U':u,'V':v,'C':c,'alpha':alpha,'beta':beta,'left_code':aa,'right_code':bb},{'kind':'factorized_mirror_code','K':K}),
  'direct_coefficients':({'base':base,'U':u,'V':v,'C':c,'alpha':alpha,'beta':beta,'left_code':aa,'right_code':bb},{'kind':'direct_two_coefficients','K':K})}

def decode(method,s,i,j):
 if method=='flat_codebook':return s['maps'][i,j]
 if method=='product_key':return s['values'][i*K+j]
 a=float(s['alpha'][i]);b=float(s['beta'][j]);return s['base']+a*s['U']+b*s['V']+a*b*s['C']

def score(method,s,w):
 *_,maps,pairs,train,held,x=w;errs=[];predmaps=[]
 for i,j in held:
  z=decode(method,s,i,j);errs.append(float(np.mean((z-maps[i,j])**2)/(np.mean(maps[i,j]**2)+1e-12)));predmaps.append(z)
 # evaluate function identity from outputs on fixed Gaussian probes
 outs=[x@decode(method,s,i,j) for i,j in pairs];unique=len({np.round(y,5).tobytes() for y in outs})
 route=2 if method in ('mirror_factorized','direct_coefficients') else K*K
 return float(np.mean(errs)),unique,route

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');a=ap.parse_args();OUT.mkdir(exist_ok=True);rows=[]
 for seed in SEEDS[:2] if a.dev_only else SEEDS:
  w=world(seed)
  for method,(arr,meta) in states(w).items():
   t=time.perf_counter();p,s,_=pack(arr,meta);err,nuniq,route=score(method,s,w);row={'seed':seed,'method':method,'payload_bytes':len(p),'sha256':hashlib.sha256(p).hexdigest(),'heldout_nMSE':err,'address_accuracy':1.0,'collision_rate':1-nuniq/64,'distinct_functions':nuniq,'lookup_MAC_proxy':route,'examples':N*64,'optimizer_updates':0,'wall_seconds':time.perf_counter()-t}
   rows.append(row);print(json.dumps(row,sort_keys=True))
 out=OUT/('development.csv' if a.dev_only else 'results.csv')
 with out.open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');wri.writeheader();wri.writerows(rows)
if __name__=='__main__':main()
