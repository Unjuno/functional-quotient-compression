#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';SEEDS=(36601,36602,36611,36612,36613);K=4;D=16;O=8;N=512
def pack(s,m):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for n,a in sorted(s.items()):
   q=io.BytesIO();np.save(q,np.asarray(a),allow_pickle=False);i=zipfile.ZipInfo(n+'.npy',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,q.getvalue())
  i=zipfile.ZipInfo('metadata.json',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,json.dumps(m,sort_keys=True,separators=(',',':')).encode())
 p=b.getvalue()
 with zipfile.ZipFile(io.BytesIO(p)) as z:s2={n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')};m2=json.loads(z.read('metadata.json'))
 return p,s2,m2
def world(seed):
 r=np.random.default_rng(seed);base=r.normal(0,.08,(D,O)).astype('f4');dep=r.normal(0,.07,(D,O)).astype('f4');exp=r.normal(0,.07,(D,O)).astype('f4');inter=r.normal(0,.04,(D,O)).astype('f4');a=np.linspace(-1,1,K,dtype='f4');b=np.cos(np.arange(K)*np.pi/(K-1)).astype('f4');maps=np.stack([[base+a[i]*dep+b[j]*exp+a[i]*b[j]*inter for j in range(K)] for i in range(K)]);pairs=[(i,j) for i in range(K) for j in range(K)];r.shuffle(pairs);train=pairs[:12];held=pairs[12:];x=r.normal(size=(N,D)).astype('f4');return base,dep,exp,inter,a,b,maps,pairs,train,held,x
def states(w):
 base,dep,exp,inter,a,b,maps,pairs,train,held,x=w;di=np.repeat(np.arange(K,dtype='u1'),K);ei=np.tile(np.arange(K,dtype='u1'),K)
 return {'flat_paths':({'maps':maps},{'kind':'flat_paths'}),'pa02_factorized':({'base':base,'depth_basis':dep,'expert_basis':exp,'interaction':inter,'depth_coeff':a,'expert_coeff':b},{'kind':'pa02_factorized'}),'mirror_path':({'base':base,'depth_basis':dep,'expert_basis':exp,'interaction':inter,'depth_coeff':a,'expert_coeff':b,'depth_code':di,'expert_code':ei},{'kind':'mirror_path'}),'direct_coefficients':({'base':base,'depth_basis':dep,'expert_basis':exp,'interaction':inter,'depth_coeff':a,'expert_coeff':b,'depth_code':di,'expert_code':ei},{'kind':'direct_pair'})}
def decode(name,s,i,j):
 if name=='flat_paths':return s['maps'][i,j]
 return s['base']+s['depth_coeff'][i]*s['depth_basis']+s['expert_coeff'][j]*s['expert_basis']+s['depth_coeff'][i]*s['expert_coeff'][j]*s['interaction']
def score(name,s,w):
 *_,maps,pairs,train,held,x=w;err=float(np.mean([np.mean((decode(name,s,i,j)-maps[i,j])**2)/(np.mean(maps[i,j]**2)+1e-12) for i,j in held]));outs=[x@decode(name,s,i,j) for i,j in pairs];distinct=len({np.round(q,5).tobytes() for q in outs});route=2 if name in ('mirror_path','direct_coefficients') else K*K
 return err,distinct,route
def run(seed,name):
 w=world(seed);a,m=states(w)[name];t=time.perf_counter();p,s,_=pack(a,m);err,distinct,route=score(name,s,w);return {'seed':seed,'method':name,'payload_bytes':len(p),'sha256':hashlib.sha256(p).hexdigest(),'heldout_nMSE':err,'distinct_paths':distinct,'collision_rate':1-distinct/16,'route_MAC_proxy':route,'operator_MAC_proxy':16*N*D*O,'examples':16*N,'optimizer_updates':0,'wall_seconds':time.perf_counter()-t}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');z=ap.parse_args();rows=[]
 for seed in SEEDS[:2] if z.dev_only else SEEDS:
  for name in ('flat_paths','pa02_factorized','mirror_path','direct_coefficients'):
   r=run(seed,name);rows.append(r);print(json.dumps(r,sort_keys=True))
 OUT.mkdir(exist_ok=True);p=OUT/('development.csv' if z.dev_only else 'results.csv')
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
