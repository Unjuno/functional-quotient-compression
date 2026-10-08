#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';SEEDS=(36401,36402,36411,36412,36413);D=64;C=8;N=4096;DEPTHS=(2,4,6);FRAC=(.5,.3,.2)
def pack(s,m):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for k,a in sorted(s.items()):
   q=io.BytesIO();np.save(q,np.asarray(a),allow_pickle=False);i=zipfile.ZipInfo(k+'.npy',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,q.getvalue())
  i=zipfile.ZipInfo('metadata.json',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,json.dumps(m,sort_keys=True,separators=(',',':')).encode())
 p=b.getvalue()
 with zipfile.ZipFile(io.BytesIO(p)) as z:s2={n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')};m2=json.loads(z.read('metadata.json'))
 return p,s2,m2
def world(seed):
 r=np.random.default_rng(seed);base=r.normal(0,.1,(D,C)).astype('f4');basis=r.normal(0,.05,(D,C)).astype('f4');coef=np.array([-1.,0.,1.],dtype='f4');heads=np.stack([base+c*basis for c in coef]);x=r.normal(size=(N,D)).astype('f4');difficulty=np.abs(x[:,0]);order=np.argsort(difficulty);exit_id=np.zeros(N,np.int8);exit_id[order[int(.5*N):int(.8*N)]]=1;exit_id[order[int(.8*N):]]=2
 probs=[]
 for w in heads:
  z=x@w;z-=z.max(1,keepdims=True);p=np.exp(z);probs.append(p/p.sum(1,keepdims=True))
 probs=np.stack(probs);y=np.array([r.choice(C,p=probs[exit_id[i],i]) for i in range(N)])
 return base,basis,coef,heads,x,exit_id,y,probs
def states(w):
 b,v,c,h,*_=w
 return {'independent_heads':({'heads':h},{'method':'independent','exits':3}), 'hard_shared_head':({'head':b},{'method':'hard_shared','exits':3}), 'direct_coefficients':({'base':b,'basis':v,'coeff':c},{'method':'direct_coeff','exits':3}), 'mirror_depth_views':({'base':b,'basis':v,'coeff':c},{'method':'mirror_depth','exits':3})}
def logits(name,s,x,e):
 if name=='independent_heads':w=s['heads'][e]
 elif name=='hard_shared_head':w=s['head']
 else:w=s['base']+s['coeff'][e]*s['basis']
 return x@w
def run(seed,name):
 w=world(seed);b,v,c,h,x,e,y,teacher=w;arr,meta=states(w)[name];t=time.perf_counter();p,s,_=pack(arr,meta);all_nll=[];all_acc=[]
 for j in range(3):
  z=logits(name,s,x,j);z-=z.max(1,keepdims=True);q=np.exp(z);q/=q.sum(1,keepdims=True);all_nll.append(float(-np.log(np.clip(q[np.arange(N),y],1e-8,1)).mean()));all_acc.append(float((q.argmax(1)==y).mean()))
 mix=np.empty((N,C),dtype='f4')
 for j in range(3):mix[e==j]=np.exp(logits(name,s,x[e==j],j)-logits(name,s,x[e==j],j).max(1,keepdims=True));mix[e==j]/=mix[e==j].sum(1,keepdims=True)
 nll=float(-np.log(np.clip(mix[np.arange(N),y],1e-8,1)).mean());acc=float((mix.argmax(1)==y).mean());mac=int(N*(sum(DEPTHS[i]*FRAC[i] for i in range(3))*D*D + D*C))
 return {'seed':seed,'method':name,'payload_bytes':len(p),'sha256':hashlib.sha256(p).hexdigest(),'per_exit_NLL':json.dumps(all_nll),'per_exit_accuracy':json.dumps(all_acc),'mixed_NLL':nll,'mixed_accuracy':acc,'exit_fractions':json.dumps(FRAC),'average_MAC_proxy':mac,'examples':N,'optimizer_updates':0,'wall_seconds':time.perf_counter()-t}
def main():
 a=argparse.ArgumentParser();a.add_argument('--dev-only',action='store_true');args=a.parse_args();rows=[]
 for seed in SEEDS[:2] if args.dev_only else SEEDS:
  for name in ('independent_heads','hard_shared_head','direct_coefficients','mirror_depth_views'):
   r=run(seed,name);rows.append(r);print(json.dumps(r,sort_keys=True))
 OUT.mkdir(exist_ok=True);p=OUT/('development.csv' if args.dev_only else 'results.csv')
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
