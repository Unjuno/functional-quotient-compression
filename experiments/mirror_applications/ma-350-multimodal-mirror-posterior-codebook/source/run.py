"""Oracle-aligned posterior mode codebook; exact serialized bytes and mixture metrics."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=16;K=32;DEV=[35021,35022];FRESH=[35031,35032,35033]
METHODS=['mirror_phase','direct_coefficients','independent_vectors','single_shared']
def world(seed):
 r=np.random.default_rng(seed);base=(r.standard_normal(D,dtype=np.float32)*.55).astype(np.float32);ph=np.arange(K,dtype=np.float32)*np.float32(2*np.pi/K);weights=np.full(K,1/K,np.float32)
 def modes():
  w=np.repeat(base[None,:],K,axis=0);c=np.cos(ph);s=np.sin(ph);a,b=base[0],base[1];w[:,0]=c*a-s*b;w[:,1]=s*a+c*b;return w
 wm=modes()
 def data(n,ood=False):
  x=r.standard_normal((n,D),dtype=np.float32)
  if ood:x[:,0]+=.5;x[:,1]*=1.5
  pm=1/(1+np.exp(-np.clip(x@wm.T,-30,30)));mix=pm@weights;y=r.binomial(1,mix).astype(np.float32)
  return x,y
 xi,yi=data(4096);xo,yo=data(4096,True)
 return dict(base=base,phase=ph,weights=weights,wm=wm,xi=xi,yi=yi,xo=xo,yo=yo)
def coeff(ph):return np.stack([np.cos(ph),np.sin(ph)],-1).astype(np.float32)
def mode_vectors(method,a):
 if method in ('mirror_phase','direct_coefficients'):return a['wm']
 if method=='independent_vectors':return a['wm'].copy()
 return a['base'][None,:]
def pack(method,a):
 arr=[a['base']]
 if method=='mirror_phase':arr += [a['phase'][:,None]]
 elif method=='direct_coefficients':arr += [coeff(a['phase'])]
 elif method=='independent_vectors':arr += [a['wm']]
 arr += [a['weights']]
 meta={'method':method,'shapes':[list(x.shape) for x in arr],'dtype':'float32','mode_count':K if method!='single_shared' else 1,'version':1};j=json.dumps(meta,sort_keys=True,separators=(',',':')).encode();return b'MA350\0'+struct.pack('<I',len(j))+j+b''.join(np.asarray(x,dtype=np.float32).tobytes() for x in arr)
def load(blob):
 assert blob[:6]==b'MA350\0';n=struct.unpack('<I',blob[6:10])[0];meta=json.loads(blob[10:10+n]);pos=10+n;arr=[]
 for sh in meta['shapes']:
  z=int(np.prod(sh))*4;arr.append(np.frombuffer(blob[pos:pos+z],dtype=np.float32).copy().reshape(sh));pos+=z
 return meta,arr
def predictions(method,a,split):
 x=a['xi'] if split=='iid' else a['xo'];y=a['yi'] if split=='iid' else a['yo'];wm=mode_vectors(method,a)
 if method=='single_shared':probs=1/(1+np.exp(-np.clip(x@wm[0],-30,30)));members=probs[:,None]
 else:members=1/(1+np.exp(-np.clip(x@wm.T,-30,30)));probs=members@a['weights']
 eps=1e-12;nll=float(-np.mean(y*np.log(probs+eps)+(1-y)*np.log(1-probs+eps)));brier=float(np.mean((probs-y)**2));acc=float(np.mean((probs>=.5)==y));ece=0.
 for i in range(10):
  lo=i/10;hi=(i+1)/10;mask=(probs>=lo)&(probs<(hi if i<9 else 1.000001))
  if mask.any():ece+=mask.mean()*abs(float(probs[mask].mean())-float(y[mask].mean()))
 if members.shape[1]>1:
  dis=float(np.mean(np.abs(members[:,:,None]-members[:,None,:]))/2)
 else:dis=0.
 return nll,brier,ece,acc,dis

def run(phase):
 rows=[]
 for seed in DEV if phase=='development' else FRESH:
  a=world(seed)
  for method in METHODS:
   blob=pack(method,a)
   for split in ('iid','ood'):
    st=time.perf_counter();nll,brier,ece,acc,dis=predictions(method,a,split);sec=time.perf_counter()-st
    rows.append({'phase':phase,'world':seed,'method':method,'split':split,'serialized_bytes':len(blob),'examples':4096,'optimizer_updates':0,'active_compute_proxy':4096*(K if method!='single_shared' else 1)*D,'wall_time_s':sec,'nll':nll,'brier':brier,'ece_10':ece,'accuracy':acc,'pairwise_disagreement':dis,'payload_sha256':hashlib.sha256(blob).hexdigest()})
 path=ROOT/f'{phase.upper()}_RESULTS.csv'
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
