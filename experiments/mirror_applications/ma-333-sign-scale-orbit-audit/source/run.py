"""MA-333 activation-specific monomial symmetry audit."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ID,H,O,N=16,24,8,4096
DEV=[33321,33322];FRESH=[33331,33332,33333]
ACTS=['relu','gelu','tanh','layernorm_relu'];METHODS=['baseline','permutation_view','positive_scale_compensated','sign_flip_compensated']
def params(world):
 r=np.random.default_rng(world);x=r.standard_normal((N,ID),dtype=np.float32)
 w1=(r.standard_normal((ID,H),dtype=np.float32)/np.sqrt(ID)).astype(np.float32);b1=(r.standard_normal(H,dtype=np.float32)*.1).astype(np.float32)
 w2=(r.standard_normal((H,O),dtype=np.float32)/np.sqrt(H)).astype(np.float32);b2=(r.standard_normal(O,dtype=np.float32)*.1).astype(np.float32)
 gamma=(1+r.standard_normal(H,dtype=np.float32)*.1).astype(np.float32);beta=(r.standard_normal(H,dtype=np.float32)*.1).astype(np.float32)
 perm=r.permutation(H).astype(np.int16);pos=np.exp(r.uniform(-.7,.7,H)).astype(np.float32);sign=np.where(r.random(H)<.5,-1.,1.).astype(np.float32)
 return x,w1,b1,w2,b2,gamma,beta,perm,pos,sign
def act(z,name,gamma,beta):
 if name=='relu': return np.maximum(z,0)
 if name=='gelu': return .5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+.044715*z**3)))
 if name=='tanh': return np.tanh(z)
 if name=='layernorm_relu':
  zn=(z-z.mean(-1,keepdims=True))/np.sqrt(z.var(-1,keepdims=True)+1e-5)
  return np.maximum(zn*gamma+beta,0)
 raise ValueError(name)
def evaluate(name,method,x,w1,b1,w2,b2,gamma,beta,perm,pos,sign):
 z=x@w1+b1
 if method=='permutation_view':
  if name=='layernorm_relu': h=act(z[:,perm],name,gamma[perm],beta[perm])
  else: h=act(z[:,perm],name,gamma[perm],beta[perm])
  return h@w2[perm]+b2
 if method=='positive_scale_compensated':
  h=act(z*pos,name,gamma,beta);return h@(w2/pos[:,None])+b2
 if method=='sign_flip_compensated':
  h=act(z*sign,name,gamma,beta);return h@(w2/sign[:,None])+b2
 return act(z,name,gamma,beta)@w2+b2
def pack(name,method,w1,b1,w2,b2,gamma,beta,perm,pos,sign):
 arr=[w1,b1,w2,b2]+([gamma,beta] if name=='layernorm_relu' else [])
 meta={'activation':name,'method':method,'shapes':[list(a.shape) for a in arr],'dtype':'float32','version':1}
 chunks=[a.tobytes() for a in arr]
 if method=='permutation_view':meta['code']='int16 permutation';chunks.append(perm.tobytes())
 elif method=='positive_scale_compensated':meta['code']='float32 positive scales';chunks.append(pos.tobytes())
 elif method=='sign_flip_compensated':meta['code']='float32 signs';chunks.append(sign.tobytes())
 j=json.dumps(meta,sort_keys=True,separators=(',',':')).encode();return b'MA333\0'+struct.pack('<I',len(j))+j+b''.join(chunks)
def load(blob):
 assert blob[:6]==b'MA333\0';n=struct.unpack('<I',blob[6:10])[0];meta=json.loads(blob[10:10+n]);pos=10+n;arr=[]
 for sh in meta['shapes']:
  k=int(np.prod(sh))*4;arr.append(np.frombuffer(blob[pos:pos+k],dtype=np.float32).copy().reshape(sh));pos+=k
 code=None
 if meta['method']=='permutation_view':code=np.frombuffer(blob[pos:],dtype=np.int16).copy()
 elif meta['method'] in ('positive_scale_compensated','sign_flip_compensated'):code=np.frombuffer(blob[pos:],dtype=np.float32).copy()
 return meta,arr,code
def nr(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))
def run(phase):
 rows=[]
 for world in (DEV if phase=='development' else FRESH):
  x,w1,b1,w2,b2,gamma,beta,perm,pos,sign=params(world)
  for name in ACTS:
   ref=evaluate(name,'baseline',x,w1,b1,w2,b2,gamma,beta,perm,pos,sign)
   for method in METHODS:
    t=time.perf_counter();y=evaluate(name,method,x,w1,b1,w2,b2,gamma,beta,perm,pos,sign);sec=time.perf_counter()-t
    blob=pack(name,method,w1,b1,w2,b2,gamma,beta,perm,pos,sign)
    rows.append({'phase':phase,'world':world,'activation':name,'method':method,'serialized_bytes':len(blob),'examples':N,'optimizer_updates':0,'active_compute_proxy':N*ID*H+N*H*O,'wall_time_s':sec,'output_nrmse':nr(y,ref),'max_abs_difference':float(np.max(np.abs(y-ref))),'payload_sha256':hashlib.sha256(blob).hexdigest()})
 path=ROOT/f'{phase.upper()}_RESULTS.csv'
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
