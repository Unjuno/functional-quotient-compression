"""MA-853 factorized role-code vs delta-rule fast memory screen."""
from __future__ import annotations
import json,struct,time,hashlib
import numpy as np
K,V,D=6,6,16
PAIRS=[(i,j) for i in range(K) for j in range(V)]
HOLDOUT={(0,0),(0,3),(1,4),(2,1),(3,5),(5,2)}
TRAIN=[p for p in PAIRS if p not in HOLDOUT]
METHODS=('delta_rule_fast_matrix','mirror_factorized_role_code','vsa_hadamard_binding','independent_pair_table')

def world(seed):
 rng=np.random.default_rng(seed);a=rng.normal(size=(K,D));b=rng.normal(size=(V,D));a/=np.linalg.norm(a,axis=1,keepdims=True);b/=np.linalg.norm(b,axis=1,keepdims=True)
 return a.astype(np.float32),b.astype(np.float32)

def pair_values(a,b):return np.stack([a[i]*b[j] for i,j in PAIRS]).astype(np.float32)

def delta_write(state,key,target):
 pred=state@key;err=target-pred;state+=np.outer(err,key)/(float(key@key)+1e-12);return state

def serialize(meta,arrays):
 specs=[];chunks=[]
 for name in sorted(arrays):
  ar=np.asarray(arrays[name]);
  if ar.dtype.kind=='f':ar=np.ascontiguousarray(ar,dtype='<f4')
  elif ar.dtype.kind in 'ui':ar=np.ascontiguousarray(ar,dtype='|u1')
  else:raise TypeError(ar.dtype)
  raw=ar.tobytes();specs.append({'name':name,'shape':list(ar.shape),'dtype':ar.dtype.str,'nbytes':len(raw)});chunks.append(raw)
 header=json.dumps({'meta':meta,'arrays':specs},sort_keys=True,separators=(',',':')).encode()
 return b'MA853\x01'+struct.pack('<I',len(header))+header+b''.join(chunks)

def deserialize(blob):
 assert blob[:6]==b'MA853\x01';n=struct.unpack('<I',blob[6:10])[0];h=json.loads(blob[10:10+n]);pos=10+n;arr={}
 for spec in h['arrays']:
  raw=blob[pos:pos+spec['nbytes']];pos+=spec['nbytes'];arr[spec['name']]=np.frombuffer(raw,dtype=np.dtype(spec['dtype'])).copy().reshape(spec['shape'])
 assert pos==len(blob)
 return h['meta'],arr

def state(method,seed):
 a,b=world(seed);y=pair_values(a,b);pairids=np.asarray(PAIRS,dtype=np.uint8);meta={'method':method,'key_roles':K,'value_roles':V,'code_dim':D,'heldout_pairs':[list(p) for p in sorted(HOLDOUT)]};arr={'pair_ids':pairids}
 if method=='delta_rule_fast_matrix':
  mem=np.zeros((D,K*V),np.float32);t0=time.perf_counter()
  for p in TRAIN:
   idx=PAIRS.index(p);key=np.zeros(K*V,np.float32);key[idx]=1
   delta_write(mem,key,y[idx])
  write_wall=time.perf_counter()-t0;arr['fast_state']=mem;meta['write_count']=len(TRAIN)
 elif method in ('mirror_factorized_role_code','vsa_hadamard_binding'):arr.update({'key_role_codes':a,'value_role_codes':b})
 elif method=='independent_pair_table':arr['pair_values']=y
 else:raise ValueError(method)
 if method!='delta_rule_fast_matrix':write_wall=0.0
 return meta,arr,write_wall

def predict(method,arr,pair):
 i,j=pair
 if method=='delta_rule_fast_matrix':return arr['fast_state'][:,PAIRS.index(pair)]
 if method in ('mirror_factorized_role_code','vsa_hadamard_binding'):return arr['key_role_codes'][i]*arr['value_role_codes'][j]
 return arr['pair_values'][PAIRS.index(pair)]

def evaluate(seed,method):
 a,b=world(seed);truth=pair_values(a,b);meta,arrays,write_wall=state(method,seed);blob=serialize(meta,arrays);meta,arr=deserialize(blob)
 seen=[];unseen=[];exact=[];query_macs=0;t0=time.perf_counter()
 for idx,pair in enumerate(PAIRS):
  t=time.perf_counter();pred=predict(method,arr,pair);_elapsed=time.perf_counter()-t
  mse=float(np.mean((pred-truth[idx])**2))
  (unseen if pair in HOLDOUT else seen).append(mse)
  nearest=int(np.argmin(np.mean((truth-pred[None,:])**2,axis=1)));exact.append(nearest==idx)
  query_macs += D*K*V if method=='delta_rule_fast_matrix' else D if method in ('mirror_factorized_role_code','vsa_hadamard_binding') else 0
 recall_wall=time.perf_counter()-t0
 norm=float(np.mean(np.square(truth)))
 return {'seed':seed,'method':method,'serialized_bytes':len(blob),'normalized_seen_mse':float(np.mean(seen)/max(norm,1e-30)),'normalized_heldout_mse':float(np.mean(unseen)/max(norm,1e-30)),'heldout_exact_nearest_accuracy':float(np.mean([exact[PAIRS.index(p)] for p in HOLDOUT])),'write_updates':len(TRAIN) if method=='delta_rule_fast_matrix' else 0,'write_macs_proxy':len(TRAIN)*D*K*V*2 if method=='delta_rule_fast_matrix' else 0,'query_macs_proxy':int(query_macs),'write_wall_s':write_wall,'recall_wall_s':recall_wall,'payload_sha256':hashlib.sha256(blob).hexdigest()}

if __name__=='__main__':
 import csv,pathlib,sys
 root=pathlib.Path(__file__).resolve().parents[1];mode=sys.argv[1] if len(sys.argv)>1 else 'dev'
 seeds=(85301,85302) if mode=='dev' else (85311,85312,85313)
 rows=[evaluate(seed,m) for seed in seeds for m in METHODS]
 path=root/('source/development.csv' if mode=='dev' else 'source/fresh.csv')
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 for r in rows:print(r)
