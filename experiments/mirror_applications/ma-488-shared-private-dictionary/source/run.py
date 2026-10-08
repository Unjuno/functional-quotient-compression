#!/usr/bin/env python3
"""MA-488 controlled private-residual allocation over a shared function basis."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
D,R,N,TRAIN,Q=16,4,192,128,64
RATES=(0.,.125,.25,.5);GATE=.10
METHODS=('independent_float32','full_int8','shared_only','mirror_shared_private','native_shared_private')

def world(seed,p):
 gen=torch.Generator().manual_seed(seed+488);u,_=torch.linalg.qr(torch.randn(D*D,R,generator=gen),mode='reduced');atoms=u.T.reshape(R,D,D);codes=torch.randn(N,R,generator=gen)*.35
 mats=(codes@atoms.reshape(R,-1)).reshape(N,D,D);priv=torch.zeros(N,D*D);npriv=round(N*p);perm=torch.randperm(N,generator=gen);idx=perm[:npriv]
 for i in idx:
  v=torch.randn(D*D,generator=gen);v-=u@(u.T@v);v=v/(v.norm()+1e-12);v*=.25+float(torch.rand((),generator=gen))*.10;priv[i]=v
 mats=(mats.reshape(N,-1)+priv).reshape(N,D,D);qg=torch.Generator().manual_seed(seed+1488);queries=torch.randn(N,Q,D,generator=qg);outputs=torch.stack([queries[i]@mats[i].T for i in range(N)])
 return {'atoms':atoms,'codes':codes,'private':priv,'matrices':mats,'queries':queries,'outputs':outputs,'private_rows':idx}

def fit(w,method):
 if method=='independent_float32':return {'matrices':w['matrices']},0.
 if method=='full_int8':
  scale=w['matrices'].abs().amax().reshape(1)/127;q=torch.round(w['matrices']/scale).clamp(-127,127).to(torch.int8);return {'q':q,'scale':scale},0.
 private=(w['private'].norm(dim=1)>GATE);state={'atoms':w['atoms'],'codes':w['codes'],'private_indices':private.nonzero().flatten().to(torch.uint8),'private_vectors':w['private'][private] if method in ('mirror_shared_private','native_shared_private') else torch.empty((0,D*D))};return state,0.

def reconstruct(method,s):
 if method=='independent_float32':return s['matrices']
 if method=='full_int8':return s['q'].float()*s['scale']
 flat=s['codes']@s['atoms'].reshape(R,-1);out=flat.clone()
 if s['private_vectors'].numel():out[s['private_indices'].long()]+=s['private_vectors']
 return out.reshape(N,D,D)

def pack(arr,meta):
 b=io.BytesIO();a={k:np.asarray(v) for k,v in arr.items()};a['__schema_json__']=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);np.savez(b,**a);return b.getvalue()

def serialize(method,s,p):
 if method=='independent_float32':a={'function_matrices':s['matrices'].numpy()}
 elif method=='full_int8':a={'matrices_int8':s['q'].numpy(),'scale':s['scale'].numpy()}
 elif method=='shared_only':a={'shared_atoms':s['atoms'].numpy(),'shared_codes':s['codes'].numpy()}
 else:a={'shared_atoms':s['atoms'].numpy(),'shared_codes':s['codes'].numpy(),'private_indices':s['private_indices'].numpy(),'private_residuals':s['private_vectors'].numpy()}
 return pack(a,{'format':'MA488-function-bank-v1','method':method,'functions':N,'dim':D,'rank':R,'heterogeneity':p,'private_gate_l2':GATE})

def evaluate(seed,p,method):
 w=world(seed,p);start=time.perf_counter();state,_=fit(w,method);fitwall=time.perf_counter()-start;start=time.perf_counter();mat=reconstruct(method,state);decwall=time.perf_counter()-start;start=time.perf_counter();pred=torch.stack([w['queries'][i]@mat[i].T for i in range(N)]);qwall=time.perf_counter()-start
 err=(pred[TRAIN:]-w['outputs'][TRAIN:]);rmse=float(torch.sqrt(torch.mean(err**2)));mask=torch.zeros(N,dtype=torch.bool);mask[w['private_rows']]=True;held_private=mask[TRAIN:];perr=float(torch.sqrt(torch.mean(err[held_private]**2))) if bool(held_private.any()) else 0.
 raw=serialize(method,state,p);n=int(state.get('private_indices',torch.empty(0)).numel());sharedops=N*R*D*D if method not in ('independent_float32','full_int8') else 0;privateops=n*D*D;queryops=(N-TRAIN)*Q*D*D
 m={'overall_heldout_rmse':rmse,'private_heldout_rmse':perr,'private_vectors':n,'private_fraction':n/N,'inference_payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'fit_wall_s':fitwall,'decode_wall_s':decwall,'query_wall_s':qwall,'optimizer_updates':0,'examples_seen':N*Q,'shared_decode_ops_proxy':sharedops,'private_residual_ops_proxy':privateops,'query_ops_proxy':queryops,'active_compute_proxy':sharedops+privateops+queryops}
 return m,raw

def run(seed,out):
 out.mkdir(parents=True,exist_ok=True);results={}
 for p in RATES:
  results[str(p)]={};cache={}
  for method in METHODS:
   key=method
   if method=='native_shared_private':key='mirror_shared_private'
   if key not in cache:cache[key]=evaluate(seed,p,key)
   m,raw=cache[key];results[str(p)][method]=m;(out/f'h{p}_{method}_payload.npz').write_bytes(raw)
  assert (out/f'h{p}_mirror_shared_private_payload.npz').read_bytes()==(out/f'h{p}_native_shared_private_payload.npz').read_bytes()
 doc={'experiment_id':'MA-488','seed':seed,'split':'dev','heterogeneity':results};(out/'metrics.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');return doc

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
