#!/usr/bin/env python3
"""MA-486 sparse functional dictionary screen with exact OMP reference."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
D,K,N,TRAIN,S,Q=16,32,192,128,3,64
METHODS=('independent','dense_float','dense_int8','mirror_sparse_float','native_omp_float','mirror_sparse_int8','native_omp_int8')

def world(seed):
 g=torch.Generator().manual_seed(seed+486);q,_=torch.linalg.qr(torch.randn(D*D,K,generator=g),mode='reduced');atoms=q.T.reshape(K,D,D)
 coeff=torch.zeros(N,K);tg=torch.Generator().manual_seed(seed+1486)
 for i in range(N):
  ids=torch.randperm(K,generator=tg)[:S];vals=(torch.rand(S,generator=tg)*.3+.3)*torch.where(torch.rand(S,generator=tg)<.5,-1.,1.);coeff[i,ids]=vals
 mats=(coeff@atoms.reshape(K,-1)).reshape(N,D,D)
 qg=torch.Generator().manual_seed(seed+2486);queries=torch.randn(N,Q,D,generator=qg);outputs=torch.stack([queries[i]@mats[i].T for i in range(N)])
 return {'atoms':atoms,'coeff':coeff,'matrices':mats,'queries':queries,'outputs':outputs}

def omp(mats,atoms):
 start=time.perf_counter();flat=mats.reshape(N,-1);A=atoms.reshape(K,-1);res=flat.clone();ids=[];vals=[]
 for _ in range(S):
  scores=res@A.T;idx=scores.abs().argmax(1);val=scores[torch.arange(N),idx];res-=val[:,None]*A[idx];ids.append(idx);vals.append(val)
 wall=time.perf_counter()-start
 return torch.stack(ids,1),torch.stack(vals,1),wall,S*N*K*D*D

def fit(w,method):
 if method=='independent':return {'matrices':w['matrices']},0.,0.
 ids,vals,encodewall,encops=omp(w['matrices'],w['atoms']);state={'atoms':w['atoms']}
 if method=='dense_float':state['codes']=w['coeff']
 elif method=='dense_int8':
  scale=w['coeff'].abs().amax().reshape(1,1).clamp_min(1e-8)/127;state.update({'q':torch.round(w['coeff']/scale).clamp(-127,127).to(torch.int8),'scale':scale})
 elif method.endswith('float'):state.update({'indices':ids,'values':vals})
 else:
  scale=vals.abs().amax().reshape(1,1).clamp_min(1e-8)/127;state.update({'indices':ids,'q':torch.round(vals/scale).clamp(-127,127).to(torch.int8),'scale':scale})
 return state,encodewall,encops

def reconstruct(method,state):
 if method=='independent':return state['matrices']
 if method=='dense_float':c=state['codes']
 elif method=='dense_int8':c=state['q'].float()*state['scale']
 else:
  v=state['values'] if method.endswith('float') else state['q'].float()*state['scale'];c=torch.zeros(N,K)
  c.scatter_add_(1,state['indices'],v)
 return (c@state['atoms'].reshape(K,-1)).reshape(N,D,D)

def pack(arr,meta):
 b=io.BytesIO();a={k:np.asarray(v) for k,v in arr.items()};a['__schema_json__']=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);np.savez(b,**a);return b.getvalue()

def serialize(method,s):
 if method=='independent':a={'function_matrices':s['matrices'].numpy()}
 elif method=='dense_float':a={'shared_atoms':s['atoms'].numpy(),'dense_codes':s['codes'].numpy()}
 elif method=='dense_int8':a={'shared_atoms':s['atoms'].numpy(),'codes_int8':s['q'].numpy(),'scale':s['scale'].numpy()}
 elif method.endswith('float'):a={'shared_atoms':s['atoms'].numpy(),'sparse_indices':s['indices'].to(torch.uint8).numpy(),'sparse_values':s['values'].numpy()}
 else:a={'shared_atoms':s['atoms'].numpy(),'sparse_indices':s['indices'].to(torch.uint8).numpy(),'sparse_values_int8':s['q'].numpy(),'scale':s['scale'].numpy()}
 return pack(a,{'format':'MA486-sparse-function-dictionary-v1','method_family':'independent' if method=='independent' else ('dense' if method.startswith('dense') else 'omp-sparse'),'functions':N,'dim':D,'dictionary_atoms':K,'nnz_per_function':S})

def evaluate(method,w):
 start=time.perf_counter();s,encodewall,encops=fit(w,method);fitwall=time.perf_counter()-start;start=time.perf_counter();mat=reconstruct(method,s);decodewall=time.perf_counter()-start;start=time.perf_counter();pred=torch.stack([w['queries'][i]@mat[i].T for i in range(N)]);querywall=time.perf_counter()-start
 rmse=float(torch.sqrt(torch.mean((pred[TRAIN:]-w['outputs'][TRAIN:])**2)));raw=serialize(method,s);dense=method.startswith('dense');sparse=method.startswith(('mirror_sparse','native_omp'))
 if sparse:
  vals=s['values'] if method.endswith('float') else s['q'].float()
  uniq=int(torch.unique(torch.round(torch.cat([s['indices'][TRAIN:].float(),vals[TRAIN:]],dim=1)*1e5),dim=0).shape[0]);decodeops=N*S*D*D
 else:uniq=N-TRAIN;decodeops=N*K*D*D if method!='independent' else 0
 queryops=(N-TRAIN)*Q*D*D
 m={'heldout_function_rmse':rmse,'distinct_sparse_code_fraction':uniq/(N-TRAIN),'serialized_payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'optimizer_updates':0,'examples_seen':N*Q,'fit_wall_s':fitwall,'omp_encode_wall_s':encodewall,'decode_wall_s':decodewall,'query_wall_s':querywall,'omp_encode_ops_proxy':encops,'shared_decode_ops_proxy':decodeops,'query_ops_proxy':queryops,'active_compute_proxy':encops+decodeops+queryops}
 return m,raw

def run(seed,out):
 out.mkdir(parents=True,exist_ok=True);w=world(seed);res={};cache={}
 pairs={'native_omp_float':'mirror_sparse_float','native_omp_int8':'mirror_sparse_int8'}
 for method in METHODS:
  key=pairs.get(method,method)
  if key not in cache:cache[key]=evaluate(key,w)
  m,raw=cache[key];res[method]=m;(out/f'{method}_payload.npz').write_bytes(raw)
 for a,b in pairs.items():assert (out/f'{a}_payload.npz').read_bytes()==(out/f'{b}_payload.npz').read_bytes()
 doc={'experiment_id':'MA-486','seed':seed,'split':'dev','task':{'functions':N,'train':TRAIN,'heldout':N-TRAIN,'dim':D,'atoms':K,'nnz':S},'methods':res};(out/'metrics.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');return doc

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
