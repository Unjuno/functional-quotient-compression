#!/usr/bin/env python3
"""Frozen MA-482 residual-VQ logical-function screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
D,R,N,TRAIN,QPER=16,4,192,128,64
METHODS=('independent','continuous','int8','vq16','vq64','vq128',*[f'{prefix}rvq{k}x{l}' for prefix in ('mirror_','native_') for k in (8,16) for l in (2,3,4)])

def normalize(method):return method.replace('mirror_','').replace('native_','')

def world(seed):
 g=torch.Generator().manual_seed(seed+482);q,_=torch.linalg.qr(torch.randn(D*D,R,generator=g),mode='reduced');atoms=q.T.reshape(R,D,D)
 coeff=torch.randn(N,R,generator=g)*.35;mat=torch.einsum('nr,rij->nij',coeff,atoms)
 xgen=torch.Generator().manual_seed(seed+1482);queries=torch.randn(N,QPER,D,generator=xgen);outputs=torch.stack([queries[e]@mat[e].T for e in range(N)])
 return {'coeff':coeff,'matrices':mat,'queries':queries,'outputs':outputs}

def kmeans(x,k,seed):
 gen=torch.Generator().manual_seed(seed+k);centers=x[torch.randperm(x.shape[0],generator=gen)[:k]].clone()
 for _ in range(100):
  labels=torch.cdist(x,centers).argmin(1);new=centers.clone()
  for j in range(k):
   mask=labels==j
   if bool(mask.any()):new[j]=x[mask].mean(0)
  centers=new
 return centers,torch.cdist(x,centers).argmin(1)

def fit(w,method,seed):
 method=normalize(method)
 start=time.perf_counter()
 if method=='independent':state={'matrices':w['matrices']};ops=0;wall=0.
 else:
  _,_,vh=torch.linalg.svd(w['matrices'][:TRAIN].reshape(TRAIN,-1),full_matrices=False);atoms=vh[:R].reshape(R,D,D);z=w['matrices'].reshape(N,-1)@vh[:R].T
  ops=TRAIN*D*D*R*3;wall=time.perf_counter()-start
  if method=='continuous':state={'atoms':atoms,'codes':z}
  elif method=='int8':
   scale=z[:TRAIN].abs().amax(0,keepdim=True).clamp_min(1e-8)/127;codes=torch.round(z/scale).clamp(-127,127).to(torch.int8);state={'atoms':atoms,'q':codes,'scale':scale}
  else:
   if method.startswith('vq'):
    ks=[int(method[2:])];depth=1
   else:
    k,l=method[3:].split('x');ks=[int(k)]*int(l);depth=int(l)
   res_train=z[:TRAIN].clone();res_all=z.clone();centers_list=[];idx_list=[]
   for stage,k in enumerate(ks):
    centers,train_idx=kmeans(res_train,k,seed+stage*7919);all_idx=torch.cdist(res_all,centers).argmin(1)
    res_train-=centers[train_idx];res_all-=centers[all_idx];centers_list.append(centers);idx_list.append(all_idx)
    ops+=TRAIN*k*R*100*3+N*k*R
   state={'atoms':atoms,'centers':torch.stack(centers_list),'indices':torch.stack(idx_list,dim=1),'ks':ks,'depth':depth}
 return state,wall,ops

def reconstruct(method,state):
 method=normalize(method)
 if method=='independent':return state['matrices']
 if method=='continuous':z=state['codes']
 elif method=='int8':z=state['q'].float()*state['scale']
 elif method.startswith('vq'):
  z=state['centers'][0][state['indices'][:,0]]
 else:
  z=torch.stack([state['centers'][s][state['indices'][:,s]] for s in range(state['depth'])]).sum(0)
 if method=='continuous' or method=='int8':atoms=state['atoms'].reshape(R,-1)
 else:atoms=state['atoms'].reshape(R,-1)
 return (z@atoms).reshape(N,D,D)

def pack(arr,meta):
 b=io.BytesIO();d={k:np.asarray(v) for k,v in arr.items()};d['__schema_json__']=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);np.savez(b,**d);return b.getvalue()

def serialize(method,state):
 method=normalize(method)
 if method=='independent':obj={'function_matrices':state['matrices'].numpy()};family='independent-v1'
 elif method=='continuous':obj={'shared_atoms':state['atoms'].numpy(),'function_codes':state['codes'].numpy()};family='continuous-v1'
 elif method=='int8':obj={'shared_atoms':state['atoms'].numpy(),'codes_int8':state['q'].numpy(),'scales':state['scale'].numpy()};family='int8-v1'
 elif method.startswith('vq'):obj={'shared_atoms':state['atoms'].numpy(),'centers':state['centers'].numpy(),'indices':state['indices'].numpy()};family='single-vq-v1'
 else:obj={'shared_atoms':state['atoms'].numpy(),'stage_centers':state['centers'].numpy(),'stage_indices':state['indices'].numpy()};family='residual-vq-v1'
 return pack(obj,{'format':'MA482-function-bank-v1','method_family':family,'functions':N,'dim':D,'rank':R,'stage_sizes':state.get('ks')})

def evaluate(method,w,seed):
 method=normalize(method);state,fitwall,fitops=fit(w,method,seed);t=time.perf_counter();predmat=reconstruct(method,state);decodewall=time.perf_counter()-t;t=time.perf_counter();pred=torch.stack([w['queries'][e]@predmat[e].T for e in range(N)]);qwall=time.perf_counter()-t
 rmse=float(torch.sqrt(torch.mean((pred[TRAIN:]-w['outputs'][TRAIN:])**2)));raw=serialize(method,state)
 if method.startswith('rvq'):uniq=int(torch.unique(state['indices'][TRAIN:],dim=0).shape[0]);fraction=uniq/(N-TRAIN);stages=state['depth'];vfit=fitops
 elif method.startswith('vq'):uniq=int(torch.unique(state['indices'][TRAIN:]).numel());fraction=uniq/(N-TRAIN);stages=1;vfit=fitops
 elif method=='continuous':uniq=int(torch.unique(torch.round(state['codes'][TRAIN:]*1e6),dim=0).shape[0]);fraction=uniq/(N-TRAIN);stages=0;vfit=0
 elif method=='int8':uniq=int(torch.unique(state['q'][TRAIN:],dim=0).shape[0]);fraction=uniq/(N-TRAIN);stages=0;vfit=0
 else:uniq=N-TRAIN;fraction=1.;stages=0;vfit=0
 decodeops=N*R*D*D if method!='independent' else 0;idxops=N*max(stages,1) if method.startswith(('vq','rvq')) else (N*R if method in ('continuous','int8') else 0);queryops=(N-TRAIN)*QPER*D*D
 m={'heldout_function_rmse':rmse,'distinct_address_fraction':fraction,'unique_heldout_addresses':uniq,'inference_payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'fit_wall_s':fitwall,'decode_wall_s':decodewall,'query_wall_s':qwall,'fit_ops_proxy':fitops,'vq_fit_ops':vfit,'shared_decode_ops':decodeops,'index_decode_ops':idxops,'query_ops_proxy':queryops,'active_ops_proxy':fitops+decodeops+idxops+queryops,'optimizer_updates':0,'examples_seen':N*QPER}
 return m,raw

def run(seed,out):
 out.mkdir(parents=True,exist_ok=True);w=world(seed);res={};cache={}
 for method in METHODS:
  key=normalize(method)
  if key not in cache:cache[key]=evaluate(key,w,seed)
  m,raw=cache[key];res[method]=m;(out/f'{method}_payload.npz').write_bytes(raw)
 for method in METHODS:
  if method.startswith('mirror_rvq'):
   native=method.replace('mirror_','native_',1);assert (out/f'{method}_payload.npz').read_bytes()==(out/f'{native}_payload.npz').read_bytes()
 doc={'experiment_id':'MA-482','seed':seed,'split':'dev','task':{'functions':N,'train':TRAIN,'heldout':N-TRAIN,'dim':D,'rank':R,'queries_per_function':QPER},'methods':res};(out/'metrics.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');return doc

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
