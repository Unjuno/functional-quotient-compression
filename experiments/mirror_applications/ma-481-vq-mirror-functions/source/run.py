#!/usr/bin/env python3
"""Frozen MA-481 function-bank VQ address screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
D,R,N,TRAIN=16,4,192,128
QPER=64
METHODS=('independent','continuous','int8','mirror_vq16','native_vq16','mirror_vq64','native_vq64','mirror_vq128','native_vq128')

def world(seed):
 g=torch.Generator().manual_seed(seed+481);q,_=torch.linalg.qr(torch.randn(D*D,R,generator=g),mode='reduced');atoms=q.T.reshape(R,D,D)
 coeff=torch.randn(N,R,generator=g)*.35;mat=torch.einsum('nr,rij->nij',coeff,atoms)
 xgen=torch.Generator().manual_seed(seed+1481);queries=torch.randn(N,QPER,D,generator=xgen)
 outputs=torch.stack([queries[e]@mat[e].T for e in range(N)])
 return {'teacher_atoms':atoms,'coeff':coeff,'matrices':mat,'queries':queries,'outputs':outputs}

def kmeans(x,k,seed):
 gen=torch.Generator().manual_seed(seed+k);centers=x[torch.randperm(x.shape[0],generator=gen)[:k]].clone()
 for _ in range(100):
  lab=torch.cdist(x,centers).argmin(1);new=centers.clone()
  for j in range(k):
   m=lab==j
   if bool(m.any()):new[j]=x[m].mean(0)
  centers=new
 return centers,torch.cdist(x,centers).argmin(1)

def fit(w,method,seed):
 start=time.perf_counter();m=method.replace('mirror_','').replace('native_','')
 if m=='independent':state={'matrices':w['matrices']};fitops=0;fitwall=0.
 else:
  _,_,vh=torch.linalg.svd(w['matrices'][:TRAIN].reshape(TRAIN,-1),full_matrices=False);atoms=vh[:R].reshape(R,D,D);z=w['matrices'].reshape(N,-1)@vh[:R].T
  fitwall=time.perf_counter()-start;fitops=TRAIN*D*D*R*3
  if m=='continuous':state={'atoms':atoms,'codes':z}
  elif m=='int8':
   scale=z[:TRAIN].abs().amax(0,keepdim=True).clamp_min(1e-8)/127;q=torch.round(z/scale).clamp(-127,127).to(torch.int8);state={'atoms':atoms,'q':q,'scale':scale}
  else:
   k=int(m.replace('vq',''));centers,lab=kmeans(z[:TRAIN],k,seed);alllab=torch.cdist(z,centers).argmin(1).to(torch.uint8);state={'atoms':atoms,'centers':centers,'indices':alllab,'k':k};fitops+=TRAIN*k*R*100*3
 return state,fitwall,fitops

def reconstruct(method,state):
 m=method.replace('mirror_','').replace('native_','')
 if m=='independent':return state['matrices']
 if m=='continuous':flat=state['codes']@state['atoms'].reshape(R,-1)
 elif m=='int8':flat=(state['q'].float()*state['scale'])@state['atoms'].reshape(R,-1)
 else:flat=state['centers'][state['indices'].long()]@state['atoms'].reshape(R,-1)
 return flat.reshape(N,D,D)

def pack(arr,meta):
 b=io.BytesIO();d={k:np.asarray(v) for k,v in arr.items()};d['__schema_json__']=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);np.savez(b,**d);return b.getvalue()

def serialize(method,state):
 m=method.replace('mirror_','').replace('native_','')
 if m=='independent':obj={'function_matrices':state['matrices'].numpy()};fam='independent-function-bank-v1'
 elif m=='continuous':obj={'shared_atoms':state['atoms'].numpy(),'function_codes':state['codes'].numpy()};fam='continuous-shared-function-bank-v1'
 elif m=='int8':obj={'shared_atoms':state['atoms'].numpy(),'codes_int8':state['q'].numpy(),'code_scales':state['scale'].numpy()};fam='int8-shared-function-bank-v1'
 else:obj={'shared_atoms':state['atoms'].numpy(),'vq_centers':state['centers'].numpy(),'function_indices':state['indices'].numpy()};fam='vq-shared-function-bank-v1'
 return pack(obj,{'format':'MA481-logical-linear-function-bank-v1','method_family':fam,'functions':N,'input_dim':D,'output_dim':D,'latent_rank':R,'vq_size':state.get('k')})

def evaluate(method,w,seed):
 state,wall,fitops=fit(w,method,seed);predmat=reconstruct(method,state);start=time.perf_counter();pred=torch.stack([w['queries'][e]@predmat[e].T for e in range(N)])
 errors=(pred[TRAIN:]-w['outputs'][TRAIN:]);rmse=float(torch.sqrt(torch.mean(errors**2)));qwall=time.perf_counter()-start
 m=method.replace('mirror_','').replace('native_','')
 if m.startswith('vq'):uniq=int(torch.unique(state['indices'][TRAIN:]).numel());unique_fraction=uniq/(N-TRAIN)
 elif m=='continuous':unique_fraction=float(torch.unique(torch.round(state['codes'][TRAIN:]*1e6),dim=0).shape[0]/(N-TRAIN))
 elif m=='int8':unique_fraction=float(torch.unique(state['q'][TRAIN:],dim=0).shape[0]/(N-TRAIN))
 else:unique_fraction=1.0
 raw=serialize(method,state);ops=(N-TRAIN)*QPER*D*D
 metrics={'heldout_function_rmse':rmse,'distinct_address_fraction':unique_fraction,'unique_heldout_addresses':int(round(unique_fraction*(N-TRAIN))),'inference_payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'fit_wall_s':wall,'query_wall_s':qwall,'fit_ops_proxy':fitops,'decode_ops_per_function':D*D*R if m!='independent' else D*D,'query_ops_proxy':ops,'active_ops_proxy':fitops+ops,'optimizer_updates':0,'examples_seen':N*QPER}
 return metrics,raw

def run(seed,out):
 out.mkdir(parents=True,exist_ok=True);w=world(seed);res={};cache={}
 for method in METHODS:
  key=method.replace('mirror_','').replace('native_','')
  if method.startswith('native_vq'):cachekey='mirror_'+key
  else:cachekey=method
  if cachekey in cache:m,raw=cache[cachekey]
  else:m,raw=evaluate(method,w,seed);cache[cachekey]=(m,raw)
  res[method]=m;(out/f'{method}_payload.npz').write_bytes(raw)
 for k in (16,64,128):
  assert (out/f'mirror_vq{k}_payload.npz').read_bytes()==(out/f'native_vq{k}_payload.npz').read_bytes()
 doc={'experiment_id':'MA-481','seed':seed,'split':'dev','task':{'functions':N,'train':TRAIN,'heldout':N-TRAIN,'dim':D,'rank':R,'queries_per_function':QPER},'methods':res};(out/'metrics.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');return doc

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
