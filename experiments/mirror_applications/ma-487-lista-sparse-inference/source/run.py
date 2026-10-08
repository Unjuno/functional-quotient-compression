#!/usr/bin/env python3
"""MA-487: learned LISTA sparse coordinate inference screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
D,K,N,TRAIN,S,Q=16,32,192,128,3,64
METHODS=('independent','dense_projection','direct_top3','omp3','lista1','lista2','lista4')

def world(seed):
 g=torch.Generator().manual_seed(seed+486);q,_=torch.linalg.qr(torch.randn(D*D,K,generator=g),mode='reduced');atoms=q.T.reshape(K,D,D)
 coeff=torch.zeros(N,K);tg=torch.Generator().manual_seed(seed+1486)
 for i in range(N):
  ids=torch.randperm(K,generator=tg)[:S];vals=(torch.rand(S,generator=tg)*.3+.3)*torch.where(torch.rand(S,generator=tg)<.5,-1.,1.);coeff[i,ids]=vals
 mats=(coeff@atoms.reshape(K,-1)).reshape(N,D,D);qg=torch.Generator().manual_seed(seed+2486);queries=torch.randn(N,Q,D,generator=qg);outputs=torch.stack([queries[i]@mats[i].T for i in range(N)])
 return {'atoms':atoms,'coeff':coeff,'matrices':mats,'queries':queries,'outputs':outputs}

def soft(x,t):return torch.sign(x)*torch.relu(torch.abs(x)-t)

def fit_lista(w,depth,seed):
 y=w['matrices'].reshape(N,-1)@w['atoms'].reshape(K,-1).T;yt=y[:TRAIN];params={'steps':torch.nn.Parameter(torch.ones(depth)*.9),'thresholds':torch.nn.Parameter(torch.ones(depth)*-3.9)};opt=torch.optim.Adam(params.values(),lr=.03);t0=time.perf_counter()
 for _ in range(100):
  z=torch.zeros_like(yt)
  for j in range(depth):z=soft(z+params['steps'][j]*(yt-z),torch.nn.functional.softplus(params['thresholds'][j]))
  rec=z@w['atoms'].reshape(K,-1);loss=((rec-w['matrices'][:TRAIN].reshape(TRAIN,-1))**2).mean()+.0005*z.abs().mean();opt.zero_grad();loss.backward();opt.step()
 trainwall=time.perf_counter()-t0
 state={'atoms':w['atoms'],'steps':params['steps'].detach(),'thresholds':torch.nn.functional.softplus(params['thresholds'].detach()),'depth':depth}
 return state,trainwall

def encode(method,w,state=None):
 flat=w['matrices'].reshape(N,-1);A=w['atoms'].reshape(K,-1);t=time.perf_counter();proj=flat@A.T
 if method=='independent':z=w['coeff']
 elif method=='direct_top3':idx=proj.abs().topk(S,dim=1).indices;z=torch.zeros_like(proj);z.scatter_(1,idx,proj.gather(1,idx))
 elif method=='omp3':
  res=flat.clone();idxs=[];vals=[]
  for _ in range(S):scores=res@A.T;idx=scores.abs().argmax(1);val=scores[torch.arange(N),idx];res-=val[:,None]*A[idx];idxs.append(idx);vals.append(val)
  z=torch.zeros_like(proj);z.scatter_add_(1,torch.stack(idxs,1),torch.stack(vals,1))
 elif method.startswith('lista'):
  z=torch.zeros_like(proj)
  for j in range(state['depth']):z=soft(z+state['steps'][j]*(proj-z),state['thresholds'][j])
 else:z=proj
 return z,time.perf_counter()-t

def pack(arr,meta):
 b=io.BytesIO();a={k:np.asarray(v) for k,v in arr.items()};a['__schema_json__']=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);np.savez(b,**a);return b.getvalue()

def payload(method,w,state=None):
 if method=='independent':return pack({'function_matrices':w['matrices'].numpy()},{'format':'MA487-independent-function-bank-v1','functions':N,'dim':D})
 a={'shared_atoms':w['atoms'].numpy()}
 if method.startswith('lista'):a.update({'steps':state['steps'].numpy(),'thresholds':state['thresholds'].numpy()})
 return pack(a,{'format':'MA487-lista-inference-v1','method':method,'dictionary_atoms':K,'dim':D,'depth':state['depth'] if state else 0})

def evaluate(method,w):
 trainwall=0.;state=None;depth=int(method[5:]) if method.startswith('lista') else 0
 if depth:state,trainwall=fit_lista(w,depth,0)
 z,encwall=encode(method,w,state);mat=(z@w['atoms'].reshape(K,-1)).reshape(N,D,D);qt=time.perf_counter();pred=torch.stack([w['queries'][i]@mat[i].T for i in range(N)]);qwall=time.perf_counter()-qt;rmse=float(torch.sqrt(torch.mean((pred[TRAIN:]-w['outputs'][TRAIN:])**2)))
 active=float((z[TRAIN:].abs()>1e-4).sum(1).float().mean());raw=payload(method,w,state);steps=depth if depth else (3 if method=='omp3' else 1)
 if method=='independent':ops=0
 elif method=='omp3':ops=S*(N*K*D*D+N*D*D)
 elif depth:ops=N*K*D*D+depth*N*K*2
 else:ops=N*K*D*D
 qops=(N-TRAIN)*Q*D*D
 m={'heldout_function_rmse':rmse,'average_active_atoms':active,'inference_payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'optimizer_updates':100 if depth else 0,'training_examples':TRAIN*100 if depth else 0,'training_wall_s':trainwall,'encode_wall_s':encwall,'query_wall_s':qwall,'inference_steps':steps,'lista_or_router_ops_proxy':ops,'query_ops_proxy':qops,'active_ops_proxy':ops+qops}
 return m,raw,z

def run(seed,out):
 out.mkdir(parents=True,exist_ok=True);w=world(seed);res={}
 for method in METHODS:
  m,raw,z=evaluate(method,w);res[method]=m;(out/f'{method}_payload.npz').write_bytes(raw)
 doc={'experiment_id':'MA-487','seed':seed,'split':'dev','task':{'functions':N,'train':TRAIN,'heldout':N-TRAIN,'dim':D,'atoms':K,'nnz':S},'methods':res};(out/'metrics.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');return doc

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
