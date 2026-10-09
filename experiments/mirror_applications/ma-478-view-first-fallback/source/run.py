#!/usr/bin/env python3
"""Frozen MA-478 shared value view and residual-private fallback screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
D,KD,R,N,TRAIN=64,32,8,256,192
HELD=64;QPER,NLOCAL=4,1024
RADIUS=.35;RESIDUAL_L2=0.05
METHODS=('explicit','shared_only','mirror_fallback','native_fallback','int8','no_edit')

def world(seed):
 g=torch.Generator().manual_seed(seed+478);basis=torch.linalg.qr(torch.randn(D,R,generator=g),mode='reduced').Q;coeff=torch.randn(N,R,generator=g)*.5;values=coeff@basis.T
 privgen=torch.Generator().manual_seed(seed+1478)
 for e in range(TRAIN,N):
  if (e-TRAIN)%4==0:
   r=torch.randn(D,generator=privgen);r=r-basis@(basis.T@r);values[e]+=r*.12
 keys=torch.randn(N,KD,generator=torch.Generator().manual_seed(seed+2478));keys=keys/keys.norm(dim=1,keepdim=True)
 qg=torch.Generator().manual_seed(seed+3478);queries=[keys[e]+torch.randn(QPER,KD,generator=qg)*.04 for e in range(N)]
 loc=torch.randn(NLOCAL,KD,generator=torch.Generator().manual_seed(seed+4478))
 return {'teacher_basis':basis,'coeff':coeff,'values':values,'private_expected':torch.tensor([e>=TRAIN and (e-TRAIN)%4==0 for e in range(N)]),'keys':keys,'queries':queries,'locality':loc}

def route(q,keys):
 d=((keys-q.unsqueeze(0))**2).sum(-1);i=int(torch.argmin(d).item());return i,float(d[i])<=RADIUS*RADIUS

def encode_basis(w):
 _,_,vh=torch.linalg.svd(w['values'][:TRAIN],full_matrices=False);basis=vh[:R].T.contiguous();codes=w['values']@basis;recon=codes@basis.T;res=w['values']-recon;flags=res.norm(dim=1)>RESIDUAL_L2
 return basis,codes,recon,res,flags

def pack(arr,meta):
 b=io.BytesIO();d={k:np.asarray(v) for k,v in arr.items()};d['__schema_json__']=np.frombuffer(json.dumps(meta,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8);np.savez(b,**d);return b.getvalue()

def serialize(method,w,state):
 obj={'keys':w['keys'].numpy(),'route_radius':np.asarray([RADIUS],np.float32)}
 if method in ('explicit','int8'):obj.update(state['stored']);family='explicit-full-value-bank-v1' if method=='explicit' else 'per-value-int8-bank-v1'
 elif method=='shared_only':obj.update({'value_basis':state['basis'].numpy(),'value_codes':state['codes'].numpy()});family='rank8-shared-only-bank-v1'
 elif method in ('mirror_fallback','native_fallback'):
  obj.update({'value_basis':state['basis'].numpy(),'shared_indices':state['shared_indices'].numpy(),'shared_codes':state['shared_codes'].numpy(),'private_indices':state['private_indices'].numpy(),'private_values':state['private_values'].numpy(),'residual_l2_gate':np.asarray([RESIDUAL_L2],np.float32)});family='rank8-shared-private-fallback-v1'
 else:family='no-edit-bank-v1'
 return pack(obj,{'format':'MA478-private-fallback-memory-v1','method_family':family,'n_values':N,'value_dim':D,'rank':R,'dtype':'typed-arrays'})

def prepare(method,w):
 if method in ('shared_only','mirror_fallback','native_fallback'):
  start=time.perf_counter();basis,codes,shared,res,flags=encode_basis(w);fitwall=time.perf_counter()-start;encstart=time.perf_counter()
  if method=='shared_only':state={'basis':basis,'codes':codes,'recon':shared}
  else:
   pi=torch.where(flags)[0];si=torch.where(~flags)[0];pv=w['values'][pi];recon=shared.clone();recon[pi]=pv
   state={'basis':basis,'recon':recon,'shared_indices':si.to(torch.uint8),'shared_codes':codes[si],'private_indices':pi.to(torch.uint8),'private_values':pv}
  encwall=time.perf_counter()-encstart;fitops=TRAIN*D*R*3
 elif method=='explicit':
  state={'recon':w['values'],'stored':{'values':w['values'].numpy()}};fitwall=0.;encwall=0.;fitops=0;flags=None
 elif method=='int8':
  encstart=time.perf_counter();scale=w['values'].abs().amax(dim=1,keepdim=True).clamp_min(1e-8)/127;q=torch.round(w['values']/scale).clamp(-127,127).to(torch.int8)
  state={'recon':q.float()*scale,'stored':{'values_int8':q.numpy(),'value_scales':scale.numpy()}};fitwall=0.;encwall=time.perf_counter()-encstart;fitops=0;flags=None
 else:
  state={'recon':torch.zeros_like(w['values'])};fitwall=0.;encwall=0.;fitops=0;flags=None
 return state,fitwall,encwall,flags,fitops

def evaluate(method,w):
 state,fitwall,encwall,flags,fitops=prepare(method,w);raw=serialize(method,w,state);recon=state['recon'];qstart=time.perf_counter();err=[];routes=[];loc=[]
 for e in range(TRAIN,N):
  for q in w['queries'][e]:
   i,on=route(q,w['keys']);routes.append(i==e and on);v=recon[i] if on and method!='no_edit' else torch.zeros(D);err.append(float(torch.mean((v-w['values'][e])**2)))
 for q in w['locality']:loc.append(route(q,w['keys'])[1])
 qwall=time.perf_counter()-qstart
 fb=float(flags[TRAIN:].float().mean()) if method in ('mirror_fallback','native_fallback') else float('nan')
 routeops=N*KD*3;decode=D*R if method in ('shared_only','mirror_fallback','native_fallback') else D
 m={'heldout_value_rmse':float(np.sqrt(np.mean(err))),'heldout_wrong_route_rate':1-float(np.mean(routes)),'locality_false_trigger_rate':float(np.mean(loc)),'heldout_private_fallback_fraction':fb,'all_memory_private_fallback_fraction':float(flags.float().mean()) if method in ('mirror_fallback','native_fallback') else float('nan'),'private_value_count':int(flags.sum()) if method in ('mirror_fallback','native_fallback') else 0,'inference_payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'basis_fit_wall_s':fitwall,'encode_wall_s':encwall,'query_wall_s':qwall,'basis_fit_ops_proxy':fitops,'encode_ops_per_value':D*R if method in ('shared_only','mirror_fallback','native_fallback') else (D if method=='int8' else 0),'decode_ops_per_query':decode,'route_distance_scalar_ops_per_query':routeops,'active_ops_proxy':((N-TRAIN)*QPER+NLOCAL)*(routeops+decode),'examples_seen':(N-TRAIN)*QPER+NLOCAL}
 return m,raw

def run(seed,out):
 out.mkdir(parents=True,exist_ok=True);w=world(seed);res={}
 for method in METHODS:
  m,raw=evaluate(method,w);res[method]=m;(out/f'{method}_payload.npz').write_bytes(raw)
 assert (out/'mirror_fallback_payload.npz').read_bytes()==(out/'native_fallback_payload.npz').read_bytes()
 doc={'experiment_id':'MA-478','seed':seed,'split':'dev','task':{'n_values':N,'train':TRAIN,'heldout':HELD,'private_heldout':16,'rank':R,'residual_gate':RESIDUAL_L2},'methods':res};(out/'metrics.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');return doc

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
