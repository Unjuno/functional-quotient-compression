#!/usr/bin/env python3
"""MA-502 aligned logical-function count and intervention-byte scaling screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
D,K,NTRAIN,NTEST=24,4,64,128
TASK_COUNTS=(1,2,4,8,16,32,64,128,256)
METHODS=('shared_mirror','native_shared_code','task_loreft_r4','full_matrix')

def world(seed,tasks):
 g=torch.Generator().manual_seed(seed+502)
 u,_=torch.linalg.qr(torch.randn(D,K,generator=g));v,_=torch.linalg.qr(torch.randn(D,K,generator=g))
 cg=torch.Generator().manual_seed(seed+1502);codes=torch.randn(tasks,K,generator=cg)*.1
 delta=torch.stack([(u*codes[t])@v.T for t in range(tasks)])
 xs=[];xt=[]
 for t in range(tasks):
  gg=torch.Generator().manual_seed(seed+2502+t);xs.append(torch.randn(NTRAIN,D,generator=gg));xt.append(torch.randn(NTEST,D,generator=gg))
 xtr=torch.stack(xs);xte=torch.stack(xt)
 ytr=xtr+torch.einsum('tni,tji->tnj',xtr,delta);yte=xte+torch.einsum('tni,tji->tnj',xte,delta)
 est=torch.stack([torch.linalg.lstsq(xtr[t],ytr[t]-xtr[t]).solution.T for t in range(tasks)])
 return {'delta':delta,'xtr':xtr,'ytr':ytr,'xte':xte,'yte':yte,'est':est}

def fit_shared(w):
 basis_n=min(8,w['est'].shape[0]);basis=w['est'][:basis_n]
 left=sum((m@m.T for m in basis),torch.zeros(D,D));right=sum((m.T@m for m in basis),torch.zeros(D,D))
 _,u=torch.linalg.eigh(left);_,v=torch.linalg.eigh(right);u=u[:,-K:];v=v[:,-K:]
 codes=torch.stack([torch.diag(u.T@m@v) for m in w['est']])
 return {'left':u,'right':v,'codes':codes},basis_n

def fit_taskwise(w):
 ls=[];ss=[];rs=[]
 for m in w['est']:
  u,s,vh=torch.linalg.svd(m,full_matrices=False);ls.append(u[:,:K]);ss.append(s[:K]);rs.append(vh[:K].T)
 return {'left':torch.stack(ls),'singular':torch.stack(ss),'right':torch.stack(rs)}

def make_objects(method,w):
 if method in ('shared_mirror','native_shared_code'):return fit_shared(w)[0]
 if method=='task_loreft_r4':return fit_taskwise(w)
 if method=='full_matrix':return {'delta':w['est']}
 raise ValueError(method)

def pack(method,obj):
 fam='shared_subspace_coefficients' if method in ('shared_mirror','native_shared_code') else method
 tasks=obj['codes'].shape[0] if fam=='shared_subspace_coefficients' else next(iter(obj.values())).shape[0]
 schema={'format':'MA502-logical-intervention-bank-v1','method_class':fam,'hidden_dimension':D,'rank':K,'task_count':tasks}
 arr={k:v.detach().numpy().astype(np.float32) for k,v in obj.items()};arr['schema_json']=np.frombuffer(json.dumps(schema,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
 b=io.BytesIO();np.savez(b,**arr);return b.getvalue()

def apply(method,obj,x,t):
 if method in ('shared_mirror','native_shared_code'):return x+(x@obj['right']*obj['codes'][t])@obj['left'].T
 if method=='task_loreft_r4':return x+(x@obj['right'][t]*obj['singular'][t])@obj['left'][t].T
 if method=='full_matrix':return x+x@obj['delta'][t].T
 raise ValueError(method)

def score(method,obj,w):
 start=time.perf_counter();pred=torch.stack([apply(method,obj,w['xte'][t],t) for t in range(w['delta'].shape[0])]);wall=time.perf_counter()-start
 base=(w['yte']-w['xte']).square().mean().sqrt().clamp_min(1e-12);rel=float((pred-w['yte']).square().mean().sqrt()/base)
 if method in ('shared_mirror','native_shared_code'):
  ops=2*D*K+K;unique=int(torch.unique(obj['codes'],dim=0).shape[0]);causal=max(float((pred[t]-(w['xte'][t]+(w['xte'][t]@obj['right']*torch.zeros(K))@obj['left'].T)).abs().max()) for t in range(w['delta'].shape[0]))
 elif method=='task_loreft_r4':ops=2*D*K+K;unique=None;causal=0.
 else:ops=D*D;unique=None;causal=0.
 distinct=int(torch.unique(torch.round(w['est'].reshape(w['est'].shape[0],-1)*1e6),dim=0).shape[0])
 return {'relative_output_rmse':rel,'function_count':int(w['delta'].shape[0]),'unique_functions':distinct,'unique_task_codes':unique,'active_compute_proxy_per_example':ops,'active_compute_proxy_all_heldout_examples':ops*w['delta'].shape[0]*NTEST,'inference_wall_s':wall,'max_output_change_when_codes_zeroed':causal}

def run(seed,out):
 out.mkdir(parents=True,exist_ok=True);doc={'experiment_id':'MA-502','seed':seed,'split':'dev','sweep':{}}
 for tasks in TASK_COUNTS:
  w=world(seed,tasks);td=out/f'tasks_{tasks}';td.mkdir(exist_ok=True);items={};cache={}
  for method in METHODS:
   key='shared_mirror' if method=='native_shared_code' else method
   if key not in cache:
    start=time.perf_counter();obj=make_objects(key,w);fitwall=time.perf_counter()-start;raw=pack(key,obj);cache[key]=(obj,raw,fitwall)
   obj,raw,fitwall=cache[key];(td/f'{method}_payload.npz').write_bytes(raw)
   met=score(key,obj,w);met.update({'inference_payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'fit_wall_s':fitwall,'optimizer_updates':0,'calibration_examples':tasks*NTRAIN,'basis_training_task_count':min(8,tasks)})
   items[method]=met
  assert (td/'shared_mirror_payload.npz').read_bytes()==(td/'native_shared_code_payload.npz').read_bytes()
  doc['sweep'][str(tasks)]=items
 (out/'metrics.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');return doc

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
