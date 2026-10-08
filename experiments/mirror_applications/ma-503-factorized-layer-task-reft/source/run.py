#!/usr/bin/env python3
"""MA-503 factorized layer-task coefficient completion screen."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np,torch
D,K,L,T,R,NTR,NTE=24,4,4,8,2,64,128
RHOS=(0.0,0.25);STEPS=3000;LR=.03
METHODS=('factor_mirror','native_bilinear','dense_pair_table','additive_main_effect','independent_seen_pair','full_matrix_oracle')

def mask():return torch.tensor([[(l+t)%4!=0 for t in range(T)] for l in range(L)],dtype=torch.bool)

def world(seed,rho):
 g=torch.Generator().manual_seed(seed+503);u,_=torch.linalg.qr(torch.randn(D,K,generator=g));v,_=torch.linalg.qr(torch.randn(D,K,generator=g))
 ag=torch.Generator().manual_seed(seed+1503);bg=torch.Generator().manual_seed(seed+2503);pg=torch.Generator().manual_seed(seed+3503)
 a=torch.randn(L,K,R,generator=ag)*.15;b=torch.randn(T,K,R,generator=bg)*.15;private=torch.randn(L,T,K,generator=pg)*.0225*rho
 codes=torch.einsum('lkr,tkr->ltk',a,b)+private
 delta=torch.stack([torch.stack([(u*codes[l,t])@v.T for t in range(T)]) for l in range(L)])
 trgen=torch.Generator().manual_seed(seed+4503);tegen=torch.Generator().manual_seed(seed+5503)
 xtr=torch.randn(L,T,NTR,D,generator=trgen);xte=torch.randn(L,T,NTE,D,generator=tegen)
 ytr=xtr+torch.einsum('ltni,ltji->ltnj',xtr,delta);yte=xte+torch.einsum('ltni,ltji->ltnj',xte,delta)
 m=mask();est=torch.zeros_like(delta)
 for l in range(L):
  for t in range(T):
   if m[l,t]:est[l,t]=torch.linalg.lstsq(xtr[l,t],ytr[l,t]-xtr[l,t]).solution.T
 return {'u':u,'v':v,'a_true':a,'b_true':b,'codes_true':codes,'delta':delta,'xtr':xtr,'ytr':ytr,'xte':xte,'yte':yte,'est':est,'mask':m}

def fit_basis(w):
 start=time.perf_counter();items=w['est'][w['mask']]
 left=sum((x@x.T for x in items),torch.zeros(D,D));right=sum((x.T@x for x in items),torch.zeros(D,D))
 _,u=torch.linalg.eigh(left);_,v=torch.linalg.eigh(right);u=u[:,-K:];v=v[:,-K:]
 observed=torch.zeros(L,T,K)
 for l in range(L):
  for t in range(T):
   if w['mask'][l,t]:observed[l,t]=torch.diag(u.T@w['est'][l,t]@v)
 return u,v,observed,time.perf_counter()-start

def fit_factor(observed,trainmask,seed):
 start=time.perf_counter();g=torch.Generator().manual_seed(seed+6503)
 a=torch.nn.Parameter(torch.randn(L,K,R,generator=g)*.18);b=torch.nn.Parameter(torch.randn(T,K,R,generator=g)*.18);opt=torch.optim.Adam([a,b],lr=LR)
 scale=observed[trainmask].std().clamp_min(1e-6)
 for _ in range(STEPS):
  pred=torch.einsum('lkr,tkr->ltk',a,b);loss=((pred-observed)[trainmask]/scale).square().mean();opt.zero_grad();loss.backward();opt.step()
 return {'layer_factors':a.detach(),'task_factors':b.detach()},time.perf_counter()-start

def fit_additive(observed,trainmask):
 rows=[];vals=[]
 for l in range(L):
  for t in range(T):
   if trainmask[l,t]:
    z=torch.zeros(L+T);z[l]=1;z[L+t]=1;rows.append(z);vals.append(observed[l,t])
 x=torch.stack(rows);y=torch.stack(vals);coef=torch.linalg.lstsq(x,y).solution
 return {'layer_effect':coef[:L],'task_effect':coef[L:]}

def fit_independent(w):
 pairs=[];left=[];sing=[];right=[]
 for l in range(L):
  for t in range(T):
   if w['mask'][l,t]:
    u,s,vh=torch.linalg.svd(w['est'][l,t],full_matrices=False);pairs.append((l,t));left.append(u[:,:K]);sing.append(s[:K]);right.append(vh[:K].T)
 return {'pair_ids':torch.tensor(pairs,dtype=torch.uint8),'left':torch.stack(left),'singular':torch.stack(sing),'right':torch.stack(right)}

def objects(method,w,seed):
 u,v,obs,_=fit_basis(w)
 if method in ('factor_mirror','native_bilinear'):
  f,_=fit_factor(obs,w['mask'],seed);return {'left':u,'right':v,**f}
 if method=='dense_pair_table':return {'left':u,'right':v,'codes':obs,'train_mask':w['mask']}
 if method=='additive_main_effect':return {'left':u,'right':v,**fit_additive(obs,w['mask'])}
 if method=='independent_seen_pair':return fit_independent(w)
 if method=='full_matrix_oracle':return {'delta':w['delta']}
 raise ValueError(method)

def pack(method,obj):
 family='native_bilinear_layer_task_coefficients' if method in ('factor_mirror','native_bilinear') else method
 arr={k:v.detach().cpu().numpy().astype(np.uint8 if k in ('pair_ids','train_mask') else np.float32) for k,v in obj.items()}
 arr['schema_json']=np.frombuffer(json.dumps({'format':'MA503-factorized-ReFT-v1','method_class':family,'D':D,'K':K,'layers':L,'tasks':T,'factor_rank':R},sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
 bio=io.BytesIO();np.savez(bio,**arr);return bio.getvalue()

def code(method,obj,l,t):
 if method in ('factor_mirror','native_bilinear'):return torch.einsum('kr,kr->k',obj['layer_factors'][l],obj['task_factors'][t])
 if method=='dense_pair_table':return obj['codes'][l,t] if bool(obj['train_mask'][l,t]) else torch.zeros(K)
 if method=='additive_main_effect':return obj['layer_effect'][l]+obj['task_effect'][t]
 raise ValueError(method)

def apply(method,obj,w,l,t):
 x=w['xte'][l,t]
 if method in ('factor_mirror','native_bilinear','dense_pair_table','additive_main_effect'):
  m=code(method,obj,l,t);return x+(x@obj['right']*m)@obj['left'].T
 if method=='independent_seen_pair':
  match=((obj['pair_ids'][:,0]==l)&(obj['pair_ids'][:,1]==t)).nonzero().flatten()
  if not len(match):return x
  i=int(match[0]);return x+(x@obj['right'][i]*obj['singular'][i])@obj['left'][i].T
 if method=='full_matrix_oracle':return x+x@obj['delta'][l,t].T
 raise ValueError(method)

def score(method,obj,w):
 pred=torch.empty_like(w['yte']);start=time.perf_counter()
 for l in range(L):
  for t in range(T):pred[l,t]=apply(method,obj,w,l,t)
 infer=time.perf_counter()-start;err=pred-w['yte'];signal=(w['yte']-w['xte']).square().mean().sqrt().clamp_min(1e-12);m=w['mask'];held=~m
 held_rel=float(err[held].square().mean().sqrt()/((w['yte'][held]-w['xte'][held]).square().mean().sqrt().clamp_min(1e-12)))
 train_rel=float(err[m].square().mean().sqrt()/((w['yte'][m]-w['xte'][m]).square().mean().sqrt().clamp_min(1e-12)))
 if method in ('factor_mirror','native_bilinear'):
  ops=2*D*K+K*R;predcodes=torch.einsum('lkr,tkr->ltk',obj['layer_factors'],obj['task_factors'])
  targetcodes=torch.stack([torch.stack([torch.diag(obj['left'].T@w['delta'][l,t]@obj['right']) for t in range(T)]) for l in range(L)])
  codeerr=float((predcodes[held]-targetcodes[held]).square().mean().sqrt()/targetcodes[held].square().mean().sqrt().clamp_min(1e-12))
 elif method in ('dense_pair_table','additive_main_effect'):ops=2*D*K+K;codeerr=None
 elif method=='independent_seen_pair':ops=2*D*K+K;codeerr=None
 else:ops=D*D;codeerr=0.
 if method in ('factor_mirror','native_bilinear'):
  zero=torch.zeros_like(pred)
  for l in range(L):
   for t in range(T):
    mcode=code(method,obj,l,t);zero[l,t]=w['xte'][l,t]+(w['xte'][l,t]@obj['right']*torch.zeros_like(mcode))@obj['left'].T
  causal=float((pred[held]-zero[held]).abs().max());unique=int(torch.unique(predcodes[held],dim=0).shape[0])
 else:causal=0.;unique=None
 return {'heldout_pair_relative_output_rmse':held_rel,'visible_pair_relative_output_rmse':train_rel,'heldout_code_relative_rmse':codeerr,'heldout_pair_count':int(held.sum()),'unique_heldout_functions':unique,'active_compute_proxy_per_example':ops,'active_compute_proxy_all_pairs':ops*L*T*NTE,'max_code_causal_output_change':causal,'inference_wall_s':infer}

def run(seed,out):
 out.mkdir(parents=True,exist_ok=True);doc={'experiment_id':'MA-503','seed':seed,'split':'dev','rhos':{}}
 for rho in RHOS:
  w=world(seed,rho);rd=out/f'rho_{rho:g}';rd.mkdir(exist_ok=True);items={};cache={}
  for method in METHODS:
   key='factor_mirror' if method=='native_bilinear' else method
   if key not in cache:
    start=time.perf_counter();obj=objects(key,w,seed);fitwall=time.perf_counter()-start;raw=pack(key,obj)
    if key=='factor_mirror':cache[key]=(obj,raw,fitwall)
   else:obj,raw,fitwall=cache[key]
   if key!='factor_mirror':obj,raw,fitwall=cache[key]
   (rd/f'{method}_payload.npz').write_bytes(raw)
   met=score(key,obj,w);met.update({'inference_payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'fit_wall_s':fitwall,'optimizer_updates':STEPS if key=='factor_mirror' else 0,'visible_calibration_examples':int(w['mask'].sum())*NTR})
   items[method]=met
  assert (rd/'factor_mirror_payload.npz').read_bytes()==(rd/'native_bilinear_payload.npz').read_bytes()
  doc['rhos'][str(rho)]=items
 (out/'metrics.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');return doc

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.seed,a.out),sort_keys=True))
if __name__=='__main__':main()
