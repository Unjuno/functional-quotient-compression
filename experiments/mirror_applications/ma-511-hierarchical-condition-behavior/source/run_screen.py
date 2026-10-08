#!/usr/bin/env python3
"""Deterministic synthetic held-out condition x behavior composition screen."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
D,C,B=64,8,8
RANKS=(2,4,8); RHOS=(0.0,0.1,0.25)

def orth(rng,n,k):
 q,_=np.linalg.qr(rng.normal(size=(n,k)))
 return q[:,:k].astype(np.float32)

def make_world(seed):
 rng=np.random.default_rng(seed)
 qc,qb=orth(rng,D,4),orth(rng,D,4)
 ccode=rng.normal(size=(C,4)).astype(np.float32)
 bcode=rng.normal(size=(B,4)).astype(np.float32)
 c=(ccode@qc.T).astype(np.float32); b=(bcode@qb.T).astype(np.float32)
 c/=np.linalg.norm(c,axis=1,keepdims=True); b/=np.linalg.norm(b,axis=1,keepdims=True)
 private=rng.normal(size=(C,B,D)).astype(np.float32)
 private/=np.linalg.norm(private,axis=2,keepdims=True)
 targets=np.empty((C,B,D),np.float32)
 for i in range(C):
  for j in range(B): targets[i,j]=c[i]+b[j]
 # Deterministic diagonal-stripe holdout; verify every factor is visible.
 held=[(i,j) for i in range(C) for j in range(B) if (i+3*j)%4==0]
 visible=[(i,j) for i in range(C) for j in range(B) if (i,j) not in held]
 assert len(held)==16 and len(visible)==48
 assert len({i for i,j in visible})==C and len({j for i,j in visible})==B
 return dict(c=c,b=b,private=private,targets=targets,visible=np.array(visible,np.int16),held=np.array(held,np.int16))

def basis(rows,r):
 _,_,vt=np.linalg.svd(rows.astype(np.float64),full_matrices=False)
 return vt[:r].T.astype(np.float32)

def fit(world,r_c,r_b,rho):
 pairs=world['visible']; y=np.stack([world['c'][i]+world['b'][j]+rho*2**0.5*world['private'][i,j] for i,j in pairs])
 # Additive least squares on visible whole pairs only.
 x=np.zeros((len(pairs),1+C+B),np.float64); x[:,0]=1
 for k,(i,j) in enumerate(pairs): x[k,1+i]=1;x[k,1+C+j]=1
 coef=np.linalg.lstsq(x,y.astype(np.float64),rcond=None)[0].astype(np.float32)
 bias=coef[0]; cf=coef[1:1+C]; bf=coef[1+C:]
 bc,bb=basis(cf,r_c),basis(bf,r_b)
 cc,cb=cf@bc,bf@bb
 return dict(bias=bias,bc=bc,cc=cc,bb=bb,cb=cb,
             decoded_c=cc@bc.T,decoded_b=cb@bb.T)

def payload(path,kind,f,r_c,r_b,world):
 if kind in ('mirror','native_additive'):
  arr=dict(global_bias=f['bias'],condition_basis=f['bc'],condition_codes=f['cc'],
           behavior_basis=f['bb'],behavior_codes=f['cb'],schema=np.array([511,r_c,r_b],np.int32))
 elif kind=='cast_additive':
  arr=dict(global_bias=f['bias'],condition_vectors=f['decoded_c'],behavior_vectors=f['decoded_b'],
           schema=np.array([511,0,0],np.int32))
 elif kind=='full_pair':
  arr=dict(pair_function_table=world['targets'],schema=np.array([511,C,B],np.int32))
 elif kind=='no_composition':
  arr=dict(schema=np.array([511,0,0],np.int32))
 np.savez(path,**arr)
 return Path(path).stat().st_size

def eval_metrics(world,f,rho):
 pred=f['bias'][None,:]+f['decoded_c'][world['held'][:,0]]+f['decoded_b'][world['held'][:,1]]
 target=np.stack([world['c'][i]+world['b'][j]+rho*2**0.5*world['private'][i,j] for i,j in world['held']])
 err=pred-target
 rmse=float(np.linalg.norm(err)/np.linalg.norm(target))
 return dict(heldout_relative_rmse=rmse,per_pair_rmse=np.sqrt(np.mean(err**2,axis=1)).tolist(),
             unique_heldout_outputs=int(len(np.unique(np.round(pred,6),axis=0))))

def run(seed,out):
 w=make_world(seed); rows=[]; mirror_refs={}
 configs=[('mirror',r,r) for r in RANKS]+[('native_additive',r,r) for r in RANKS]+[('cast_additive',0,0),('full_pair',0,0),('no_composition',0,0)]
 for rho in RHOS:
  for kind,rc,rb in configs:
   target_world=dict(w,targets=np.stack([[w['c'][i]+w['b'][j]+rho*2**0.5*w['private'][i,j] for j in range(B)] for i in range(C)]).astype(np.float32))
   t=time.perf_counter()
   if kind in ('mirror','native_additive','cast_additive'): f=fit(w,rc or 8,rb or 8,rho)
   else: f=fit(w,8,8,rho)
   fit_s=time.perf_counter()-t
   key=f'{kind}_c{rc}_b{rb}_rho{rho:g}'
   nbytes=payload(Path(out)/(key+'.npz'),kind,f,rc,rb,target_world)
   t=time.perf_counter()
   if kind=='full_pair':
    pred=target_world['targets'][w['held'][:,0],w['held'][:,1]]
    metrics=dict(heldout_relative_rmse=0.0,per_pair_rmse=[0.0]*len(pred),unique_heldout_outputs=len(np.unique(np.round(pred,6),axis=0)))
   elif kind=='no_composition':
    metrics=dict(heldout_relative_rmse=1.0,per_pair_rmse=[],unique_heldout_outputs=0)
   else: metrics=eval_metrics(w,f,rho)
   infer_s=time.perf_counter()-t
   ops=(2*D*rc+D) if kind in ('mirror','native_additive') else (D if kind=='full_pair' else (2*D+D if kind=='cast_additive' else 0))
   row=dict(seed=seed,rho=rho,key=key,method=kind,condition_rank=rc,behavior_rank=rb,bytes=nbytes,
            train_pairs=48,optimizer_updates=0,ops_per_pair=ops,fit_seconds=fit_s,inference_seconds=infer_s,
            metrics=metrics,native_alias=False)
   if kind=='mirror': mirror_refs[(rho,rc)]=row
   if kind=='native_additive':
    ref=mirror_refs[(rho,rc)]
    row['native_alias']=row['bytes']==ref['bytes'] and row['metrics']==ref['metrics']
   rows.append(row)
 return rows,w

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 p=Path(a.out);p.mkdir(parents=True,exist_ok=True)
 t=time.perf_counter();rows,w=run(a.seed,p)
 np.savez(p/'split_manifest.npz',visible_pairs=w['visible'],heldout_pairs=w['held'])
 report=dict(experiment_id='MA-511',seed=a.seed,fit_and_eval_seconds=time.perf_counter()-t,rows=rows)
 (p/'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'seed':a.seed,'rows':len(rows),'heldout_pairs':len(w['held']),'seconds':report['fit_and_eval_seconds']},indent=2))
if __name__=='__main__':main()
