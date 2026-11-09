#!/usr/bin/env python3
"""Oracle-generated variable-diffusion operator compression screen."""
from __future__ import annotations
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
from scipy.linalg import solve
ROOT=Path(__file__).resolve().parents[1]
SEED=73601; N=24; DOMAINS=8; FIT=6; RANK=4
CONTRASTS=(0.,.05,.1,.2,.4,.8,.3,.6); TOPKS=(0,4,8,16,32,64)

def payload(**arrays):
 b=io.BytesIO();np.savez_compressed(b,**arrays);return b.getvalue()

def make_green(rho,domain):
 x=np.linspace(0,1,N+2)[1:-1]; dx=1/(N+1)
 center=.28+.055*domain
 local=np.exp(-.5*((x-center)/.095)**2)
 kappa=1+rho*(.45*np.sin(2*np.pi*x)+.75*local)
 edges=np.empty(N+1);edges[0]=kappa[0];edges[-1]=kappa[-1]
 edges[1:-1]=2*kappa[:-1]*kappa[1:]/(kappa[:-1]+kappa[1:])
 A=np.diag((edges[:-1]+edges[1:])/dx**2)
 A+=np.diag(-edges[1:-1]/dx**2,1)+np.diag(-edges[1:-1]/dx**2,-1)
 return solve(A,np.eye(N),assume_a='pos').astype(np.float32),x,center

def nrmse(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))

def topk_sparse(r,k):
 if not k:return np.empty((0,2),np.int16),np.empty(0,np.float32),np.zeros_like(r)
 flat=np.argpartition(np.abs(r).ravel(),-k)[-k:];ij=np.stack(np.unravel_index(flat,r.shape),axis=1).astype(np.int16);v=r[ij[:,0],ij[:,1]].astype(np.float32);out=np.zeros_like(r);out[ij[:,0],ij[:,1]]=v;return ij,v,out

def run():
 rng=np.random.default_rng(SEED); xs=[]; ks=[]; centers=[]
 for d,rho in enumerate(CONTRASTS):
  k,x,c=make_green(rho,d);ks.append(k);xs.append(x);centers.append(c)
 ks=np.stack(ks); mean=ks[:FIT].mean(axis=0); centered=ks[:FIT]-mean
 u,s,vt=np.linalg.svd(centered.reshape(FIT,-1),full_matrices=False)
 basis=vt[:RANK].reshape(RANK,N,N).astype(np.float32)
 coeff=np.einsum('dij,kij->dk',ks-mean,basis).astype(np.float32)
 recon=mean[None]+np.einsum('dk,kij->dij',coeff,basis)
 # VeRA-like control: one fixed random shared matrix basis with learned per-domain scales.
 rand=rng.normal(size=(RANK,N*N));rand/=np.linalg.norm(rand,axis=1,keepdims=True);rb=rand.reshape(RANK,N,N).astype(np.float32)
 rc=np.einsum('dij,kij->dk',ks-mean,rb).astype(np.float32);rrecon=mean[None]+np.einsum('dk,kij->dij',rc,rb)
 f=rng.normal(size=(DOMAINS,256,N)).astype(np.float32); f*=np.linspace(.5,1.5,N)[None,None,:]
 methods=[]
 full=payload(green_operators=ks,domain_ids=np.arange(DOMAINS,dtype=np.int8),contrasts=np.asarray(CONTRASTS,np.float32))
 methods.append(('independent_full',ks,full,'full',0))
 sh=payload(shared_mean=mean,domain_ids=np.arange(DOMAINS,dtype=np.int8));methods.append(('shared_mean',np.broadcast_to(mean,ks.shape),sh,'mean',0))
 pc=payload(shared_mean=mean,basis=basis,coordinates=coeff,domain_ids=np.arange(DOMAINS,dtype=np.int8));methods.append(('native_pca',recon,pc,'pca',0))
 # Exact same stored parameters/functions as the native PCA method.
 methods.append(('mirror_view',recon,pc,'pca',0))
 vr=payload(shared_mean=mean,random_basis=rb,scales=rc,domain_ids=np.arange(DOMAINS,dtype=np.int8));methods.append(('vera_like_random_basis',rrecon,vr,'random',0))
 for k in TOPKS:
  private_ops=[];all_idx=[];all_vals=[];offsets=[0]
  for d in range(DOMAINS):
   ij,v,sp=topk_sparse(ks[d]-recon[d],k);private_ops.append(recon[d]+sp)
   all_idx.append(ij);all_vals.append(v);offsets.append(offsets[-1]+len(v))
  indices=np.concatenate(all_idx,axis=0) if all_idx else np.empty((0,2),np.int16)
  values=np.concatenate(all_vals) if all_vals else np.empty(0,np.float32)
  full_private=payload(shared_mean=mean,basis=basis,coordinates=coeff,
    residual_offsets=np.asarray(offsets,np.int32),residual_indices=indices,
    residual_values=values,domain_ids=np.arange(DOMAINS,dtype=np.int8),contrasts=np.asarray(CONTRASTS,np.float32))
  methods.extend([('native_pca_private',np.stack(private_ops),full_private,'private',k),
                  ('mirror_private',np.stack(private_ops),full_private,'private',k)])
 rows=[]
 for name,ops,blob,family,k in methods:
  for group,domain_list in [('fit',list(range(FIT))),('heldout',list(range(FIT,DOMAINS)))]:
   ge=[];ie=[];times=[]
   for d in domain_list:
    op=ops[d];t0=time.perf_counter();pred=f[d]@op.T;times.append((time.perf_counter()-t0)*1000)
    truth=f[d]@ks[d].T;ge.append(nrmse(pred,truth));mask=np.abs(xs[d]-centers[d])<=.12
    ie.append(nrmse(pred[:,mask],truth[:,mask]))
   rows.append(dict(seed=SEED,method=name,contrast_group=group,private_topk=k,domains=len(domain_list),serialized_bytes=len(blob),payload_sha256=hashlib.sha256(blob).hexdigest(),global_solution_nrmse=float(np.mean(ge)),interface_solution_nrmse=float(np.mean(ie)),active_multiplies=len(domain_list)*256*N*N,wall_ms=float(np.sum(times)),status_note='exact native PCA alias' if name.startswith('mirror') else ''))
 out=ROOT/'RESULTS_CORE.csv'
 with out.open('w',newline='') as fobj:
  w=csv.DictWriter(fobj,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('wrote',len(rows),'rows')
if __name__=='__main__':run()
