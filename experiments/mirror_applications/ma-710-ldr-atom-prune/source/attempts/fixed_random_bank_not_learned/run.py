#!/usr/bin/env python3
"""Sparse Toeplitz atom fitting, post-fit pruning and actual model-payload accounting."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D,N,M,K,SUPPORT,AUDIT=32,24,32,2,128,128
SEEDS={'dev-71001':71001,'audit-71011':71011,'audit-71012':71012,'audit-71013':71013}
RHO_LEVELS=(0.0,0.05,0.2)
OFFSETS=np.arange(-D+1,D)
def pack(**arr):
 b=io.BytesIO();np.savez(b,**arr);return b.getvalue()
def matrix_from_gen(g):
 A=np.zeros((D,D),dtype=np.float64)
 for z,off in zip(g,OFFSETS):
  if off>=0:
   i=np.arange(off,D);A[i,i-off]=z
  else:
   j=np.arange(-off,D);A[j+off,j]=z
 return A
def all_matrices(gens):return np.stack([matrix_from_gen(g) for g in gens])
def fit_coeff(X,Y,A):
 F=np.stack([X@a.T for a in A],axis=-1).reshape(-1,A.shape[0])
 return np.linalg.lstsq(F,Y.reshape(-1),rcond=None)[0]
def fit_dense(X,Y):return np.linalg.lstsq(X,Y,rcond=None)[0].T
def fit_toeplitz(X,Y,basis):
 return fit_coeff(X,Y,basis)
def normalize_error(P,Y):return float(np.linalg.norm(P-Y)/max(np.linalg.norm(Y),1e-30))
def op_fft():return int(6*D*np.log2(2*D))
def aggregate_payload(method,gens,coeff,ids,matrices,meta):
 if method in ('mirror_sparse_fullbank','native_sparse_coding'):
  return pack(generators=gens.astype(np.float32),indices=ids.astype(np.int16),coefficients=coeff.astype(np.float32),metadata=meta)
 if method=='postfit_pruned':
  return pack(generators=gens.astype(np.float32),indices=ids.astype(np.int16),coefficients=coeff.astype(np.float32),metadata=meta)
 if method in ('random_pruned','small_from_start'):
  return pack(generators=gens.astype(np.float32),indices=ids.astype(np.int16),coefficients=coeff.astype(np.float32),metadata=meta)
 if method=='fullbank_dense_codes':
  return pack(generators=gens.astype(np.float32),coefficients=coeff.astype(np.float32),metadata=meta)
 if method=='independent_toeplitz':
  return pack(generators=gens.astype(np.float32),metadata=meta)
 if method=='independent_dense':
  return pack(matrices=matrices.astype(np.float32),metadata=meta)
 raise KeyError(method)
def score_payload(method,payload,Xs,Ys):
 errs=[]
 with np.load(io.BytesIO(payload),allow_pickle=False) as p:
  for n,(X,Y) in enumerate(zip(Xs,Ys)):
   if method in ('mirror_sparse_fullbank','native_sparse_coding','postfit_pruned','random_pruned','small_from_start'):
    G=p['generators'];ix=p['indices'][n].astype(int);co=p['coefficients'][n]
    mat=sum((float(a)*matrix_from_gen(G[int(i)]) for i,a in zip(ix,co)),start=np.zeros((D,D)))
   elif method=='fullbank_dense_codes':
    G=p['generators'];co=p['coefficients'][n];mat=np.einsum('m,mij->ij',co,all_matrices(G))
   elif method=='independent_toeplitz':mat=matrix_from_gen(p['generators'][n])
   else:mat=p['matrices'][n]
   errs.append(normalize_error(X@mat.T,Y))
 return errs
def run_one(label,seed,outdir):
 rng=np.random.default_rng(seed);outdir.mkdir(parents=True,exist_ok=True);rows=[]
 for rho in RHO_LEVELS:
  # Overcomplete learned-bank stand-in: fixed sampled Toeplitz atoms; first six are the teacher support.
  G=rng.normal(size=(M,2*D-1)).astype(np.float64)/np.sqrt(2*D-1)
  A=all_matrices(G);true_active=6
  true_tasks=[];Xs=[];Ys=[];Xtraces=[];Ytraces=[];fullcodes=[];densetasks=[]
  for n in range(N):
   chosen=rng.choice(true_active,size=K,replace=False);weights=rng.normal(size=K)
   code=np.zeros(M);code[chosen]=weights
   W=np.einsum('m,mij->ij',code,A)
   if rho:
    # Private Toeplitz perturbation sets an off-orbit task component.
    private=rng.normal(size=2*D-1);private*=rho*np.linalg.norm(W)/max(np.linalg.norm(matrix_from_gen(private)),1e-30)
    W += matrix_from_gen(private)
   Xtr=rng.normal(size=(SUPPORT,D));Xte=rng.normal(size=(AUDIT,D))
   Ytr=Xtr@W.T;Yte=Xte@W.T
   t=time.perf_counter();c=fit_coeff(Xtr,Ytr,A);fit_s=time.perf_counter()-t
   fullcodes.append(c);true_tasks.append(W);Xtraces.append(Xtr);Ytraces.append(Ytr);Xs.append(Xte);Ys.append(Yte);densetasks.append(fit_dense(Xtr,Ytr))
  fullcodes=np.stack(fullcodes);true_tasks=np.stack(true_tasks);densetasks=np.stack(densetasks)
  # Sparse View code is top-2 coefficients from the support-fitted shared bank.
  sparse_ids=np.argsort(np.abs(fullcodes),axis=1)[:,-K:][:,::-1]
  sparse_vals=np.take_along_axis(fullcodes,sparse_ids,axis=1)
  used=np.unique(sparse_ids)
  remap={int(old):i for i,old in enumerate(used)}
  pruned_ids=np.vectorize(remap.__getitem__)(sparse_ids)
  # Random pruning and small-from-start use an equal-size random subset, then fit the same sparse code budget.
  random_ids=np.sort(rng.choice(M,size=len(used),replace=False));Ar=A[random_ids]
  rand_code=[]
  for X,Y in zip(Xtraces,Ytraces):
   c=fit_coeff(X,Y,Ar);top=np.argsort(np.abs(c))[-K:][::-1];rand_code.append((top,c[top]))
  random_idx=np.stack([x[0] for x in rand_code]);random_val=np.stack([x[1] for x in rand_code])
  # Independent Toeplitz fit on support; use generator basis with one unit diagonal per offset.
  basis_g=np.eye(2*D-1);basis_A=all_matrices(basis_g)
  indep_g=[];indep_dense=[]
  for X,Y in zip(Xtraces,Ytraces):
   indep_g.append(fit_toeplitz(X,Y,basis_A));indep_dense.append(fit_dense(X,Y))
  indep_g=np.stack(indep_g);indep_dense=np.stack(indep_dense)
  full_methods={
   'mirror_sparse_fullbank':(G,sparse_vals,sparse_ids),
   'native_sparse_coding':(G,sparse_vals,sparse_ids),
   'postfit_pruned':(G[used],sparse_vals,pruned_ids),
   'random_pruned':(G[random_ids],random_val,random_idx),
   'small_from_start':(G[random_ids],random_val,random_idx),
   'fullbank_dense_codes':(G,fullcodes,None),
   'independent_toeplitz':(indep_g,None,None),
   'independent_dense':(None,None,None),
  }
  for method,(gens,co,ids) in full_methods.items():
   if method=='independent_dense':payload=pack(matrices=indep_dense.astype(np.float32),metadata=np.array([D,N],np.int16))
   elif method=='independent_toeplitz':payload=pack(generators=indep_g.astype(np.float32),metadata=np.array([D,N],np.int16))
   elif method=='fullbank_dense_codes':payload=pack(generators=G.astype(np.float32),coefficients=co.astype(np.float32),metadata=np.array([D,N,M],np.int16))
   else:payload=aggregate_payload(method,gens,co,ids,None,np.array([D,N,len(gens),K],np.int16))
   errs=score_payload(method,payload,Xs,Ys)
   path=outdir/f'{label}_rho{rho:g}.{method}.npz';path.write_bytes(payload)
   if method in ('mirror_sparse_fullbank','native_sparse_coding','postfit_pruned','random_pruned','small_from_start'):
    setup=len(gens)*N*(2*D-1) if method=='postfit_pruned' else len(gens)*N*(2*D-1)
    nactive=len(gens)
   elif method=='fullbank_dense_codes':setup=M*N*(2*D-1);nactive=M
   else:setup=0;nactive=0
   apply=op_fft();active=apply+setup//max(N*AUDIT,1)
   rows.append(dict(condition=label,seed=seed,private_fraction=rho,method=method,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),mean_heldout_nrmse=float(np.mean(errs)),max_task_heldout_nrmse=float(np.max(errs)),atoms_retained=nactive,active_task_codes=K,examples=N*(SUPPORT+AUDIT),optimizer_updates=0,fit_wall_s=0.0,apply_ops_proxy=apply,one_time_compose_ops_proxy=setup,amortized_ops_per_query=active,status_note='synthetic Toeplitz atom span; support-fit; analytic FFT/apply proxy only'))
 return rows
def main():
 rows=[]
 for label,seed in SEEDS.items():rows.extend(run_one(label,seed,ROOT/'source'/'payloads'))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('wrote',len(rows),'model-level rows')
if __name__=='__main__':main()
