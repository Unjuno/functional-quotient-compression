#!/usr/bin/env python3
"""Learn a Toeplitz atom bank from support tasks, sparsify codes, then prune atoms."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import orthogonal_mp
ROOT=Path(__file__).resolve().parents[1]
D,N,M,K,SUPPORT,AUDIT=32,24,32,2,128,128
SEEDS={'dev-71001':71001}  # Audit IDs were exposed by the invalid fixed-bank attempt and are intentionally unavailable.
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
def nrmse(P,Y):return float(np.linalg.norm(P-Y)/max(np.linalg.norm(Y),1e-30))
def op_fft():return int(6*D*np.log2(2*D))
def learn_dictionary(task_gens,ncomp,seed):
 model=DictionaryLearning(n_components=ncomp,alpha=0.1,max_iter=500,fit_algorithm='lars',transform_algorithm='omp',transform_n_nonzero_coefs=K,tol=1e-8,random_state=seed)
 t=time.perf_counter();codes=model.fit_transform(task_gens);elapsed=time.perf_counter()-t
 return model.components_,codes,elapsed,model.n_iter_
def omp_codes(bank,targets):
 return np.stack([orthogonal_mp(bank.T,t,n_nonzero_coefs=K) for t in targets])
def encode_sparse(bank,codes):
 ids=np.zeros((len(codes),K),dtype=np.int16);vals=np.zeros((len(codes),K),dtype=np.float32)
 for n,c in enumerate(codes):
  ix=np.flatnonzero(np.abs(c)>1e-10)
  if len(ix)>K:ix=ix[np.argsort(np.abs(c[ix]))[-K:]]
  if len(ix)<K:
   order=np.argsort(np.abs(c))[-K:];ix=order
  ids[n]=ix;vals[n]=c[ix]
 return ids,vals
def serialize(method,bank,ids,vals,dense,meta):
 if method=='independent_dense':return pack(matrices=dense.astype(np.float32),metadata=meta)
 if method=='independent_toeplitz':return pack(generators=bank.astype(np.float32),metadata=meta)
 if method=='fullbank_dense_codes':return pack(generators=bank.astype(np.float32),coefficients=vals.astype(np.float32),metadata=meta)
 return pack(generators=bank.astype(np.float32),indices=ids.astype(np.int16),coefficients=vals.astype(np.float32),metadata=meta)
def score(method,payload,Xs,Ys):
 errs=[]
 with np.load(io.BytesIO(payload),allow_pickle=False) as p:
  for n,(X,Y) in enumerate(zip(Xs,Ys)):
   if method=='independent_dense':A=p['matrices'][n]
   elif method=='independent_toeplitz':A=matrix_from_gen(p['generators'][n])
   elif method=='fullbank_dense_codes':A=np.einsum('m,mij->ij',p['coefficients'][n],all_matrices(p['generators']))
   else:
    ids=p['indices'][n].astype(int);co=p['coefficients'][n]
    A=sum((float(a)*matrix_from_gen(p['generators'][int(i)]) for i,a in zip(ids,co)),start=np.zeros((D,D)))
   errs.append(nrmse(X@A.T,Y))
 return errs
def run_one(label,seed,outdir,rhos=RHO_LEVELS):
 rng=np.random.default_rng(seed);outdir.mkdir(parents=True,exist_ok=True);rows=[]
 # Six unknown teacher atoms generate task operators; learner sees only task support IO pairs.
 teacher=rng.normal(size=(6,2*D-1));teacher/=np.linalg.norm(teacher,axis=1,keepdims=True)
 for rho in rhos:
  Xtraces=[];Ytraces=[];Xtests=[];Ytests=[];truth=[];task_fit_s=[];dense_fit_s=[];dense_fits=[]
  for n in range(N):
   chosen=rng.choice(6,size=K,replace=False);weights=rng.normal(size=K);g=weights@teacher[chosen]
   if rho:
    priv=rng.normal(size=2*D-1);priv*=rho*np.linalg.norm(g)/max(np.linalg.norm(priv),1e-30);g=g+priv
   A=matrix_from_gen(g);Xtr=rng.normal(size=(SUPPORT,D));Xte=rng.normal(size=(AUDIT,D));Ytr=Xtr@A.T;Yte=Xte@A.T
   Xtraces.append(Xtr);Ytraces.append(Ytr);Xtests.append(Xte);Ytests.append(Yte)
   t=time.perf_counter();gfit=fit_coeff(Xtr,Ytr,all_matrices(np.eye(2*D-1)));task_fit_s.append(time.perf_counter()-t);truth.append(gfit)
   t=time.perf_counter();dense_fits.append(fit_dense(Xtr,Ytr));dense_fit_s.append(time.perf_counter()-t)
  task_gens=np.stack(truth);dense_fits=np.stack(dense_fits)
  # Learn the overcomplete bank from support-derived task Toeplitz generators.
  bank,learned_codes,dict_wall,dict_iters=learn_dictionary(task_gens,M,seed+int(rho*1000))
  sparse_ids,sparse_vals=encode_sparse(bank,learned_codes)
  used=np.unique(sparse_ids);remap={int(old):i for i,old in enumerate(used)};pruned_ids=np.vectorize(remap.__getitem__)(sparse_ids)
  # Fit equal-size random-pruned and small-from-start dictionaries from support tasks.
  random_sel=np.sort(rng.choice(M,size=len(used),replace=False));random_bank=bank[random_sel]
  random_codes=omp_codes(random_bank,task_gens);random_ids,random_vals=encode_sparse(random_bank,random_codes)
  small_bank,small_codes,small_wall,small_iters=learn_dictionary(task_gens,len(used),seed+10000+int(rho*1000));small_ids,small_vals=encode_sparse(small_bank,small_codes)
  # Dense least-squares codes over the full learned bank provide a non-sparse upper control.
  dense_codes=np.stack([np.linalg.lstsq(bank.T,t,rcond=None)[0] for t in task_gens])
  methods={
   'mirror_sparse_fullbank':(bank,sparse_ids,sparse_vals,bank.shape[0],dict_wall,dict_iters),
   'native_sparse_coding':(bank,sparse_ids,sparse_vals,bank.shape[0],dict_wall,dict_iters),
   'postfit_pruned':(bank[used],pruned_ids,sparse_vals,len(used),dict_wall,dict_iters),
   'random_pruned':(random_bank,random_ids,random_vals,len(random_sel),dict_wall,dict_iters),
   'small_from_start':(small_bank,small_ids,small_vals,len(small_bank),small_wall,small_iters),
   'fullbank_dense_codes':(bank,None,dense_codes,bank.shape[0],dict_wall,dict_iters),
   'independent_toeplitz':(task_gens,None,None,0,sum(task_fit_s),0),
   'independent_dense':(None,None,None,0,sum(dense_fit_s),0),
  }
  for method,(b,ids,vals,retained,fitwall,iters) in methods.items():
   if method=='independent_dense':payload=serialize(method,None,None,None,dense_fits,np.array([D,N],np.int16))
   elif method=='independent_toeplitz':payload=serialize(method,task_gens,None,None,None,np.array([D,N],np.int16))
   else:payload=serialize(method,b,ids,vals,None,np.array([D,N,len(b),K],np.int16))
   errs=score(method,payload,Xtests,Ytests);(outdir/f'{label}_rho{rho:g}.{method}.npz').write_bytes(payload)
   setup=int(retained*N*(2*D-1)) if method in ('postfit_pruned','random_pruned','small_from_start') else 0
   rows.append(dict(condition=label,seed=seed,private_fraction=rho,method=method,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),mean_heldout_nrmse=float(np.mean(errs)),max_task_heldout_nrmse=float(np.max(errs)),atoms_retained=retained,examples=N*(SUPPORT+AUDIT),dictionary_iterations=iters,optimizer_updates=0,fit_wall_s=fitwall,apply_ops_proxy=op_fft(),one_time_compose_ops_proxy=setup,amortized_ops_per_query=op_fft()+setup//(N*AUDIT),status_note='learned Toeplitz atom bank from support-only task generators; analytic FFT proxy'))
 return rows
def main():
 rows=[]
 for label,seed in SEEDS.items():rows.extend(run_one(label,seed,ROOT/'source'/'payloads'))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('wrote',len(rows),'model-level rows')
if __name__=='__main__':main()
