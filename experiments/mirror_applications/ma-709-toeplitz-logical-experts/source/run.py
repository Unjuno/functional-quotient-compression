#!/usr/bin/env python3
"""Synthetic circular-convolution task screen; every method is replayed from bytes."""
import csv, hashlib, io, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D,N,SUPPORT,AUDIT=64,16,128,128
SEEDS={'dev-70901':70901,'audit-70911':70911,'audit-70912':70912,'audit-70913':70913}
RHOS=(0.0,0.05,0.2)
def pack(**arrays):
 b=io.BytesIO();np.savez(b,**arrays);return b.getvalue()
def apply_kernel(h,x):
 y=np.zeros_like(x,dtype=np.float64)
 for j,v in enumerate(h): y += v*np.roll(x,j,axis=-1)
 return y
def fit_toeplitz(X,Y):
 A=np.stack([np.roll(X,j,axis=1).reshape(-1) for j in range(D)],axis=1)
 return np.linalg.lstsq(A,Y.reshape(-1),rcond=None)[0]
def fit_mirror(X,Y,base):
 best=None
 for s in range(D):
  f=apply_kernel(np.roll(base,s),X);den=float(np.sum(f*f));a=float(np.sum(f*Y)/max(den,1e-30));err=float(np.sum((Y-a*f)**2))
  if best is None or err<best[0]: best=(err,s,a)
 return best[1],best[2]
def fit_dense(X,Y): return np.linalg.lstsq(X,Y,rcond=None)[0].T
def dct_basis():
 i=np.arange(D)[:,None];k=np.arange(D)[None,:]
 U=np.cos(np.pi*(2*i+1)*k/(2*D));U[:,0]*=1/np.sqrt(2);return U*np.sqrt(2/D)
def fit_dct_diag(X,Y,U):
 Z=X@U;A=(Z[:,None,:]*U[None,:,:]).reshape(-1,D)
 return np.linalg.lstsq(A,Y.reshape(-1),rcond=None)[0]
def nrmse(pred,target): return float(np.linalg.norm(pred-target)/max(np.linalg.norm(target),1e-30))
def op_count(method):
 if method in ('mirror_shared_shift_gain','native_shared_shift_gain','independent_toeplitz'): return int(3*D*np.log2(D))
 if method=='dct_diagonal': return int(2*D*np.log2(D)+D)
 if method=='hard_tied': return int(3*D*np.log2(D))
 return D*D
def run_one(label,seed,outdir):
 rng=np.random.default_rng(seed);outdir.mkdir(parents=True,exist_ok=True);rows=[];U=dct_basis()
 base=rng.normal(size=D).astype(np.float32);base/=np.linalg.norm(base)
 for rho in (0.0,0.05,0.2):
  Xtests=[];Ytests=[];toe=[];dense=[];shifts=[];gains=[];dcts=[]
  fit_times={'mirror_shared_shift_gain':0.0,'independent_toeplitz':0.0,'independent_dense':0.0,'dct_diagonal':0.0}
  for task in range(N):
   shift=int(rng.integers(D));gain=float(rng.uniform(.7,1.3));private=rng.normal(size=D);private*=np.linalg.norm(base)*rho/max(np.linalg.norm(private),1e-30)
   h=(gain*np.roll(base,shift)+private).astype(np.float64)
   Xtr=rng.normal(size=(SUPPORT,D));Xte=rng.normal(size=(AUDIT,D));Ytr=apply_kernel(h,Xtr);Yte=apply_kernel(h,Xte)
   t=time.perf_counter();sh,ga=fit_mirror(Xtr,Ytr,base);fit_times['mirror_shared_shift_gain']+=time.perf_counter()-t
   t=time.perf_counter();ht=fit_toeplitz(Xtr,Ytr);fit_times['independent_toeplitz']+=time.perf_counter()-t
   t=time.perf_counter();Bd=fit_dense(Xtr,Ytr);fit_times['independent_dense']+=time.perf_counter()-t
   t=time.perf_counter();ad=fit_dct_diag(Xtr,Ytr,U);fit_times['dct_diagonal']+=time.perf_counter()-t
   shifts.append(sh);gains.append(ga);toe.append(ht);dense.append(Bd);dcts.append(ad);Xtests.append(Xte);Ytests.append(Yte)
  shifts=np.asarray(shifts,np.int16);gains=np.asarray(gains,np.float32);toe=np.asarray(toe,np.float32);dense=np.asarray(dense,np.float32);dcts=np.asarray(dcts,np.float32)
  payloads={'mirror_shared_shift_gain':pack(base=base,shift=shifts,gain=gains,metadata=np.array([D,N],np.int16)),'native_shared_shift_gain':pack(base=base,shift=shifts,gain=gains,metadata=np.array([D,N],np.int16)),'independent_toeplitz':pack(kernel=toe,metadata=np.array([D,N],np.int16)),'independent_dense':pack(matrix=dense,metadata=np.array([D,N],np.int16)),'hard_tied':pack(base=base,metadata=np.array([D,N],np.int16)),'dct_diagonal':pack(diagonal=dcts,basis=U.astype(np.float32),metadata=np.array([D,N],np.int16))}
  for method,payload in payloads.items():
   errs=[]
   with np.load(io.BytesIO(payload),allow_pickle=False) as q:
    for n,(Xte,Yte) in enumerate(zip(Xtests,Ytests)):
     if method in ('mirror_shared_shift_gain','native_shared_shift_gain'): pred=apply_kernel(np.roll(q['base'],int(q['shift'][n]))*float(q['gain'][n]),Xte)
     elif method=='independent_toeplitz': pred=apply_kernel(q['kernel'][n],Xte)
     elif method=='independent_dense': pred=Xte@q['matrix'][n].T
     elif method=='hard_tied': pred=apply_kernel(q['base'],Xte)
     else: pred=(Xte@q['basis']*q['diagonal'][n])@q['basis'].T
     errs.append(nrmse(pred,Yte))
   path=outdir/f'{label}_rho{rho:g}.{method}.npz';path.write_bytes(payload)
   rows.append(dict(condition=label,seed=seed,private_kernel_fraction=rho,logical_tasks=N,method=method,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),mean_heldout_nrmse=float(np.mean(errs)),max_task_heldout_nrmse=float(np.max(errs)),examples=N*(SUPPORT+AUDIT),optimizer_updates=0,fit_wall_s=fit_times.get(method,0.0),inference_ops_proxy=op_count(method),operations_note='per-vector analytic proxy; no fused-kernel runtime claim'))
 return rows

def main():
 rows=[]
 for label,seed in SEEDS.items():rows.extend(run_one(label,seed,ROOT/'source'/'payloads'))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('wrote',len(rows),'task-method rows')
if __name__=='__main__':main()
