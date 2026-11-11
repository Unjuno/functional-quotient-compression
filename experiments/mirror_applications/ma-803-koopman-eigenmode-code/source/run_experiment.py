"""Synthetic diagonal Koopman eigenmode views under a known nonlinear observation map."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[1];SEED=80301;D=8;T=48;N=128
REG=np.array([(a,b) for a in (-1.,0.,1.) for b in (-1.,0.,1.)],np.float32);HELD=np.array([0,8]);TRAIN=np.array([i for i in range(9) if i not in HELD])

def pack(**x):
 b=io.BytesIO();np.savez_compressed(b,**x);return b.getvalue()
def encode(x,q):return np.arctanh(np.clip(x,-.999999,.999999))@q
def observe(z,q):return np.tanh(z@q.T)
def sim(lam,g,q,z0,u):
 z=np.empty((len(z0),T+1,D),np.float32);x=np.empty_like(z);z[:,0]=z0;x[:,0]=observe(z[:,0],q)
 for t in range(T):z[:,t+1]=z[:,t]*lam+u[:,t]@g.T;x[:,t+1]=observe(z[:,t+1],q)
 return z,x
def nrmse(p,y):return float(np.sqrt(np.mean((p-y)**2)/(np.mean(y*y)+1e-12)))

def run():
 rng=np.random.default_rng(SEED);q,_=np.linalg.qr(rng.normal(size=(D,D)));q=q.astype(np.float32)
 l0=rng.uniform(.52,.66,D).astype(np.float32);d1=rng.normal(0,.012,D).astype(np.float32);d2=rng.normal(0,.012,D).astype(np.float32)
 true_lam=np.stack([l0+a*d1+b*d2 for a,b in REG]);g=(.035*np.eye(D)+rng.normal(0,.004,(D,D))).astype(np.float32)
 train_z=[];train_u=[];train_next=[]
 for k in TRAIN:
  z0=rng.normal(0,.12,(N,D)).astype(np.float32);u=rng.normal(0,.2,(N,T,D)).astype(np.float32)
  z,x=sim(true_lam[k],g,q,z0,u);enc=encode(x,q)
  train_z.append(enc[:,:-1].reshape(-1,D));train_u.append(u.reshape(-1,D));train_next.append(enc[:,1:].reshape(-1,D))
 fit_lam=[];fit_g=[]
 for j,k in enumerate(TRAIN):
  rows=[];gm=np.zeros((D,D),np.float32)
  for d in range(D):
   design=np.concatenate([train_z[j][:,d:d+1],train_u[j]],axis=1)
   beta=np.linalg.lstsq(design,train_next[j][:,d],rcond=None)[0]
   rows.append(beta[0]);gm[d]=beta[1:]
  fit_lam.append(rows);fit_g.append(gm)
 fit_lam=np.asarray(fit_lam,np.float32);g_est=np.mean(fit_g,axis=0).astype(np.float32)
 design=np.concatenate([np.ones((len(TRAIN),1),np.float32),REG[TRAIN]],axis=1)
 basis=np.linalg.lstsq(design,fit_lam,rcond=None)[0].astype(np.float32)
 pred_lam=np.concatenate([np.ones((9,1),np.float32),REG],axis=1)@basis
 shared=np.broadcast_to(fit_lam.mean(axis=0),(9,D)).copy()
 z0s=rng.normal(0,.12,(9,N,D)).astype(np.float32);us=rng.normal(0,.2,(9,N,T,D)).astype(np.float32)
 targets=[];x0=[]
 for k in range(9):
  _,x=sim(true_lam[k],g,q,z0s[k],us[k]);targets.append(x);x0.append(x[:,0])
 targets=np.stack(targets);x0=np.stack(x0)
 ops=np.stack([np.diag(lam) for lam in true_lam]).astype(np.float32)
 payloads=[('independent_dense_modal_operators',true_lam,pack(operators=ops,input_operator=g,orthogonal_encoder=q,regime_ids=np.arange(9,dtype=np.int8)),0,'dense'),
  ('native_diagonal_eigenvalue_table',true_lam,pack(eigenvalues=true_lam,input_operator=g,orthogonal_encoder=q,regime_ids=np.arange(9,dtype=np.int8)),0,'diagonal'),
  ('shared_eigenvalues_no_regime_code',shared,pack(eigenvalues=fit_lam.mean(axis=0),input_operator=g,orthogonal_encoder=q,regime_ids=np.arange(9,dtype=np.int8)),0,'diagonal'),
  ('mirror_factorized_eigenmode_codes',pred_lam,pack(lambda_basis=basis,regime_codes=REG,regime_ids=np.arange(9,dtype=np.int8),input_operator=g,orthogonal_encoder=q),0,'factorized'),
  ('native_spectral_factorization',pred_lam,pack(lambda_basis=basis,regime_codes=REG,regime_ids=np.arange(9,dtype=np.int8),input_operator=g,orthogonal_encoder=q),0,'factorized')]
 rows=[]
 for name,lams,payload,decode,kind in payloads:
  saved=np.load(io.BytesIO(payload))
  if kind=='dense':restored=np.diagonal(saved['operators'],axis1=-2,axis2=-1)
  elif kind=='diagonal':restored=saved['eigenvalues'] if 'eigenvalues' in saved else np.broadcast_to(saved['eigenvalues'],(9,D))
  else:restored=np.concatenate([np.ones((9,1),np.float32),saved['regime_codes']],axis=1)@saved['lambda_basis']
  rt=float(np.max(np.abs(restored-lams)))
  if rt>1e-5:raise AssertionError(f'{name} payload roundtrip {rt}')
  start=time.perf_counter();pred=[];rhos=[]
  for k in range(9):
   zinit=encode(x0[k],q);_,xp=sim(lams[k],g_est,q,zinit,us[k]);pred.append(xp);rhos.append(np.max(np.abs(lams[k])))
  wall=time.perf_counter()-start;pred=np.stack(pred)
  one_true=targets[:,:,1];one_pred=pred[:,:,1]
  extrap=(basis[0]+2*basis[1]+2*basis[2])
  active=(3*D*D) if kind=='dense' else (2*D*D+D)
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),
   heldout_one_step_nrmse=nrmse(one_pred[HELD],one_true[HELD]),heldout_horizon48_nrmse=nrmse(pred[HELD],targets[HELD]),
   all_regime_horizon48_nrmse=nrmse(pred,targets),max_registered_spectral_radius=float(max(rhos)),out_of_grid_code_spectral_radius=float(np.max(np.abs(extrap))),
   active_macs_per_step=active,reconstruction_macs_total=decode,train_sequences=len(TRAIN)*N,test_sequences=9*N,sequence_length=T,
   optimizer_updates=0,wall_time_s=wall,payload_roundtrip_max_abs=rt,status_note='exact native spectral alias' if name=='mirror_factorized_eigenmode_codes' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows

if __name__=='__main__':
 for r in run():print(r)
