"""Synthetic Koopman regime bank with sparse private nonlinear residual codes."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[1];SEED=80401;D=8;T=40;N=128
REG=np.array([(a,b) for a in (-1.,0.,1.) for b in (-1.,0.,1.)],np.float32)
THRESH=.02

def pack(**x):
 b=io.BytesIO();np.savez_compressed(b,**x);return b.getvalue()
def observe(z,q):return np.tanh(z@q.T)
def encode(x,q):return np.arctanh(np.clip(x,-.999999,.999999))@q
def sim(a,g,q,w,v,alpha,z0,u):
 z=np.empty((len(z0),T+1,D),np.float32);x=np.empty_like(z);z[:,0]=z0;x[:,0]=observe(z[:,0],q)
 for t in range(T):
  phi=np.tanh(z[:,t]@w)**2
  z[:,t+1]=z[:,t]@a.T+u[:,t]@g.T+phi[:,None]*alpha*v[None]
  x[:,t+1]=observe(z[:,t+1],q)
 return z,x
def nrmse(p,y):return float(np.sqrt(np.mean((p-y)**2)/(np.mean(y*y)+1e-12)))

def jacobian_rho(a,w,v,alpha,z):
 r=[]
 for state in z.reshape(-1,D):
  t=np.tanh(float(w@state));grad=2*t*(1-t*t)*w
  j=a+alpha*np.outer(v,grad);r.append(float(max(abs(np.linalg.eigvals(j)))))
 return max(r,default=0.)

def run():
 rng=np.random.default_rng(SEED);q,_=np.linalg.qr(rng.normal(size=(D,D)));q=q.astype(np.float32)
 a0=(.44*np.eye(D)+rng.normal(0,.008,(D,D))).astype(np.float32)
 b1=rng.normal(0,.006,(D,D)).astype(np.float32);b2=rng.normal(0,.006,(D,D)).astype(np.float32)
 teacher_a=np.stack([a0+a*b1+b*b2 for a,b in REG])
 g=(.028*np.eye(D)+rng.normal(0,.003,(D,D))).astype(np.float32)
 w=rng.normal(0,1,D).astype(np.float32);w=w/np.linalg.norm(w)*.7
 v=rng.normal(0,.16,D).astype(np.float32);alpha=np.zeros(9,np.float32);alpha[0]=3.0;alpha[8]=-2.5
 train_z=[];train_u=[];train_next=[];train_phi=[]
 for k in range(9):
  z0=rng.normal(0,.25,(N,D)).astype(np.float32);u=rng.normal(0,.18,(N,T,D)).astype(np.float32)
  z,x=sim(teacher_a[k],g,q,w,v,alpha[k],z0,u);enc=encode(x,q)
  zt=enc[:,:-1].reshape(-1,D);un=u.reshape(-1,D);zn=enc[:,1:].reshape(-1,D)
  phi=(np.tanh(zt@w)**2).reshape(-1,1)
  train_z.append(zt);train_u.append(un);train_next.append(zn);train_phi.append(phi)
 fit_a=[];fit_g=[];fit_q=[]
 for k in range(9):
  design=np.concatenate([train_z[k],train_u[k],train_phi[k]],axis=1)
  beta=np.linalg.lstsq(design,train_next[k],rcond=None)[0]
  fit_a.append(beta[:D].T);fit_g.append(beta[D:2*D].T);fit_q.append(beta[-1])
 fit_a=np.asarray(fit_a,np.float32);gfit=np.mean(fit_g,axis=0).astype(np.float32);fit_q=np.asarray(fit_q,np.float32)
 reg_design=np.concatenate([np.ones((9,1),np.float32),REG],axis=1)
 coeff=np.linalg.lstsq(reg_design,fit_a.reshape(9,-1),rcond=None)[0].astype(np.float32)
 core=coeff[0].reshape(D,D);operator_bases=coeff[1:].reshape(2,D,D)
 pred_a=np.stack([core+a*operator_bases[0]+b*operator_bases[1] for a,b in REG])
 # Shared nonlinear residual direction is the leading output-space component; sparse regime scalars are private.
 ures,sres,vh=np.linalg.svd(fit_q,full_matrices=False);vfit=vh[0].astype(np.float32);alpha_fit=(ures[:,0]*sres[0]).astype(np.float32)
 active=np.flatnonzero(np.abs(alpha_fit)>THRESH).astype(np.int8);active_values=alpha_fit[active.astype(int)].astype(np.float32)
 alpha_sparse=np.zeros(9,np.float32);alpha_sparse[active.astype(int)]=active_values
 z0s=rng.normal(0,.25,(9,N,D)).astype(np.float32);us=rng.normal(0,.18,(9,N,T,D)).astype(np.float32)
 targets=[];x0=[]
 for k in range(9):
  _,x=sim(teacher_a[k],g,q,w,v,alpha[k],z0s[k],us[k]);targets.append(x);x0.append(x[:,0])
 targets=np.stack(targets);x0=np.stack(x0)
 view_payload=pack(shared_operator=core,operator_bases=operator_bases,regime_codes=REG,regime_ids=np.arange(9,dtype=np.int8),input_operator=g,orthogonal_encoder=q,nonlinear_feature=w,residual_direction=vfit,residual_indices=active,residual_values=active_values)
 independent=pack(operators=teacher_a,input_operator=g,orthogonal_encoder=q,nonlinear_feature=w,residual_vectors=alpha[:,None]*v[None],regime_ids=np.arange(9,dtype=np.int8))
 shared=pack(shared_operator=core,operator_bases=operator_bases,regime_codes=REG,regime_ids=np.arange(9,dtype=np.int8),input_operator=g,orthogonal_encoder=q,nonlinear_feature=w)
 methods=[('independent_full_dynamics',teacher_a,alpha,v,independent,'independent'),
  ('shared_operator_no_private_residual',pred_a,np.zeros(9,np.float32),vfit,shared,'shared'),
  ('mirror_shared_operator_sparse_residual',pred_a,alpha_sparse,vfit,view_payload,'view'),
  ('native_shared_operator_private_residual',pred_a,alpha_sparse,vfit,view_payload,'view')]
 rows=[]
 for name,ops,alphas,vuse,payload,kind in methods:
  saved=np.load(io.BytesIO(payload))
  if kind=='independent':restored_a=saved['operators'];restored_alpha=np.linalg.norm(saved['residual_vectors'],axis=1)/max(np.linalg.norm(v),1e-12);restored_alpha=np.sign(alphas)*restored_alpha
  else:
   restored_a=np.stack([saved['shared_operator']+a*saved['operator_bases'][0]+b*saved['operator_bases'][1] for a,b in saved['regime_codes']])
   restored_alpha=np.zeros(9,np.float32)
   if kind=='view':restored_alpha[saved['residual_indices'].astype(int)]=saved['residual_values']
  rt=max(float(np.max(np.abs(restored_a-ops))),float(np.max(np.abs(restored_alpha-alphas))))
  if rt>2e-4:raise AssertionError(f'{name} payload roundtrip error {rt}')
  start=time.perf_counter();pred=[];rhos=[];features=[]
  for k in range(9):
   zinit=encode(x0[k],q);_,xp=sim(ops[k],gfit,q,w,vuse,alphas[k],zinit,us[k]);pred.append(xp);rhos.append(jacobian_rho(ops[k],w,vuse,alphas[k],zinit))
   features.append(zinit)
  wall=time.perf_counter()-start;pred=np.stack(pred)
  common=[k for k in range(9) if alpha[k]==0];exceptional=[k for k in range(9) if alpha[k]!=0]
  active_mac=(2*D*D+2*D) if kind=='independent' else (4*D*D+2*D)
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),
   horizon40_nrmse=nrmse(pred,targets),common_regime_nrmse=nrmse(pred[common],targets[common]),exceptional_regime_nrmse=nrmse(pred[exceptional],targets[exceptional]),
   max_local_jacobian_spectral_radius=float(max(rhos)),active_private_residual_count=int(np.count_nonzero(np.abs(alphas)>THRESH)),
   active_macs_per_step=active_mac,reconstruction_macs_total=0,train_sequences=9*N,test_sequences=9*N,sequence_length=T,optimizer_updates=0,
   wall_time_s=wall,payload_roundtrip_max_abs=rt,status_note='exact native shared/private alias' if name=='mirror_shared_operator_sparse_residual' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');wr.writeheader();wr.writerows(rows)
 return rows

if __name__=='__main__':
 for r in run():print(r)
