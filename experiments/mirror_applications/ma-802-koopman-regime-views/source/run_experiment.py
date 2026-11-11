"""Synthetic nonlinear observation map with low-rank latent Koopman regime views."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[1];SEED=80201;D=8;U=8;T=40;N=128
REGIMES=np.array([(a,b) for a in (-1.,0.,1.) for b in (-1.,0.,1.)],np.float32)
HELD=np.array([0,8]);TRAIN=np.array([i for i in range(9) if i not in HELD])

def pack(**x):
 b=io.BytesIO();np.savez_compressed(b,**x);return b.getvalue()

def observe(z,q):return np.tanh(z@q.T)
def encode(x,q):return np.arctanh(np.clip(x,-.999999,.999999))@q

def simulate(a,g,q,z0,us):
 z=np.empty((len(z0),T+1,D),np.float32);x=np.empty_like(z);z[:,0]=z0;x[:,0]=observe(z[:,0],q)
 for t in range(T):z[:,t+1]=z[:,t]@a.T+us[:,t]@g.T;x[:,t+1]=observe(z[:,t+1],q)
 return z,x

def nrmse(p,y):return float(np.sqrt(np.mean((p-y)**2)/(np.mean(y*y)+1e-12)))

def run():
 rng=np.random.default_rng(SEED);q,_=np.linalg.qr(rng.normal(size=(D,D)));q=q.astype(np.float32)
 a0=(.48*np.eye(D)+rng.normal(0,.008,(D,D))).astype(np.float32)
 b1=rng.normal(0,.008,(D,D)).astype(np.float32);b2=rng.normal(0,.008,(D,D)).astype(np.float32)
 teacher=np.stack([a0+a*b1+b*b2 for a,b in REGIMES]);g=(.035*np.eye(D)+rng.normal(0,.004,(D,D))).astype(np.float32)
 rho=np.array([max(abs(np.linalg.eigvals(a))) for a in teacher])
 ztrain=[];utrain=[];znext=[]
 for k in TRAIN:
  z0=rng.normal(0,.12,(N,D)).astype(np.float32);us=rng.normal(0,.2,(N,T,U)).astype(np.float32)
  z,x=simulate(teacher[k],g,q,z0,us)
  encoded=encode(x,q)
  ztrain.append(encoded[:,:-1].reshape(-1,D));utrain.append(us.reshape(-1,U));znext.append(encoded[:,1:].reshape(-1,D))
 fitted=[];gfit=[]
 for j,k in enumerate(TRAIN):
  x=np.concatenate([ztrain[j],utrain[j]],axis=1);beta=np.linalg.lstsq(x,znext[j],rcond=None)[0]
  fitted.append(beta[:D].T.astype(np.float32));gfit.append(beta[D:].T.astype(np.float32))
 fitted=np.stack(fitted);g_est=np.mean(gfit,axis=0).astype(np.float32)
 design=np.concatenate([np.ones((len(TRAIN),1),np.float32),REGIMES[TRAIN]],axis=1)
 coef=np.linalg.lstsq(design,fitted.reshape(len(TRAIN),-1),rcond=None)[0].astype(np.float32)
 core=coef[0].reshape(D,D);bases=coef[1:].reshape(2,D,D)
 predicted=np.stack([core+a*bases[0]+b*bases[1] for a,b in REGIMES])
 shared=np.broadcast_to(core,(9,D,D)).copy()
 z0s=rng.normal(0,.12,(9,N,D)).astype(np.float32);us=rng.normal(0,.2,(9,N,T,U)).astype(np.float32)
 targets=[];x0=[]
 for k in range(9):
  _,x=simulate(teacher[k],g,q,z0s[k],us[k]);targets.append(x);x0.append(x[:,0])
 targets=np.stack(targets);x0=np.stack(x0)
 payloads=[('independent_regime_operators',teacher,pack(operators=teacher,input_operator=g,orthogonal_encoder=q,regime_ids=np.arange(9,dtype=np.int8)),0),
  ('shared_operator_no_regime_code',shared,pack(operator=core,input_operator=g,orthogonal_encoder=q,regime_ids=np.arange(9,dtype=np.int8)),0),
  ('mirror_latent_regime_views',predicted,pack(shared_operator=core,operator_bases=bases,regime_codes=REGIMES,regime_ids=np.arange(9,dtype=np.int8),input_operator=g,orthogonal_encoder=q),0),
  ('native_lowrank_conditional_dynamics',predicted,pack(shared_operator=core,operator_bases=bases,regime_codes=REGIMES,regime_ids=np.arange(9,dtype=np.int8),input_operator=g,orthogonal_encoder=q),0)]
 rows=[]
 for name,ops,payload,decode in payloads:
  saved=np.load(io.BytesIO(payload))
  if name=='independent_regime_operators':restored=saved['operators']
  elif name=='shared_operator_no_regime_code':restored=np.broadcast_to(saved['operator'],(9,D,D))
  else:restored=np.stack([saved['shared_operator']+a*saved['operator_bases'][0]+b*saved['operator_bases'][1] for a,b in saved['regime_codes']])
  rt=float(np.max(np.abs(restored-ops)))
  if rt>1e-5:raise AssertionError(f'{name} serialized operator reconstruction error {rt}')
  start=time.perf_counter();preds=[];rhos=[]
  for k in range(9):
   # Begin from the observed nonlinear state and invert the paid orthogonal encoder.
   zinit=encode(x0[k],q)
   a=ops[k] if name!='mirror_latent_regime_views' else core+REGIMES[k,0]*bases[0]+REGIMES[k,1]*bases[1]
   _,xp=simulate(a,g_est,q,zinit,us[k]);preds.append(xp);rhos.append(max(abs(np.linalg.eigvals(a))))
  wall=time.perf_counter()-start;preds=np.stack(preds)
  one_true=targets[:, :,1];one_pred=preds[:,:,1]
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),
   heldout_one_step_nrmse=nrmse(one_pred[HELD],one_true[HELD]),heldout_horizon40_nrmse=nrmse(preds[HELD],targets[HELD]),
   all_regime_horizon40_nrmse=nrmse(preds,targets),max_spectral_radius=float(max(rhos)),
   active_macs_per_step=(5 if name in ('mirror_latent_regime_views','native_lowrank_conditional_dynamics') else 3)*D*D,
   reconstruction_macs_total=decode,train_sequences=len(TRAIN)*N,test_sequences=9*N,sequence_length=T,optimizer_updates=0,
   wall_time_s=wall,payload_roundtrip_max_abs=rt,status_note='exact native low-rank alias' if name=='mirror_latent_regime_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows

if __name__=='__main__':
 for r in run():print(r)
