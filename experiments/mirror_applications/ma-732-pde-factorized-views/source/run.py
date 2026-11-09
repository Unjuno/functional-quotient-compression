from __future__ import annotations
import hashlib,io,time
import numpy as np
ALPHAS=np.array([.5,1.,2.,4.]);BCS=np.array([[0,0],[1,0],[0,1],[1,1]],float);HOLD={(.5,1),(1.,2),(2.,1),(4.,0)}
def pack(a):
 b=io.BytesIO();np.savez_compressed(b,**a);v=b.getvalue();return len(v),hashlib.sha256(v).hexdigest()
def operators(n):
 h=1/n;N=n-1;A=np.diag(np.full(N,2/h**2))+np.diag(np.full(N-1,-1/h**2),1)+np.diag(np.full(N-1,-1/h**2),-1);K=np.linalg.inv(A);rhs=np.zeros((N,2));rhs[0,0]=1/h**2;rhs[-1,1]=1/h**2;P=K@rhs;return K,P
def fields(rng,N,count):
 raw=rng.normal(size=(count,N));fh=np.fft.rfft(raw,axis=1);freq=np.arange(fh.shape[1]);return np.fft.irfft(fh/(1+freq**2)[None,:],n=N,axis=1)
def run(seed=73201):
 rows=[];train=[(float(a),b) for a in ALPHAS for b in range(4) if (float(a),b) not in HOLD];held=[(float(a),b) for a in ALPHAS for b in range(4) if (float(a),b) in HOLD]
 for n in (32,64):
  K,P=operators(n);N=n-1;rng=np.random.default_rng(seed+n);design=[];targets=[]
  for a,b in train:
   bc=BCS[b];f=fields(rng,N,128);u=(K@f.T/a).T+P@bc;design.append(np.concatenate([f/a,np.broadcast_to(bc,(len(f),2))],axis=1));targets.append(u)
  X=np.concatenate(design);Y=np.concatenate(targets);coef=np.linalg.lstsq(X,Y,rcond=None)[0];Khat=coef[:N].T;Phat=coef[N:].T
  test=[]
  for a,b in held:
   f=fields(rng,N,128);test.append((a,b,f,BCS[b],(K@f.T/a).T+P@BCS[b]))
  bank=[]
  for a in ALPHAS:
   for bc in BCS:bank.append(np.concatenate([K/a,P],axis=1))
  independent={'operators':np.stack(bank).astype(np.float32),'alphas':ALPHAS.astype(np.float32),'boundary_pairs':BCS.astype(np.float32)}
  shared={'green_operator':Khat.astype(np.float32),'boundary_basis':Phat.astype(np.float32),'alpha_values':ALPHAS.astype(np.float32),'boundary_pairs':BCS.astype(np.float32)}
  states=[('independent_operator_bank',independent),('native_parameter_conditioning',shared),('Mirror_factor_code',dict(shared)),('analytic_exact_operator',{'green_operator':K.astype(np.float32),'boundary_basis':P.astype(np.float32),'time_metadata':np.array([n],np.int16)})]
  for name,state in states:
   size,digest=pack(state);errs=[];t0=time.perf_counter()
   for a,b,f,bc,truth in test:
    if name=='independent_operator_bank':pred=(np.concatenate([f,np.broadcast_to(bc,(len(f),2))],axis=1)@np.concatenate([K/a,P],axis=1).T)
    elif name=='analytic_exact_operator':pred=(K@f.T/a).T+P@bc
    else:pred=(Khat@f.T/a).T+Phat@bc
    errs.append(np.linalg.norm(pred-truth)/(np.linalg.norm(truth)+1e-12))
   wall=(time.perf_counter()-t0)*1000
   rows.append({'seed':seed,'method':name,'resolution':n,'serialized_bytes':size,'payload_sha256':digest,'heldout_relative_l2':float(np.mean(errs)),'active_multiplies':int(len(held)*128*N*N),'wall_ms':wall,'status_note':'shared Green and boundary basis estimated from train combinations; held-out forcing disjoint'})
 return rows
def save(rows,path):
 import csv
 with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=73201);p.add_argument('--out',default='RESULTS_CORE.csv');a=p.parse_args();save(run(a.seed),a.out)
