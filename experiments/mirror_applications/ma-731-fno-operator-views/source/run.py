from __future__ import annotations
import hashlib,io,time
import numpy as np
K=16;T=.04;TRAIN_A=np.linspace(.02,.16,20);TEST_A=np.array([.03,.05,.07,.09,.11,.13,.18]);RES=(64,128);FIELDS=128;RANKS=(2,4,8,16)
def pack(a):
 b=io.BytesIO();np.savez_compressed(b,**a);v=b.getvalue();return len(v),hashlib.sha256(v).hexdigest()
def solve(op,alpha,res,seed):
 rng=np.random.default_rng(seed+int(alpha*1000)+res);x=rng.normal(size=(FIELDS,res));xf=np.fft.rfft(x,axis=1);out=np.zeros_like(xf);m=min(K,xf.shape[1]-1);ks=np.arange(1,m+1);out[:,1:m+1]=xf[:,1:m+1]*op(alpha)[:m];y=np.fft.irfft(out,n=res,axis=1);truth=np.zeros_like(xf);truth[:,1:m+1]=xf[:,1:m+1]*np.exp(-alpha*ks**2*T);yt=np.fft.irfft(truth,n=res,axis=1);return float(np.linalg.norm(y-yt)/(np.linalg.norm(yt)+1e-12))
def run(seed=73101):
 rng=np.random.default_rng(seed);ks=np.arange(1,K+1);train=np.stack([np.exp(-a*ks**2*T) for a in TRAIN_A]);truth=lambda a:np.exp(-a*ks**2*T)
 # Train-only centered PCA over operator multipliers, then fit code polynomial against alpha.
 mean=train.mean(0);u,s,vt=np.linalg.svd((train-mean).T,full_matrices=False);rows=[]
 methods=[]
 independent={'multipliers':np.stack([truth(a) for a in TEST_A]).astype(np.float32),'alphas':TEST_A.astype(np.float32)};methods.append(('independent_kernel_bank',0,independent,lambda a:truth(a)))
 # Native scalar alpha polynomial per Fourier mode.
 poly=np.stack([np.polyfit(TRAIN_A,train[:,k],3) for k in range(K)])
 methods.append(('native_per_mode_cubic',K,{'poly_coefficients':poly.astype(np.float32),'modes':ks,'time':np.array([T])},lambda a:np.array([np.polyval(poly[k],a) for k in range(K)])))
 # Lower-cost linear interpolation baseline stores native training multiplier bank.
 methods.append(('native_parameter_interpolation',0,{'alphas':TRAIN_A.astype(np.float32),'multipliers':train.astype(np.float32)},lambda a:np.array([np.interp(a,TRAIN_A,train[:,k]) for k in range(K)])))
 methods.append(('analytic_physics_oracle',0,{'modes':ks,'time':np.array([T])},truth))
 for q in RANKS:
  B=u[:,:q];codes=(train-mean)@B;coef=np.stack([np.polyfit(TRAIN_A,codes[:,j],3) for j in range(q)])
  def op(a,B=B,mean=mean,coef=coef):return mean+B@np.array([np.polyval(coef[j],a) for j in range(B.shape[1])])
  state={'mean':mean.astype(np.float32),'basis_atoms':B.astype(np.float32),'alpha_code_polynomials':coef.astype(np.float32),'modes':ks,'time':np.array([T])}
  methods.append(('Mirror_operator_codes',q,state,op))
  methods.append(('native_PCA_operator_codes',q,state,op))
 for name,q,state,op in methods:
  size,digest=pack(state);t0=time.perf_counter()
  start=len(rows)
  for res in RES:
   errs=[solve(op,a,res,seed) for a in TEST_A];rows.append({'seed':seed,'method':name,'basis_rank':q,'resolution':res,'serialized_bytes':size,'payload_sha256':digest,'relative_solution_error':float(np.mean(errs)),'active_multiplies':int(len(TEST_A)*FIELDS*min(K,res//2)),'wall_ms':0.,'status_note':'synthetic periodic diffusion; train alpha grid disjoint from test alphas'})
  elapsed=(time.perf_counter()-t0)*1000
  for row in rows[start:]:row['wall_ms']=elapsed
 return rows
def save(rows,path):
 import csv
 with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=73101);p.add_argument('--out',default='RESULTS_CORE.csv');a=p.parse_args();t=time.perf_counter();rows=run(a.seed);save(rows,a.out)
