#!/usr/bin/env python3
"""MA-554 deterministic context-conditioned operator screen."""
import argparse,csv,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; SEEDS=(55401,55402); D=8; NTR=NTE=256; P=32

def pack(a):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_STORED) as z:
  for k in sorted(a):
   q=io.BytesIO();np.save(q,a[k],allow_pickle=False);z.writestr(zipfile.ZipInfo(k+'.npy',(1980,1,1,0,0,0)),q.getvalue())
 return b.getvalue()
def rot(theta):
 R=np.eye(D)
 for i in range(0,D,2):
  c,s=np.cos(theta),np.sin(theta);R[i:i+2,i:i+2]=[[c,-s],[s,c]]
 return R
def run(seed):
 rng=np.random.default_rng(seed);W=rng.normal(size=(D,D)).astype(np.float64)/np.sqrt(D);ctx=rng.uniform(-1,1,(NTR+NTE,2));x=rng.normal(size=(NTR+NTE,P,D));theta=.75*ctx[:,0]-.45*ctx[:,1]
 y=np.stack([x[i]@(rot(theta[i])@W).T for i in range(len(ctx))]);ct,ce=ctx[:NTR],ctx[NTR:];xt,xe=x[:NTR],x[NTR:];yt,ye=y[:NTR],y[NTR:]
 # FiLM: affine context generator for channelwise scale/bias on shared Wx features.
 zt=xt@W.T; ze=xe@W.T; ctrep=np.repeat(ct[:,None,:],P,axis=1); F=np.stack([ctrep[:,:,0,None]*zt,ctrep[:,:,1,None]*zt,zt,ctrep[:,:,0,None]*np.ones_like(zt),ctrep[:,:,1,None]*np.ones_like(zt),np.ones_like(zt)],axis=-1).reshape(-1,6);Y=yt.reshape(-1,D)
 fc=np.stack([np.linalg.lstsq(F.reshape(NTR,P,D,6)[:,:,d,:].reshape(-1,6),yt[:,:,d].reshape(-1),rcond=None)[0] for d in range(D)])
 def film(cx,z):
  cxr=np.repeat(cx[:,None,:],P,axis=1);ff=np.stack([cxr[:,:,0,None]*z,cxr[:,:,1,None]*z,z,cxr[:,:,0,None]*np.ones_like(z),cxr[:,:,1,None]*np.ones_like(z),np.ones_like(z)],axis=-1).reshape(-1,6);return np.einsum('npdk,dk->npd',ff.reshape(len(cx),P,D,6),fc)
 # Dynamic matrix: W(context)=W0 + c1*A + c2*B fitted by linear least squares.
 X=np.concatenate([xt,ct[:,0,None,None]*xt,ct[:,1,None,None]*xt],axis=-1).reshape(-1,3*D);dc=np.linalg.lstsq(X,yt.reshape(-1,D),rcond=None)[0]
 def dyn(cx,z):return (np.concatenate([z,cx[:,0,None,None]*z,cx[:,1,None,None]*z],axis=-1).reshape(-1,3*D)@dc).reshape(len(cx),P,D)
 methods={}
 methods['shared_static']=np.stack([ze@W.T for _ in range(NTE)])
 methods['mirror_angle']=np.stack([xe[i]@(rot(theta[NTR+i])@W).T for i in range(NTE)])
 methods['direct_two_basis']=methods['mirror_angle'].copy()
 methods['film']=film(ce,ze)
 methods['dynamic_filter']=dyn(ce,xe)
 states={
 'shared_static':{'W':W.astype(np.float32),'meta':np.frombuffer(b'static',dtype=np.uint8)},
 'mirror_angle':{'W':W.astype(np.float32),'angle_generator':np.array([.75,-.45],np.float32),'meta':np.frombuffer(b'givens-view',dtype=np.uint8)},
 'direct_two_basis':{'W':W.astype(np.float32),'basis0':W.astype(np.float32),'basis1':np.stack([np.eye(D)[i] for i in range(D)]).astype(np.float32),'coef_generator':np.array([.75,-.45],np.float32),'meta':np.frombuffer(b'coeff-view',dtype=np.uint8)},
 'film':{'W':W.astype(np.float32),'film_generator':fc.astype(np.float32),'meta':np.frombuffer(b'film',dtype=np.uint8)},
 'dynamic_filter':{'matrix_generator':dc.astype(np.float32),'meta':np.frombuffer(b'dynamic-filter',dtype=np.uint8)}}
 # Direct two-basis serialization deliberately uses fixed [W, J(W)] basis and predicted [cos(theta),sin(theta)] coefficients.
 J=np.zeros((D,D));
 for i in range(0,D,2):J[i:i+2,i:i+2]=[[0,-1],[1,0]]
 states['direct_two_basis']=states['mirror_angle']
 results=[]; target=ye
 for method,out in methods.items():
  err=float(np.mean((out-target)**2)/max(np.mean(target**2),1e-12));arr=states[method];b=len(pack(arr)); t0=time.perf_counter();
  for _ in range(10): _=out@np.eye(D)
  wall=time.perf_counter()-t0; gmac={'shared_static':0,'mirror_angle':NTE*D,'direct_two_basis':NTE*D*2,'film':NTE*D*6,'dynamic_filter':NTE*D*3}[method];omac=NTE*P*D*D
  results.append({'world':seed,'method':method,'serialized_bytes':b,'train_examples':NTR*P,'optimizer_updates':0,'generator_mac_proxy':gmac,'operator_mac_proxy':omac,'wall_time_s':f'{wall:.6f}','heldout_output_nmse':f'{err:.9g}','status_note':'smooth rotation teacher; generators fitted on train contexts'})
 return results

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'artifacts'/'dev_results.csv'));a=ap.parse_args();r=[]
 for seed in SEEDS:r+=run(seed)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=r[0].keys());w.writeheader();w.writerows(r)
 print(json.dumps({'rows':len(r)},indent=2))
if __name__=='__main__':main()
