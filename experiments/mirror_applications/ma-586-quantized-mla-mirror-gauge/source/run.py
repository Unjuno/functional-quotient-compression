#!/usr/bin/env python3
import argparse,csv,json,struct,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SEEDS=(58601,58602);ROLES=8;SEQ=4;D=16;BITS=4;ANGLES=np.linspace(0,np.pi/2,64,endpoint=False)
def hadamard(n):
 h=np.array([[1.]])
 while h.shape[0]<n:h=np.block([[h,h],[h,-h]])
 return h/np.sqrt(n)
H=hadamard(D)
def givens(t):
 q=np.eye(D)
 for i in range(0,D,2):q[i:i+2,i:i+2]=[[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]
 return q
def quant(a):
 scale=np.maximum(np.max(np.abs(a),axis=(1,2),keepdims=True)/7,1e-8);v=np.clip(np.rint(a/scale),-7,7).astype(np.int8);u=(v.astype(np.int16)+8).astype(np.uint8).ravel();packed=(u[0::2]<<4)|u[1::2];return v.astype(np.float32)*scale,packed.tobytes(),scale.astype(np.float32).ravel()
def payload(kb,vb,scales,idx,tag):
 meta=json.dumps({'tag':tag,'bits':4,'roles':ROLES,'seq':SEQ,'dim':D,'angle_index':int(idx)},sort_keys=True,separators=(',',':')).encode()
 return struct.pack('<H',len(meta))+meta+np.asarray(scales,dtype='<f4').tobytes()+kb+vb
def attend(q,k,v):
 s=q@k.T/np.sqrt(D);s-=s.max(axis=-1,keepdims=True);p=np.exp(s);p/=p.sum(axis=-1,keepdims=True);return p@v
def run(seed):
 rng=np.random.default_rng(seed);K=rng.standard_t(3,size=(ROLES,SEQ,D)).astype(np.float32);V=rng.standard_t(3,size=(ROLES,SEQ,D)).astype(np.float32);Q=rng.normal(size=(ROLES,64,D)).astype(np.float32);qt,qe=Q[:,:32],Q[:,32:]
 # add shared low-rank structure and sparse outliers to create a cache-like latent bank
 shared=rng.normal(size=(ROLES,1,4)).astype(np.float32)@rng.normal(size=(4,D)).astype(np.float32)[None,:,:];K+=.4*shared;V+=.4*shared
 outmask=rng.random(K.shape)<.015;K+=outmask*rng.normal(0,7,K.shape);outmask=rng.random(V.shape)<.015;V+=outmask*rng.normal(0,7,V.shape)
 reftrain=np.stack([attend(qt[i],K[i],V[i]) for i in range(ROLES)]);ref=np.stack([attend(qe[i],K[i],V[i]) for i in range(ROLES)])
 transforms={'identity':np.eye(D),'hadamard':H}
 def encode(Qm,tag,idx,queries,reference):
  kr,pk,sk=quant(K@Qm.T);vr,pv,sv=quant(V@Qm.T);recK=kr@Qm;recV=vr@Qm;out=np.stack([attend(queries[i],recK[i],recV[i]) for i in range(ROLES)]);err=float(np.mean((out-reference)**2)/max(np.mean(reference**2),1e-12));return err,payload(pk,pv,np.concatenate([sk,sv]),idx,tag),out
 rows=[]
 for base in ('identity','hadamard'):
  err,p,_=encode(transforms[base],base,255,qe,ref);rows.append({'world':seed,'method':base,'serialized_bytes':len(p),'train_examples':ROLES*SEQ,'optimizer_updates':0,'angle_evaluations':0,'reconstruction_mac_proxy':ROLES*SEQ*D*D,'attention_mac_proxy':ROLES*64*SEQ*D*2,'wall_time_s':'0','attention_output_nmse':f'{err:.9g}','status_note':'fixed gauge int4'})
 start=time.perf_counter();scores=[]
 for idx,t in enumerate(ANGLES):
  err,p,_=encode(givens(t),'mirror',idx,qt,reftrain);scores.append(err)
 best=int(np.argmin(scores));angle=ANGLES[best];err,p,_=encode(givens(angle),'mirror',best,qe,ref);search=time.perf_counter()-start
 # Direct ordinary coefficient control gets same fixed basis and exact same selected code/payload.
 errd,pd,_=encode(givens(angle),'direct',best,qe,ref)
 for method,payload_bytes,e in [('mirror_givens',p,err),('direct_givens',pd,errd)]:rows.append({'world':seed,'method':method,'serialized_bytes':len(payload_bytes),'train_examples':ROLES*SEQ,'optimizer_updates':0,'angle_evaluations':64,'reconstruction_mac_proxy':ROLES*SEQ*D*D,'attention_mac_proxy':ROLES*64*SEQ*D*2,'wall_time_s':f'{search:.6f}','attention_output_nmse':f'{e:.9g}','status_note':f'best development-grid angle index {best}; shared gauge'})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'artifacts'/'dev_results.csv'));a=ap.parse_args();r=[]
 for s in SEEDS:r+=run(s)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=r[0].keys());w.writeheader();w.writerows(r)
 print(json.dumps({'rows':len(r)},indent=2))
if __name__=='__main__':main()
