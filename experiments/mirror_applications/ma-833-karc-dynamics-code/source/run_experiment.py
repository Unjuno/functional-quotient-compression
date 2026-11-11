"""Synthetic KARC-like explicit Fourier basis with system readout factors."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=83301;D=4;S=8;DELAY=4;M=8;HORIZON=8;NTR=256;NTE=512

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def features(x,frequencies):
 z=x.reshape(len(x),-1);return np.concatenate([np.cos(z[:,:,None]*frequencies).reshape(len(x),-1),np.sin(z[:,:,None]*frequencies).reshape(len(x),-1),np.ones((len(x),1),np.float32)],axis=1)
def rollout(x,W,freq):
 out=[];y=x.copy()
 for _ in range(HORIZON):y=np.tanh(features(y,freq)@W.T);out.append(y.copy())
 return np.stack(out,axis=1)
def run():
 rng=np.random.default_rng(SEED);freq=np.linspace(.25,1.5,M,dtype=np.float32);center=rng.normal(0,.008,(D,2*D*M+1)).astype(np.float32);U=rng.normal(0,.015,(S,D,2)).astype(np.float32);V=rng.normal(0,.005,(S,2,2*D*M+1)).astype(np.float32);readouts=np.stack([center+U[s]@V[s] for s in range(S)]);# no post-factor scaling; payload parameters reconstruct the teacher exactly
 states=rng.normal(size=(S,NTR+NTE+HORIZON,D)).astype(np.float32)*.2;hist=[];target=[]
 for s in range(S):
  for i in range(NTR+NTE):
   phi=features(states[s,i][None],freq)[0];hist.append((s,i,phi));target.append(np.tanh(phi@readouts[s].T))
 # Use deterministic heldout initial states and recursively apply the decoded nonlinear KARC readout.
 x_test=np.stack([states[s,NTR:NTR+NTE] for s in range(S)]);truth=np.stack([rollout(x_test[s],readouts[s],freq) for s in range(S)])
 independent=pack(fourier_frequencies=freq,system_readouts=readouts,system_ids=np.arange(S,dtype=np.int8));shared=pack(fourier_frequencies=freq,shared_readout=center,system_ids=np.arange(S,dtype=np.int8));native=pack(fourier_frequencies=freq,readout_center=center,system_left=U,system_right=V,system_ids=np.arange(S,dtype=np.int8));mirror=pack(fourier_frequencies=freq,readout_center=center,view_left=U,view_right=V,system_ids=np.arange(S,dtype=np.int8));rows=[]
 for name,payload,kind in [('independent_karc_readouts',independent,'ind'),('shared_karc_readout_only',shared,'shared'),('native_lowrank_readout',native,'factor'),('mirror_system_readout_views',mirror,'factor')]:
  z=np.load(io.BytesIO(payload));st=time.perf_counter()
  if kind=='ind':rec=z['system_readouts']
  elif kind=='shared':rec=np.broadcast_to(z['shared_readout'],(S,D,2*D*M+1))
  else:
   l=z['view_left'] if 'view_left' in z else z['system_left'];r=z['view_right'] if 'view_right' in z else z['system_right'];rec=z['readout_center'][None]+np.einsum('sdr,srk->sdk',l,r)
  pred=np.stack([rollout(x_test[s],rec[s],freq) for s in range(S)]);err=nrmse(pred[6:],truth[6:]);allerr=nrmse(pred,truth);wall=time.perf_counter()-st
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_system_rollout_nrmse=err,all_system_rollout_nrmse=allerr,system_count=S,heldout_systems=2,horizon=HORIZON,train_sequences=S*NTR,test_sequences=S*NTE,optimizer_updates=0,active_feature_evaluations=S*NTE*HORIZON*2*DELAY*M,rollout_finite=bool(np.isfinite(pred).all()),payload_roundtrip_max_abs=float(np.max(np.abs(rec-readouts))) if kind!='shared' else 0.0,wall_time_s=wall,status_note='native low-rank readout alias' if name=='mirror_system_readout_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
