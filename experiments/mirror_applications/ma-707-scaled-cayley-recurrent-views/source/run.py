#!/usr/bin/env python3
"""Shared recurrent base plus low-rank skew/Cayley task Views."""
import csv,io,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D,N,T=32,8,128
SEEDS={'dev-70701':70701,'audit-70711':70711,'audit-70712':70712,'audit-70713':70713};RANKS=(1,2,4);SCALE=.95
def npz(**a):
 b=io.BytesIO();np.savez(b,**a);return b.getvalue()
def cayley(S):
 I=np.eye(D,dtype=np.float32);return np.linalg.solve(I-S,I+S).astype(np.float32)
def traj(Q,x0,inp):
 states=np.zeros((T+1,D),np.float32);states[0]=x0
 for t in range(T):states[t+1]=SCALE*(Q@states[t])+np.float32(.1)*inp[t]
 return states
def run_one(label,seed,outdir):
 rng=np.random.default_rng(seed);outdir.mkdir(parents=True,exist_ok=True)
 Q0=np.linalg.qr(rng.normal(size=(D,D)))[0].astype(np.float32)
 x0=rng.normal(size=(N,D)).astype(np.float32);inp=rng.normal(size=(N,T,D)).astype(np.float32)
 rows=[]
 for rank in RANKS:
  U=rng.normal(size=(N,D,rank)).astype(np.float32)/np.sqrt(D*rank);V=rng.normal(size=(N,D,rank)).astype(np.float32)/np.sqrt(D*rank)
  Qs=[];Y=[];native_err=[]
  t0=time.perf_counter()
  for n in range(N):
   S=U[n]@V[n].T-V[n]@U[n].T;C=cayley(S);Q=(Q0@C).astype(np.float32)
   Qs.append(Q);Y.append(traj(Q,x0[n],inp[n]))
   Qn=(Q0@cayley(U[n]@V[n].T-V[n]@U[n].T)).astype(np.float32)
   native_err.append(float(np.linalg.norm(Q-Qn)/max(np.linalg.norm(Q),1e-30)))
  rec_wall=time.perf_counter()-t0;Qs=np.stack(Qs);Y=np.stack(Y)
  payload=npz(Q0=Q0,U=U,V=V,scale=np.array([SCALE],np.float32),metadata=np.array([D,N,T,rank],np.int32))
  native=npz(Q0=Q0,U=U,V=V,scale=np.array([SCALE],np.float32),metadata=np.array([D,N,T,rank],np.int32))
  independent=npz(Q_views=Qs,scale=np.array([SCALE],np.float32),metadata=np.array([D,N,T],np.int32))
  hard=np.stack([traj(Q0,x0[n],inp[n]) for n in range(N)])
  # Replay the candidate from its serialized factors and compare with independent stored transitions.
  with np.load(io.BytesIO(payload),allow_pickle=False) as pm, np.load(io.BytesIO(independent),allow_pickle=False) as pi:
   Q0r=pm['Q0'];Ur=pm['U'];Vr=pm['V'];Qind=pi['Q_views']
   Ycandidate=[];Yind=[]
   for n in range(N):
    Sr=Ur[n]@Vr[n].T-Vr[n]@Ur[n].T
    Qr=(Q0r@cayley(Sr)).astype(np.float32)
    Ycandidate.append(traj(Qr,x0[n],inp[n]));Yind.append(traj(Qind[n],x0[n],inp[n]))
   Ycandidate=np.stack(Ycandidate);Yind=np.stack(Yind)
   err=float(np.linalg.norm(Ycandidate-Yind)/max(np.linalg.norm(Yind),1e-30))
  harderr=float(np.linalg.norm(Y-hard)/max(np.linalg.norm(Y),1e-30))
  spectral=float(max(np.max(np.abs(np.linalg.eigvals(SCALE*q))) for q in Qs))
  maxnorm=float(np.max(np.linalg.norm(Y,axis=-1)))
  for nm,z in [('mirror',payload),('native',native),('independent',independent)]: (outdir/f'{label}_r{rank}.{nm}.npz').write_bytes(z)
  # Dense solve, Q0*C composition, and skew construction are charged once per selected transition per T-step sequence.
  reconstruct_mac=N*((2/3)*D**3 + D**3 + 2*D*D*rank)
  recurrent_mac=N*T*D*D
  total_mac=reconstruct_mac+recurrent_mac
  rows.append({'condition':label,'seed':seed,'skew_rank':rank,'method':'shared_Q0_lowrank_skew_cayley',
   'serialized_bytes':len(payload),'independent_bytes':len(independent),'native_bytes':len(native),
   'payload_sha256':hashlib.sha256(payload).hexdigest(),'native_sha256':hashlib.sha256(native).hexdigest(),
   'train_tokens_or_examples':0,'optimizer_updates':0,'active_compute_proxy':total_mac,'independent_compute_proxy':recurrent_mac,
   'reconstruction_overhead_fraction':reconstruct_mac/recurrent_mac,'trajectory_nrmse':err,'hard_tied_trajectory_nrmse':harderr,
   'native_transition_alias_nrmse':max(native_err),'max_spectral_radius':spectral,'max_state_norm':maxnorm,
   'reconstruction_wall_s':rec_wall,'wall_time_s':rec_wall,'status_note':'synthetic driven recurrence; dense Cayley solve and Q0 composition charged per sequence'})
 return rows
def main():
 rows=[];out=ROOT/'source'/'payloads'
 for k,s in SEEDS.items():rows.extend(run_one(k,s,out))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
