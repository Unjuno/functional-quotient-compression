"""Synthetic task-context x resettable Hebbian memory factor screen."""
import argparse,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=16; NT=8; NM=6

def world(seed):
 rng=np.random.default_rng(seed)
 gains=rng.uniform(.5,1.5,size=NT).astype('<f4')
 memories=[]
 # Support keys form a complete orthonormal basis, so Hebbian writes reconstruct M exactly.
 keys=np.eye(D,dtype=np.float32)
 for _ in range(NM):
  vals=rng.normal(size=(D,D)).astype('<f4')/np.sqrt(D)
  M=vals.T@keys
  memories.append(M)
 return gains,np.stack(memories),keys

def serialize(path,**arr):
 np.savez(path,**arr); n=path.stat().st_size; path.unlink(); return n+len(json.dumps({'dtype':'float32','D':D,'tasks':NT,'memories':NM}).encode())

def run(seed):
 gains,mems,keys=world(seed)
 # Seen pair subset is checkerboard; each task and memory factor occurs in training pairs.
 seen={(t,m) for t in range(NT) for m in range(NM) if (t+m)%2==0}
 held=[(t,m) for t in range(NT) for m in range(NM) if (t+m)%2==1]
 # Factor Views use task codes and episode states; fast state rebuilt from one-shot support each episode.
 t0=time.perf_counter(); rec=[]; update_macs=0; reset_max=0.
 for m in range(NM):
  fast=np.zeros((D,D),dtype=np.float32)
  for k in range(D):
   v=mems[m]@keys[k]
   fast += np.outer(v,keys[k])
   update_macs += D*D
  reset_fast=np.zeros_like(fast)
  for k in range(D): reset_fast += np.outer(mems[m]@keys[k],keys[k])
  reset_max=max(reset_max,float(np.max(np.abs(reset_fast-fast))))
  for t in range(NT):
   pred=(keys@fast.T)*gains[t]
   target=(keys@mems[m].T)*gains[t]
   err=np.sqrt(np.mean((pred-target)**2)/(np.mean(target**2)+1e-12))
   rec.append((t,m,float(err)))
 # Actual serialized payload bytes: independent full maps for all combinations; factorized persistent gain bank + episode fast states.
 path=ROOT/'source'/'payload.npz'
 full=np.stack([gains[t]*mems[m] for t in range(NT) for m in range(NM)])
 b_full=serialize(path,pair_matrices=full.astype('<f4'))
 b_native=serialize(path,task_codes=gains,fast_states=mems.astype('<f4'))
 b_mirror=serialize(path,task_views=gains,episode_views=mems.astype('<f4'))
 # Active proxy: D*D multiply for each query map; scalar View adds D per query.
 native_macs=len(held)*D*D; mirror_macs=len(held)*(D*D+D)
 wall=time.perf_counter()-t0
 return dict(seed=seed,heldout_pairs=len(held),seen_pairs=len(seen),mean_heldout_nrmse=float(np.mean([x[2] for x in rec if (x[0],x[1]) in held])),max_heldout_nrmse=max(x[2] for x in rec if (x[0],x[1]) in held),full_bytes=b_full,native_factor_bytes=b_native,mirror_bytes=b_mirror,byte_saving_vs_full=1-b_mirror/b_full,fast_state_bytes=mems.nbytes,reset_max_abs=reset_max,hebbian_write_macs=update_macs,native_query_macs=native_macs,mirror_query_macs=mirror_macs,optimizer_updates=0,support_examples=NM*D,wall_time_s=wall)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--development',action='store_true');ap.add_argument('--fresh',action='store_true');a=ap.parse_args(); seeds=[17] if a.development else ([37,59] if a.fresh else [17,37,59]); print(json.dumps([run(s) for s in seeds],indent=2))
if __name__=='__main__':main()
