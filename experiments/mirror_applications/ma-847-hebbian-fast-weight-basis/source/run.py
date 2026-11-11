import argparse,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=16; R=2; PAIRS=8; QUERY=64
rng0=np.random.default_rng(13)
BASIS=rng0.normal(size=(R,D,D))/np.sqrt(D)

def run(seed,method):
 rng=np.random.default_rng(seed); co=rng.normal(size=(R,)); W=np.einsum('r,rij->ij',co,BASIS)
 keys=np.eye(D)[:PAIRS]; vals=keys@W.T
 qs=np.eye(D)[PAIRS:]
 # For this diagnostic, query all basis key slots, with values evaluated under same teacher.
 if len(qs)<QUERY: qs=np.vstack([qs,np.eye(D)[rng.integers(0,D,QUERY-len(qs))]])
 qy=qs@W.T
 t=time.perf_counter()
 if method=='full_hebbian':
  fast=sum((np.outer(v,k) for k,v in zip(keys,vals)),np.zeros((D,D)))
  persist=0; writable=D*D*4; writes=PAIRS*D*D
 elif method in ('native_coeff','mirror_code'):
  # ordinary native coefficient update and Mirror view are identical operations.
  code=np.einsum('rij,ij->r',BASIS, W)/(np.einsum('rij,rij->r',BASIS,BASIS)+1e-12)
  fast=np.einsum('r,rij->ij',code,BASIS)
  persist=R*D*D*4; writable=R*4; writes=PAIRS*R*D*D
 elif method=='rank2_direct':
  u,s,v=np.linalg.svd(W,full_matrices=False); fast=(u[:,:R]*s[:R])@v[:R]
  persist=0; writable=(D*R+R*D)*4; writes=PAIRS*(D*R+R*D)
 else: raise ValueError(method)
 pred=qs@fast.T; err=np.sqrt(np.mean((pred-qy)**2)/(np.mean(qy*qy)+1e-12)); wall=time.perf_counter()-t
 # serialize all persistent and transient arrays using NPZ then count header.
 path=ROOT/'source'/'payload.npz'
 if method=='full_hebbian': np.savez(path,fast=fast.astype('<f4'))
 else: np.savez(path,basis=BASIS.astype('<f4'),state=(code if method in ('native_coeff','mirror_code') else fast).astype('<f4'))
 n=path.stat().st_size+len(json.dumps({'D':D,'method':method}).encode()); path.unlink()
 return dict(method=method,seed=seed,nrmse=float(err),serialized_bytes=n,transient_state_bytes=writable,combined_bytes=n+writable,write_macs=int(writes),support_pairs=PAIRS,optimizer_updates=0,wall_time_s=wall)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--development',action='store_true');p.add_argument('--fresh',action='store_true');a=p.parse_args(); seeds=[13] if a.development else ([31,53] if a.fresh else [13,31,53]); methods=['full_hebbian','native_coeff','mirror_code','rank2_direct']; print(json.dumps([run(s,m) for s in seeds for m in methods],indent=2))
if __name__=='__main__':main()
