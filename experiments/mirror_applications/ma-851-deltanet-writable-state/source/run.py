"""Synthetic Delta-rule full versus shared-subspace writable-state screen."""
import argparse,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=32; R=2; SEQ=16; NQUERY=24

def payload(path,**arrays):
 np.savez(path,**arrays); n=path.stat().st_size
 with np.load(path) as z:
  assert set(z.files)==set(arrays)
  for k,v in arrays.items(): assert np.array_equal(z[k],v)
 path.unlink()
 return n+len(json.dumps({'D':D,'rank':R,'seq':SEQ,'dtype':'f32'}).encode())

def run(seed):
 rng=np.random.default_rng(seed)
 U,_=np.linalg.qr(rng.normal(size=(D,R))); U=U.astype('<f4')
 q,_=np.linalg.qr(rng.normal(size=(D,D))); keys=q[:,:SEQ].astype('<f4')
 coeff=rng.normal(size=(SEQ,R)).astype('<f4'); vals=coeff@U.T
 queries=np.concatenate([keys.T,rng.normal(size=(NQUERY-SEQ,D)).astype('<f4')/np.sqrt(D)],axis=0)
 target=queries@((U@coeff.T@keys.T).T)
 W=np.zeros((D,D),dtype=np.float32); C=np.zeros((R,D),dtype=np.float32)
 full_update=low_update=0; t0=time.perf_counter()
 for i in range(SEQ):
  k=keys[:,i]; v=vals[i]
  # Delta rule on full matrix.
  e=v-W@k; full_update+=2*D*D+D; W+=np.outer(e,k)
  # Same delta rule projected into the shared output subspace.
  pred=U@(C@k); e2=v-pred; C+=np.outer(U.T@e2,k)
  low_update+=4*D*R
 full_pred=queries@W.T
 low_pred=(queries@C.T)@U.T
 ferr=np.sqrt(np.mean((full_pred-target)**2)/(np.mean(target**2)+1e-12))
 lerr=np.sqrt(np.mean((low_pred-target)**2)/(np.mean(target**2)+1e-12))
 path=ROOT/'source'/'payload.npz'
 full_bytes=payload(path,state=W.astype('<f4'))
 native_bytes=payload(path,basis=U,state=C.astype('<f4'))
 mirror_bytes=payload(path,view_basis=U,state_code=C.astype('<f4'))
 wall=time.perf_counter()-t0
 return dict(seed=seed,full_nrmse=float(ferr),native_lowrank_nrmse=float(lerr),mirror_nrmse=float(lerr),full_bytes=full_bytes,native_lowrank_bytes=native_bytes,mirror_bytes=mirror_bytes,full_state_bytes=W.nbytes,lowrank_basis_bytes=U.nbytes,lowrank_writable_state_bytes=C.nbytes,full_update_macs=full_update,lowrank_update_macs=low_update,full_query_macs=NQUERY*D*D,lowrank_query_macs=NQUERY*2*D*R,support_examples=SEQ,updates=SEQ,wall_time_s=wall)

def main():
 p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');p.add_argument('--fresh',action='store_true');a=p.parse_args();seeds=[19] if a.development else ([41,67] if a.fresh else [19,41,67]);print(json.dumps([run(s) for s in seeds],indent=2))
if __name__=='__main__':main()
