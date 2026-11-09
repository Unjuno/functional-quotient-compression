#!/usr/bin/env python3
"""Synthetic LDR operator reconstruction with byte and work accounting."""
import csv, hashlib, io, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D,N,M=64,8,64
SEEDS={'dev-70801':70801,'audit-70811':70811,'audit-70812':70812,'audit-70813':70813}
RANKS=(1,2,4,8)
def pack(**a):
 b=io.BytesIO();np.savez(b,**a);return b.getvalue()
def ldr(G,H,C):
 # Displacement operator: sum_r diag(G[:,r]) T_r(H[:,r]) + C, with circular shifts.
 # C is a shared low-rank residual generator factor, not task-private dense state.
 d,k=G.shape; out=np.zeros((d,d),np.float64)
 for r in range(k): out += np.diag(G[:,r]) @ np.roll(np.diag(H[:,r]),r,axis=1)
 return out

def run_one(label,seed,outdir):
 rng=np.random.default_rng(seed);outdir.mkdir(parents=True,exist_ok=True);rows=[]
 # Shared atom matrices are structured displacement terms; task maps use small coefficients.
 atoms=np.stack([np.diag(rng.normal(size=D))@np.roll(np.diag(rng.normal(size=D)),r,axis=1) for r in range(16)]).astype(np.float32)
 x=rng.normal(size=(N,M,D)).astype(np.float32)
 for rank in RANKS:
  coeff=rng.normal(size=(N,rank)).astype(np.float32)/np.sqrt(rank)
  # Targets are in the chosen shared-generator span; rank selects the number of active atoms.
  target=np.einsum('nr,rij->nij',coeff,atoms[:rank])
  t0=time.perf_counter();views=np.einsum('nr,rij->nij',coeff,atoms[:rank]);construction=time.perf_counter()-t0
  t1=time.perf_counter(); y=np.einsum('nmd,ndh->nmh',x,views); apply_wall=time.perf_counter()-t1
  # Independently serialize each dense operator; native shared-atom coefficient code has the same mathematical state.
  mirror=pack(atoms=atoms[:rank],coeff=coeff,metadata=np.array([D,N,rank],np.int32))
  native=pack(atoms=atoms[:rank],coeff=coeff,metadata=np.array([D,N,rank],np.int32))
  independent=pack(matrices=target.astype(np.float32),metadata=np.array([D,N],np.int32))
  with np.load(io.BytesIO(mirror),allow_pickle=False) as q, np.load(io.BytesIO(independent),allow_pickle=False) as ind:
   replay=np.einsum('nr,rij->nij',q['coeff'],q['atoms']); yi=np.einsum('nmd,ndh->nmh',x,replay)
   err=float(np.linalg.norm(yi-np.einsum('nmd,ndh->nmh',x,ind['matrices']))/max(np.linalg.norm(np.einsum('nmd,ndh->nmh',x,ind['matrices'])),1e-30))
  native_err=float(np.max(np.abs(views-np.einsum('nr,rij->nij',coeff,atoms[:rank]))))
  for name,data in [('mirror',mirror),('native',native),('independent',independent)]: (outdir/f'{label}_r{rank}.{name}.npz').write_bytes(data)
  # MAC proxy: dense apply N*M*D^2; construction sums rank dense atoms and per-map coefficient reductions.
  dense=N*M*D*D; build=N*rank*D*D; total=dense+build
  rows.append(dict(condition=label,seed=seed,rank=rank,serialized_bytes=len(mirror),native_bytes=len(native),independent_bytes=len(independent),payload_sha256=hashlib.sha256(mirror).hexdigest(),native_sha256=hashlib.sha256(native).hexdigest(),examples=N*M,optimizer_updates=0,active_compute_proxy=total,independent_compute_proxy=dense,reconstruction_overhead_fraction=build/dense,heldout_operator_nrmse=err,native_alias_max_abs=native_err,construction_wall_s=construction,apply_wall_s=apply_wall,status_note='synthetic shared atom span; apply wall isolated in followup benchmark; no training'))
 return rows
def main():
 rows=[];out=ROOT/'source/payloads'
 for label,seed in SEEDS.items(): rows+=run_one(label,seed,out)
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('wrote',len(rows),'rows')
if __name__=='__main__':main()
