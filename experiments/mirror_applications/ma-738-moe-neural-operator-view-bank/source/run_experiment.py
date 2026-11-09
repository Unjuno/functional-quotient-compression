#!/usr/bin/env python3
"""MoE-style routed Fourier-diagonal operator-bank screen."""
from __future__ import annotations
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SEED=73801;M=32;D=8;FIT=6;R=4;FIELDS=256
BANDS=((0,8),(8,16),(16,24),(24,32))

def pack(**a):
 b=io.BytesIO();np.savez_compressed(b,**a);return b.getvalue()
def nrmse(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
def run():
 rng=np.random.default_rng(SEED);k=np.arange(M,dtype=np.float32)
 base=np.exp(-.007*k*k);atoms=np.zeros((R,M),np.float32)
 for j,(lo,hi) in enumerate(BANDS):atoms[j,lo:hi]=.12+.025*np.cos(np.linspace(0,np.pi,hi-lo))
 codes=rng.uniform(-.7,.7,size=(D,R)).astype(np.float32)
 codes[:,0]+=np.linspace(-.25,.25,D)
 targets=base[None]+codes@atoms
 # Rank-4 shared PCA control trained on six operator identities only.
 center=targets[:FIT].mean(axis=0);_,_,vt=np.linalg.svd(targets[:FIT]-center,full_matrices=False)
 basis=vt[:R].astype(np.float32);coords=(targets-center)@basis.T
 pca=center[None]+coords@basis
 fields=(rng.normal(size=(D,FIELDS,M))+1j*rng.normal(size=(D,FIELDS,M))).astype(np.complex64)
 fields/=((1+k)[None,None,:]**1.2)
 # Oracle regime route. For top-2, retain own expert and blend 10% of its nearest code.
 route1=np.arange(D,dtype=np.int8)
 dist=np.sum((codes[:,None,:]-codes[None,:,:])**2,axis=-1);np.fill_diagonal(dist,np.inf)
 route2=np.stack([route1,np.argmin(dist,axis=1).astype(np.int8)],axis=1)
 mix2=np.tile(np.asarray([.9,.1],np.float32),(D,1))
 native_payload=pack(shared_mean=center.astype(np.float32),basis=basis,coordinates=coords.astype(np.float32),
   route=route1,domain_ids=np.arange(D,dtype=np.int8),fit_count=np.asarray([FIT],np.int8))
 mirror_payload=native_payload
 pca2=pack(shared_mean=center.astype(np.float32),basis=basis,coordinates=coords.astype(np.float32),route=route2,
   route_weights=mix2,domain_ids=np.arange(D,dtype=np.int8),fit_count=np.asarray([FIT],np.int8))
 full1=pack(operator_experts=targets.astype(np.float32),route=route1,domain_ids=np.arange(D,dtype=np.int8))
 full2=pack(operator_experts=targets.astype(np.float32),route=route2,route_weights=mix2,domain_ids=np.arange(D,dtype=np.int8))
 universal=pack(shared_operator=center.astype(np.float32),domain_ids=np.arange(D,dtype=np.int8))
 methods=[('independent_full',1,targets,full1),('universal_one',1,np.broadcast_to(center,targets.shape),universal),
  ('native_pca_top1',1,pca,native_payload),('mirror_view_top1',1,pca,mirror_payload)]
 mix_full=.9*targets+.1*targets[route2[:,1]]
 mix_pca=.9*pca+.1*pca[route2[:,1]]
 methods.extend([('independent_full',2,mix_full,full2),('native_pca_top2',2,mix_pca,pca2),('mirror_view_top2',2,mix_pca,pca2)])
 rows=[]
 for name,top,ops,blob in methods:
  te=[];he=[];times=[]
  for d in range(D):
   ref=fields[d]*targets[d][None,:];t0=time.perf_counter();out=fields[d]*ops[d][None,:];times.append((time.perf_counter()-t0)*1000)
   te.append(nrmse(out,ref));he.append(nrmse(out[:,24:],ref[:,24:]))
  route_compute=D*FIELDS*M*top
  if name.startswith(('native_pca','mirror_view')):route_compute+=D*M*R*top
  rows.append(dict(seed=SEED,method=name,top_k=top,contrast_group='fit+heldout',serialized_bytes=len(blob),payload_sha256=hashlib.sha256(blob).hexdigest(),total_solution_nrmse=float(np.mean(te)),high_band_solution_nrmse=float(np.mean(he)),active_spectral_multiplies=route_compute,wall_ms=float(sum(times)),status_note='exact native PCA alias' if name.startswith('mirror') else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('wrote',len(rows),'rows')
if __name__=='__main__':run()
