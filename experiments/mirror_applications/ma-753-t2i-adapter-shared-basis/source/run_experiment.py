#!/usr/bin/env python3
"""Synthetic shared-basis bank for plug-and-play spatial adapters."""
from __future__ import annotations
import csv,hashlib,io,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SEED=75301; N=12; TRAIN=8; IN=16; OUT=12; RANK=2; CAL=8

def pack(**a):
    b=io.BytesIO(); np.savez_compressed(b,**a); return b.getvalue()

def run():
    rng=np.random.default_rng(SEED)
    # Synthetic adapter deltas lie in a shared rank-2 function subspace.
    # The basis is estimated from training conditions only; heldout codes are
    # estimated from CAL input/output calibration examples.
    q,_=np.linalg.qr(rng.normal(size=(IN*OUT,RANK)))
    basis=q.T.reshape(RANK,OUT,IN).astype(np.float32)
    codes=rng.normal(size=(N,RANK)).astype(np.float32)
    deltas=np.einsum('nr,roi->noi',codes,basis)
    x=rng.normal(size=(N,CAL,IN)).astype(np.float32)
    y=np.einsum('noi,nbi->nbo',deltas,x)
    # Train-side basis estimate via SVD across the eight trained adapter deltas.
    U,S,Vt=np.linalg.svd(deltas[:TRAIN].reshape(TRAIN,-1),full_matrices=False)
    learned=Vt[:RANK].reshape(RANK,OUT,IN).astype(np.float32)
    # fit task codes on all train conditions, and heldout codes on calibration pairs.
    train_codes=np.linalg.lstsq(learned.reshape(RANK,-1).T,deltas[:TRAIN].reshape(TRAIN,-1).T,rcond=None)[0].T.astype(np.float32)
    held_codes=[]
    for n in range(TRAIN,N):
        design=np.einsum('roi,bi->bro',learned,x[n]).transpose(0,2,1).reshape(CAL*OUT,RANK)
        held_codes.append(np.linalg.lstsq(design,y[n].reshape(-1),rcond=None)[0])
    all_codes=np.concatenate([train_codes,np.asarray(held_codes,dtype=np.float32)],axis=0)
    reconstructed=np.einsum('nr,roi->noi',all_codes,learned)
    test_x=rng.normal(size=(N,32,IN)).astype(np.float32)
    test_y=np.einsum('noi,nbi->nbo',deltas,test_x)
    independent=pack(adapter_deltas=deltas)
    # Native PCA and Mirror are deliberately stored in the same canonical representation.
    shared=pack(shared_basis=learned,condition_codes=all_codes)
    lowrank=pack(per_condition_A=np.stack([np.linalg.svd(d,full_matrices=False)[0][:,:RANK] for d in deltas]),
                 per_condition_s=np.stack([np.linalg.svd(d,full_matrices=False)[1][:RANK] for d in deltas]),
                 per_condition_V=np.stack([np.linalg.svd(d,full_matrices=False)[2][:RANK] for d in deltas]))
    # Compare composition of two selected controls against direct addition.
    pairs=[(8,9),(9,11),(0,10),(2,7)]
    rows=[]
    def add_composition(weights):
        es=[]
        for a,b in pairs:
            ref=deltas[a]+deltas[b]; pred=weights[a]+weights[b]
            es.append(np.linalg.norm(pred-ref)/np.linalg.norm(ref))
        return float(np.mean(es))
    for name,blob,wts,extra in [
      ('independent_full_adapters',independent,deltas,0),
      ('independent_rank2_adapter_control',lowrank,np.stack([np.linalg.svd(d,full_matrices=False)[0][:,:RANK]@np.diag(np.linalg.svd(d,full_matrices=False)[1][:RANK])@np.linalg.svd(d,full_matrices=False)[2][:RANK] for d in deltas]),0),
      ('native_shared_PCA_basis',shared,reconstructed,0),
      ('mirror_shared_adapter_views',shared,reconstructed,0)]:
      per=[]
      t0=time.perf_counter()
      for n in range(N):
        pred=np.einsum('oi,bi->bo',wts[n],test_x[n]); truth=test_y[n]
        per.append(np.linalg.norm(pred-truth)/np.linalg.norm(truth))
      composition=add_composition(wts); elapsed=(time.perf_counter()-t0)*1000
      # Count all N individual-condition and four two-control requests.
      # Dense adapters cost IN*OUT per token; shared rank-R basis costs R times that.
      n_requests=N+len(pairs)
      macs=n_requests*32*IN*OUT*(RANK if name.startswith(('mirror','native')) else (RANK if name.startswith('independent_rank2') else 1))
      rows.append(dict(seed=SEED,method=name,serialized_bytes=len(blob),payload_sha256=hashlib.sha256(blob).hexdigest(),
          heldout_condition_nrmse=float(np.mean(per[TRAIN:])),all_condition_nrmse=float(np.mean(per)),
          composition_nrmse=composition,active_macs=macs,wall_ms=elapsed,updates=0,
          calibration_examples=N*CAL,status_note='exact native shared-PCA alias' if name=='mirror_shared_adapter_views' else ''))
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('wrote',len(rows),'rows; rank',RANK,'train conditions',TRAIN,'heldout conditions',N-TRAIN)
if __name__=='__main__':run()
