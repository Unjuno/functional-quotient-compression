#!/usr/bin/env python3
"""Synthetic relation-transform bank comparison inspired by R-GCN."""
from __future__ import annotations
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SEED=74101;E=64;REL=32;TRAIN=24;DIM=16;NB=4;CODE=2;TRIPLES=64

def pack(**a):
 b=io.BytesIO();np.savez_compressed(b,**a);return b.getvalue()
def nrmse(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
def evaluate(emb,weights,rels):
 hits=[];rr=[];err=[];start=__import__('time').perf_counter();mult=0
 for r in rels:
  w=weights[r]
  for h in range(E):
   x=emb[h];pred=x@w
   # Deterministic positive object is the exact operator's top-ranked entity.
   truth=x@TRUE_W[r];target=int(np.argmax(emb@truth))
   scores=emb@pred;order=np.argsort(-scores);rank=int(np.flatnonzero(order==target)[0])+1
   hits.append(float(rank==1));rr.append(1/rank);err.append(np.linalg.norm(pred-truth)/max(np.linalg.norm(truth),1e-30));mult+=DIM*DIM
 return float(np.mean(err)),float(np.mean(hits)),float(np.mean(rr)),mult,(__import__('time').perf_counter()-start)*1000

def run():
 global TRUE_W
 rng=np.random.default_rng(SEED);emb=rng.normal(size=(E,DIM)).astype(np.float32);emb/=np.linalg.norm(emb,axis=1,keepdims=True)
 bases=rng.normal(0,.3,size=(NB,DIM,DIM)).astype(np.float32)
 codes=rng.normal(0,.45,size=(REL,CODE)).astype(np.float32)
 generator=rng.normal(0,.35,size=(CODE,NB)).astype(np.float32)
 coeff=codes@generator
 TRUE_W=np.eye(DIM,dtype=np.float32)[None]+np.einsum('rb,bij->rij',coeff,bases)
 mean=TRUE_W[:TRAIN].mean(axis=0)
 # Fit the two-to-four generator from training relation coefficient identities only.
 fitgen=np.linalg.lstsq(codes[:TRAIN],coeff[:TRAIN],rcond=None)[0].astype(np.float32)
 predcoeff=codes@fitgen
 mirror_w=np.eye(DIM,dtype=np.float32)[None]+np.einsum('rb,bij->rij',predcoeff,bases)
 rgcn_w=np.eye(DIM,dtype=np.float32)[None]+np.einsum('rb,bij->rij',coeff,bases)
 rows=[]
 controls=[
  ('independent_full',TRUE_W,pack(entity_embeddings=emb,relation_matrices=TRUE_W,relation_ids=np.arange(REL,dtype=np.int8)),True),
  ('rgcn_free_basis',rgcn_w,pack(entity_embeddings=emb,bases=bases,relation_coefficients=coeff,relation_ids=np.arange(REL,dtype=np.int8)),True),
  ('mirror_relation_code',mirror_w,pack(entity_embeddings=emb,bases=bases,relation_codes=codes,code_generator=fitgen,relation_ids=np.arange(REL,dtype=np.int8)),False),
  ('native_rank2_factor',mirror_w,pack(entity_embeddings=emb,bases=bases,relation_codes=codes,code_generator=fitgen,relation_ids=np.arange(REL,dtype=np.int8)),False),
  ('universal_tied',np.broadcast_to(mean,(REL,DIM,DIM)),pack(entity_embeddings=emb,shared_relation_matrix=mean,relation_ids=np.arange(REL,dtype=np.int8)),True)]
 for name,w,payload,oracle in controls:
  for split,rels in [('train',list(range(TRAIN))),('heldout',list(range(TRAIN,REL)))]:
   # Quality evaluation includes operator outputs and entity ranking over held-out relations.
   err,hit,mrr,mult,ms=evaluate(emb,w,rels)
   if name in ('rgcn_free_basis','mirror_relation_code','native_rank2_factor'):
    mult+=len(rels)*(NB*DIM*DIM+CODE*NB)
   rows.append(dict(seed=SEED,method=name,split=split,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),operator_nrmse=err,hits_at_1=hit,mrr=mrr,message_multiply_proxy=mult,wall_ms=ms,status_note='oracle relation coeffs' if oracle else ('exact native factor alias' if name=='mirror_relation_code' else '')))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('wrote',len(rows),'rows')
if __name__=='__main__':run()
