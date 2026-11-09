#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SEED=74201;E=64;REL=32;DIM=16;NB=4;TRAIN_COMBOS=6;REPS=4

def pack(**a):
 b=io.BytesIO();np.savez_compressed(b,**a);return b.getvalue()
def nrmse(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
def evaluate(emb,ws,truth,ids):
 err=[];hits=[];mult=0;t=time.perf_counter()
 for r in ids:
  for h in range(E):
   pred=emb[h]@ws[r];y=emb[h]@truth[r];target=int(np.argmax(emb@y));order=np.argsort(-(emb@pred));rank=int(np.where(order==target)[0][0]);hits.append(rank==0);err.append(np.linalg.norm(pred-y)/max(np.linalg.norm(y),1e-30));mult+=DIM*DIM
 return float(np.mean(err)),float(np.mean(hits)),mult,(time.perf_counter()-t)*1000

def run():
 rng=np.random.default_rng(SEED);emb=rng.normal(size=(E,DIM)).astype(np.float32);emb/=np.linalg.norm(emb,axis=1,keepdims=True)
 bases=rng.normal(0,.28,size=(NB,DIM,DIM)).astype(np.float32)
 combos=np.asarray([[a,b,c] for a in (0,1) for b in (0,1) for c in (0,1)],np.int8)
 attrs=np.repeat(combos,REPS,axis=0);phi=np.concatenate([np.ones((REL,1),np.float32),attrs.astype(np.float32)],axis=1)
 generator=rng.normal(0,.5,size=(4,NB)).astype(np.float32);coeff=phi@generator
 truew=np.eye(DIM,dtype=np.float32)[None]+np.einsum('rb,bij->rij',coeff,bases)
 train_ids=np.flatnonzero(np.repeat(np.arange(8)<TRAIN_COMBOS,REPS));held_ids=np.flatnonzero(np.repeat(np.arange(8)>=TRAIN_COMBOS,REPS))
 fitgen=np.linalg.lstsq(phi[train_ids],coeff[train_ids],rcond=None)[0].astype(np.float32)
 attr_coeff=phi@fitgen;attrw=np.eye(DIM,dtype=np.float32)[None]+np.einsum('rb,bij->rij',attr_coeff,bases)
 freecoeff=coeff.copy();freecoeff[held_ids]=coeff[train_ids].mean(axis=0)
 freew=np.eye(DIM,dtype=np.float32)[None]+np.einsum('rb,bij->rij',freecoeff,bases)
 mean=truew[train_ids].mean(axis=0);univ=np.broadcast_to(mean,truew.shape)
 full=pack(entity_embeddings=emb,relation_matrices=truew,attributes=attrs,relation_ids=np.arange(REL,np.int8) if False else np.arange(REL,dtype=np.int8))
 rgcn=pack(entity_embeddings=emb,bases=bases,relation_coefficients=freecoeff,attributes=attrs,relation_ids=np.arange(REL,dtype=np.int8))
 attr=pack(entity_embeddings=emb,bases=bases,attributes=attrs,coefficient_generator=fitgen,relation_ids=np.arange(REL,dtype=np.int8))
 native=attr
 uni=pack(entity_embeddings=emb,shared_matrix=mean,attributes=attrs,relation_ids=np.arange(REL,dtype=np.int8))
 methods=[('independent_full',truew,full),('rgcn_free_coefficients',freew,rgcn),('native_attribute_generator',attrw,attr),('mirror_attribute_code',attrw,native),('universal_tied',univ,uni)]
 rows=[]
 for name,ws,blob in methods:
  for split,ids in [('fit',train_ids),('heldout',held_ids)]:
   err,hit,mult,ms=evaluate(emb,ws,truew,ids)
   if name in ('rgcn_free_coefficients','native_attribute_generator','mirror_attribute_code'):mult+=len(ids)*NB*DIM*DIM
   rows.append(dict(seed=SEED,method=name,split=split,serialized_bytes=len(blob),payload_sha256=hashlib.sha256(blob).hexdigest(),operator_nrmse=err,hits_at_1=hit,message_multiply_proxy=mult,wall_ms=ms,status_note='exact native attribute factorization' if name=='mirror_attribute_code' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('wrote',len(rows),'rows')
if __name__=='__main__':run()
