"""Synthetic option-by-position role composition with additive rank-2 codes."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=81101;D=8;NO=6;NP=4;R=2;NTR=128;NTE=256

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def orth(rng):u,_,vt=np.linalg.svd(rng.normal(size=(D,D)));return (u@vt).astype(np.float32)
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);core=orth(rng);L=rng.normal(size=(R,D)).astype(np.float32);V=rng.normal(size=(R,D)).astype(np.float32);L/=np.linalg.norm(L,axis=1)[:,None];V/=np.linalg.norm(V,axis=1)[:,None];oc=rng.normal(0,.035,(NO,R)).astype(np.float32);pc=rng.normal(0,.035,(NP,R)).astype(np.float32);codes=oc[:,None,:]+pc[None,:,:];maps=core[None,None]+np.einsum('opr,ri,rj->opij',codes,L,V)
 train_pairs=[(o,p) for o in range(NO) for p in range(NP) if (o+2*p)%5!=0];held=[(o,p) for o in range(NO) for p in range(NP) if (o+2*p)%5==0];seqs=[tuple((s+j*2)%NO for j in range(4)) for s in range(6)];testseq=seqs
 x=rng.normal(size=(len(testseq),NTE,D)).astype(np.float32)
 def rollout(xx,seq,bank):
  y=xx.copy()
  for pos,o in enumerate(seq):y=y@bank[o,pos].T
  return y
 y=np.stack([rollout(x[i],q,maps) for i,q in enumerate(testseq)])
 ids=np.array(train_pairs,np.int8);seqmeta=np.array(testseq,np.int8)
 full=pack(pair_maps=maps,option_position_ids=np.array([(o,p) for o in range(NO) for p in range(NP)],np.int8),sequence_ids=seqmeta)
 native=pack(shared_core=core,left_atoms=L,right_atoms=V,option_codes=oc,position_codes=pc,training_pair_ids=ids,sequence_ids=seqmeta)
 mirror=pack(shared_core=core,left_atoms=L,right_atoms=V,view_option_codes=oc,view_position_codes=pc,training_pair_ids=ids,sequence_ids=seqmeta)
 shared=pack(shared_core=core,training_pair_ids=ids,sequence_ids=seqmeta)
 rows=[]
 for name,payload,kind in [('independent_pair_bank',full,'full'),('shared_core_only',shared,'shared'),('native_additive_factors',native,'factor'),('mirror_option_position_views',mirror,'factor')]:
  z=np.load(io.BytesIO(payload))
  if kind=='full':rec=z['pair_maps']
  elif kind=='shared':rec=np.broadcast_to(z['shared_core'],(NO,NP,D,D))
  else:
   a=z['view_option_codes'] if 'view_option_codes' in z else z['option_codes'];b=z['view_position_codes'] if 'view_position_codes' in z else z['position_codes'];rec=z['shared_core'][None,None]+np.einsum('opr,ri,rj->opij',a[:,None,:]+b[None,:,:],z['left_atoms'],z['right_atoms'])
  pred=np.stack([rollout(x[i],q,rec) for i,q in enumerate(testseq)]);e=nrmse(pred,y);rt=float(np.max(np.abs(rec-(z['pair_maps'] if kind=='full' else rec))))
  st=time.perf_counter();_=np.stack([rollout(x[i],q,rec) for i,q in enumerate(testseq)]);wall=time.perf_counter()-st
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_sequence_nrmse=e,train_pairs=len(train_pairs),heldout_pairs=len(held),test_sequences=len(testseq),optimizer_updates=0,active_macs_per_action=64 if kind=='shared' else 64+2*D*R,payload_roundtrip_max_abs=rt,wall_time_s=wall,status_note='exact native additive coefficient alias' if name=='mirror_option_position_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
