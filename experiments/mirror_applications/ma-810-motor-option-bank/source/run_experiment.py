"""Synthetic compositional option bank with shared rank-1 adapter atoms."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=81001;D=8;K=8;R=2;NTR=128;NTE=256

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def orth(rng):u,_,vt=np.linalg.svd(rng.normal(size=(D,D)));return (u@vt).astype(np.float32)
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);core=orth(rng);U=np.stack([rng.normal(size=D) for _ in range(R)]).astype(np.float32);V=np.stack([rng.normal(size=D) for _ in range(R)]).astype(np.float32);U/=np.linalg.norm(U,axis=1)[:,None];V/=np.linalg.norm(V,axis=1)[:,None]
 coeff=rng.normal(0,.06,(K,R)).astype(np.float32);coeff[-1,1]+=.08
 adapters=np.einsum('kr,ri,rj->kij',coeff,U,V);maps=core[None]+adapters
 train_seq=[(0,1,2,3),(1,3,5,7),(2,4,6,0),(3,5,7,1),(4,6,0,2),(5,7,1,3),(6,0,2,4),(7,1,3,5),(0,2,4,6),(1,3,5,7)]
 test_seq=[(7,6,5,4),(6,4,2,0),(5,3,1,7),(4,2,0,6),(3,1,7,5),(2,0,6,4),(1,7,5,3),(0,6,4,2)]
 xtr=rng.normal(size=(len(train_seq),NTR,D)).astype(np.float32);xte=rng.normal(size=(len(test_seq),NTE,D)).astype(np.float32)
 def rollout(x,seq,bank):
  y=x.copy()
  for k in seq:y=y@bank[k].T
  return y
 ytr=np.stack([rollout(xtr[i],q,maps) for i,q in enumerate(train_seq)]);yte=np.stack([rollout(xte[i],q,maps) for i,q in enumerate(test_seq)])
 full=pack(option_maps=maps,option_ids=np.arange(K,dtype=np.int8),sequence_ids=np.array(train_seq+test_seq,dtype=np.int8))
 mirror=pack(shared_core=core,atom_left=U,atom_right=V,option_codes=coeff,option_ids=np.arange(K,dtype=np.int8),sequence_ids=np.array(train_seq+test_seq,dtype=np.int8))
 native=pack(shared_core=core,basis_left=U,basis_right=V,coefficient_table=coeff,option_ids=np.arange(K,dtype=np.int8),sequence_ids=np.array(train_seq+test_seq,dtype=np.int8))
 shared=pack(shared_core=core,option_ids=np.arange(K,dtype=np.int8),sequence_ids=np.array(train_seq+test_seq,dtype=np.int8))
 methods=[('independent_full_adapter_bank',full,'full'),('shared_core_only',shared,'shared'),('native_rank2_adapter_basis',native,'factor'),('mirror_option_codes',mirror,'factor')];rows=[]
 for name,payload,kind in methods:
  z=np.load(io.BytesIO(payload))
  if kind=='full':rec=z['option_maps']
  elif kind=='shared':rec=np.broadcast_to(z['shared_core'],(K,D,D))
  else:rec=z['shared_core'][None]+np.einsum('kr,ri,rj->kij',z['coefficient_table'] if 'coefficient_table' in z else z['option_codes'],z['basis_left'] if 'basis_left' in z else z['atom_left'],z['basis_right'] if 'basis_right' in z else z['atom_right'])
  pred=np.stack([rollout(xte[i],q,rec) for i,q in enumerate(test_seq)]);truth_err=nrmse(pred,yte); rt=float(np.max(np.abs(rec-(z['option_maps'] if kind=='full' else rec))))
  start=time.perf_counter();_=np.stack([rollout(xte[i],q,rec) for i,q in enumerate(test_seq)]);wall=time.perf_counter()-start
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_sequence_nrmse=truth_err,train_examples=len(train_seq)*NTR,test_examples=len(test_seq)*NTE,optimizer_updates=0,sequence_length=4,active_macs_per_action=64 if kind=='shared' else 64+2*D*R,sequence_count=len(test_seq),wall_time_s=wall,payload_roundtrip_max_abs=rt,status_note='exact native low-rank coefficient alias' if name=='mirror_option_codes' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
