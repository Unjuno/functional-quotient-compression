"""Synthetic what-by-how factorized computation operator bank."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=81601;D=8;NW=6;NH=5;R=2;N=256

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);core=rng.normal(size=(D,D)).astype(np.float32)*.1;A=rng.normal(size=(R,D)).astype(np.float32);B=rng.normal(size=(R,D)).astype(np.float32);C=rng.normal(size=(NW,R)).astype(np.float32)*.04;H=rng.normal(size=(NH,R)).astype(np.float32)*.04
 coeff=C[:,None,:]*H[None,:,:];ops=core[None,None]+np.einsum('whr,ri,rj->whij',coeff,A,B)
 tr=[(w,h) for w in range(NW) for h in range(NH) if (w+2*h)%5!=0];te=[(w,h) for w in range(NW) for h in range(NH) if (w+2*h)%5==0];x=rng.normal(size=(len(te),N,D)).astype(np.float32);y=np.stack([x[i]@ops[w,h].T for i,(w,h) in enumerate(te)])
 full=pack(pair_operators=ops,pair_ids=np.array([(w,h) for w in range(NW) for h in range(NH)],np.int8));native=pack(core=core,left_basis=A,right_basis=B,what_codes=C,how_codes=H,train_pair_ids=np.array(tr,np.int8));mirror=pack(core=core,left_basis=A,right_basis=B,view_what=C,view_how=H,train_pair_ids=np.array(tr,np.int8));shared=pack(core=core,train_pair_ids=np.array(tr,np.int8));rows=[]
 for name,payload,kind in [('independent_pair_operators',full,'full'),('shared_core_only',shared,'shared'),('native_tensor_factorization',native,'factor'),('mirror_what_how_views',mirror,'factor')]:
  z=np.load(io.BytesIO(payload))
  if kind=='full':rec=z['pair_operators']
  elif kind=='shared':rec=np.broadcast_to(z['core'],(NW,NH,D,D))
  else:
   c=z['view_what'] if 'view_what' in z else z['what_codes'];h=z['view_how'] if 'view_how' in z else z['how_codes'];rec=z['core'][None,None]+np.einsum('wr,hr,ri,rj->whij',c,h,z['left_basis'],z['right_basis'])
  pred=np.stack([x[i]@rec[w,h].T for i,(w,h) in enumerate(te)]);err=nrmse(pred,y);rt=float(np.max(np.abs(rec-(z['pair_operators'] if kind=='full' else rec))));st=time.perf_counter();_=np.stack([x[i]@rec[w,h].T for i,(w,h) in enumerate(te)]);wall=time.perf_counter()-st
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_cross_product_nrmse=err,train_pairs=len(tr),heldout_pairs=len(te),test_examples=len(te)*N,optimizer_updates=0,active_macs=64+2*D*R,payload_roundtrip_max_abs=rt,wall_time_s=wall,status_note='exact native tensor factorization alias' if name=='mirror_what_how_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
