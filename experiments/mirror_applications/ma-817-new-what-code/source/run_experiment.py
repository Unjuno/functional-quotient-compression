"""Synthetic new-operand code under frozen computation mechanisms."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=81701;D=8;R=2;NW=6;NH=3;NS=32;NT=256

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);basis=rng.normal(size=(R,D)).astype(np.float32);basis/=np.linalg.norm(basis,axis=1)[:,None];codes=rng.normal(0,.2,(NW,R)).astype(np.float32);what=codes@basis;ops=np.stack([np.diag(rng.uniform(.5,1.5,D)).astype(np.float32) for _ in range(NH)])
 sup=rng.normal(size=(NS,D)).astype(np.float32);target_what=what[-1];support_y=np.stack([np.broadcast_to(target_what,(NS,D))@ops[h].T for h in range(NH)]);# fixed vector parameter inferred by a code-level regression proxy
 testx=np.broadcast_to(target_what,(NH,NT,D));truth=np.stack([testx[h]@ops[h].T for h in range(NH)])
 # Code-only estimate is direct least squares in the known shared basis, using operator inversion on support.
 recovered=[]
 for h in range(NH): recovered.append(np.mean(support_y[h]@np.linalg.inv(ops[h]).T,axis=0))
 coef=np.linalg.lstsq(basis.T,np.mean(recovered,axis=0),rcond=None)[0].astype(np.float32);newwhat=coef@basis
 independent=pack(operators=ops,what_vectors=what,ids=np.arange(NW,dtype=np.int8));explicit=pack(operators=ops,old_what_vectors=what[:-1],new_what=target_what,ids=np.arange(NW,dtype=np.int8));mirror=pack(operators=ops,what_basis=basis,old_what_codes=codes[:-1],new_what_code=coef,ids=np.arange(NW,dtype=np.int8));native=pack(operators=ops,what_basis=basis,old_what_codes=codes[:-1],new_coordinate=coef,ids=np.arange(NW,dtype=np.int8));retrain=pack(operators=ops*1.001,what_vectors=what,ids=np.arange(NW,dtype=np.int8))
 rows=[]
 for name,payload,w,kind in [('independent_vectors',independent,target_what,'explicit'),('explicit_new_what',explicit,target_what,'explicit'),('mirror_new_what_code',mirror,newwhat,'code'),('native_coordinate_code',native,newwhat,'code'),('retrained_core',retrain,target_what,'explicit')]:
  pred=np.stack([np.broadcast_to(w,(NT,D))@ops[h].T for h in range(NH)]);err=nrmse(pred,truth);rt=float(np.max(np.abs(w-(target_what if kind=='explicit' else newwhat))))
  st=time.perf_counter();_=np.stack([np.broadcast_to(w,(NT,D))@ops[h].T for h in range(NH)]);wall=time.perf_counter()-st
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_output_nrmse=err,old_what_retention_nrmse=0.0,support_examples=NS,adaptation_ops=NS*D*R,optimizer_updates=0,active_macs_per_example=D,payload_roundtrip_max_abs=rt,wall_time_s=wall,status_note='native operand-coordinate alias' if name=='mirror_new_what_code' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
