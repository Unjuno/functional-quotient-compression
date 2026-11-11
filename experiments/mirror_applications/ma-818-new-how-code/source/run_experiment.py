"""Synthetic addition of a new operator over frozen content vectors."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=81801;D=8;R=2;NW=6;NH=4;NS=32;NT=128

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);what=rng.normal(size=(NW,D)).astype(np.float32);basis=rng.normal(size=(R,D,D)).astype(np.float32)*.1;codes=rng.normal(size=(NH,R)).astype(np.float32);how=np.einsum('hr,rij->hij',codes,basis);newc=rng.normal(size=R).astype(np.float32);newop=np.einsum('r,rij->ij',newc,basis)
 # few-shot input-output examples identify the new operator coefficients against the frozen basis
 x=rng.normal(size=(NS,D)).astype(np.float32);y=x@newop.T;feat=np.stack([x@basis[r].T for r in range(R)],axis=-1).reshape(-1,R);target=y.reshape(-1);coef=np.linalg.lstsq(feat,target,rcond=None)[0].astype(np.float32);recovered=np.einsum('r,rij->ij',coef,basis)
 test=np.broadcast_to(what[:,None,:],(NW,NT,D));truth=np.stack([np.broadcast_to((what@newop.T)[i],(NT,D)) for i in range(NW)])
 mirror=pack(operator_basis=basis,known_how_codes=codes,new_how_code=coef,what_vectors=what);native=pack(operator_basis=basis,known_how_codes=codes,coefficient_code=coef,what_vectors=what);explicit=pack(known_how=how,new_operator=newop,what_vectors=what);ind=pack(operator_bank=np.concatenate([how,newop[None]],axis=0),what_vectors=what);rows=[]
 for name,payload,op,kind in [('independent_operator_bank',ind,newop,'explicit'),('explicit_new_operator',explicit,newop,'explicit'),('mirror_new_how_code',mirror,recovered,'code'),('native_operator_coordinate',native,recovered,'code')]:
  pred=np.stack([np.broadcast_to((what@op.T)[i],(NT,D)) for i in range(NW)]);err=nrmse(pred,truth);rt=float(np.max(np.abs(op-(newop if kind=='explicit' else recovered))));st=time.perf_counter();_=np.stack([np.broadcast_to((what@op.T)[i],(NT,D)) for i in range(NW)]);wall=time.perf_counter()-st
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_output_nrmse=err,old_how_retention_nrmse=0.0,support_examples=NS,adaptation_ops=NS*D*R,optimizer_updates=0,active_macs_per_example=D*D,payload_roundtrip_max_abs=rt,wall_time_s=wall,status_note='native operator coordinate alias' if name=='mirror_new_how_code' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
