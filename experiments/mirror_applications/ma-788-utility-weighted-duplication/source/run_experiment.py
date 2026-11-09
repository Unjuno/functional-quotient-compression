"""Utility-ranked physical copying vs shared rank-1 logical codes."""
import csv,hashlib,io
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=78801;N=16;D=12

def pack(**x):
 b=io.BytesIO();np.savez_compressed(b,**x);return b.getvalue()
def weighted_nrmse(p,t,w):return float(np.sqrt(np.sum((p-t)**2*w[:,None,None])/np.sum(t*t*w[:,None,None])))
def run():
 rng=np.random.default_rng(SEED);parent=rng.normal(0,.2,(D,D)).astype(np.float32);basis=rng.normal(0,.07,(D,D)).astype(np.float32);basis/=np.linalg.norm(basis)
 utility=np.arange(N,0,-1,dtype=np.float32);codes=rng.normal(0,.35,N).astype(np.float32);noise=rng.normal(0,.01,(N,D,D)).astype(np.float32);noise[:2]+=rng.normal(0,.15,(2,D,D)).astype(np.float32)
 teacher=parent[None]+codes[:,None,None]*basis[None]+noise
 # Shared basis and code state; complete child weights are only paid for physical copies.
 fits=[]
 for i in range(N): fits.append(np.sum((teacher[i]-parent)*basis)/(np.sum(basis*basis)+1e-12))
 fits=np.asarray(fits,np.float32);view=parent[None]+fits[:,None,None]*basis[None]
 rows=[];view_payload=pack(parent=parent,basis=basis,codes=fits,utility_weights=utility,child_ids=np.arange(N,dtype=np.int16))
 for name,selection in [('all_view',[]),('utility_hybrid_k2',[0,1]),('random_hybrid_k2',[3,11]),('utility_hybrid_k4',[0,1,2,3]),('utility_hybrid_k8',list(range(8))),('all_physical_copy',list(range(N)))]:
  sel=np.asarray(selection,dtype=np.int16);pred=view.copy();pred[sel]=teacher[sel]
  arrays={'parent':parent,'basis':basis,'codes':fits,'utility_weights':utility,'child_ids':np.arange(N,dtype=np.int16)}
  if len(sel): arrays.update(physical_ids=sel,physical_weights=teacher[sel])
  payload=pack(**arrays) if len(sel)<N else pack(weights=teacher,utility_weights=utility,child_ids=np.arange(N,dtype=np.int16))
  macs=np.where(np.isin(np.arange(N),sel),D*D,2*D*D)
  rows.append({'seed':SEED,'method':name,'physical_copies':len(sel),'serialized_bytes':len(payload),'payload_sha256':hashlib.sha256(payload).hexdigest(),'utility_weighted_nrmse':weighted_nrmse(pred,teacher,utility),'all_child_nrmse':float(np.sqrt(np.mean((pred-teacher)**2)/(np.mean(teacher**2)+1e-12))),'utility_weighted_macs_per_child':float(np.average(macs,weights=utility)),'optimizer_updates':0,'view_payload_bytes':len(view_payload),'status_note':''})
 rows.append({'seed':SEED,'method':'native_rank1_lora_all_view','physical_copies':0,'serialized_bytes':len(view_payload),'payload_sha256':hashlib.sha256(view_payload).hexdigest(),'utility_weighted_nrmse':weighted_nrmse(view,teacher,utility),'all_child_nrmse':float(np.sqrt(np.mean((view-teacher)**2)/(np.mean(teacher**2)+1e-12))),'utility_weighted_macs_per_child':2*D*D,'optimizer_updates':0,'view_payload_bytes':len(view_payload),'status_note':'exact native rank-1 LoRA alias'})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
