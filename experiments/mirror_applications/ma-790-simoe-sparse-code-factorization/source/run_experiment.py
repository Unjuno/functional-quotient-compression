"""Sparse SIMoE interpolation coefficients vs PCA-factorized task codes."""
import csv,hashlib,io
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=79001;T=128;A=16;D=8;K=2

def pack(**x):
 b=io.BytesIO();np.savez_compressed(b,**x);return b.getvalue()
def nrmse(p,t):return float(np.sqrt(np.mean((p-t)**2)/(np.mean(t**2)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);anchors=rng.normal(0,.2,(A,D,D)).astype(np.float32);C=np.zeros((T,A),np.float32)
 for t in range(T):
  ids=rng.choice(A,K,replace=False);C[t,ids]=rng.normal(0,.5,K)
 weights=np.einsum('ta,aij->tij',C,anchors);x=rng.normal(size=(T,64,D)).astype(np.float32);target=np.einsum('tni,tji->tnj',x,weights)
 train=np.arange(T)<96;held=~train
 # Sparse SIMoE control stores all supports and values.
 ids=np.argsort(np.abs(C),axis=1)[:,-K:].astype(np.uint8);vals=np.take_along_axis(C,ids,axis=1).astype(np.float32)
 raw=pack(anchors=anchors,sparse_ids=ids,sparse_values=vals,expert_ids=np.arange(T,dtype=np.int16))
 rows=[{'seed':SEED,'method':'simoe_raw_sparse_coefficients','rank':K,'serialized_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'heldout_output_nrmse':0.,'heldout_coefficient_nrmse':0.,'coefficient_decode_macs_per_expert':K*D*D,'optimizer_updates':0,'status_note':'raw top-2 supports and values'}]
 _,_,vh=np.linalg.svd(C[train],full_matrices=False)
 for r in (1,2,4,8,16):
  basis=vh[:r].astype(np.float32);codes=(C@basis.T).astype(np.float32);chat=codes@basis;w_hat=np.einsum('ta,aij->tij',chat,anchors);out=np.einsum('tni,tji->tnj',x,w_hat)
  p=pack(anchors=anchors,basis=basis,task_codes=codes,expert_ids=np.arange(T,dtype=np.int16))
  rows.append({'seed':SEED,'method':'mirror_factorized_code','rank':r,'serialized_bytes':len(p),'payload_sha256':hashlib.sha256(p).hexdigest(),'heldout_output_nrmse':nrmse(out[held],target[held]),'heldout_coefficient_nrmse':nrmse(chat[held],C[held]),'coefficient_decode_macs_per_expert':r*A+A*D*D,'optimizer_updates':0,'status_note':'factorized coefficient view'})
  q=pack(anchors=anchors,basis=basis,task_codes=codes,expert_ids=np.arange(T,dtype=np.int16))
  rows.append({'seed':SEED,'method':'native_pca_coefficient_code','rank':r,'serialized_bytes':len(q),'payload_sha256':hashlib.sha256(q).hexdigest(),'heldout_output_nrmse':nrmse(out[held],target[held]),'heldout_coefficient_nrmse':nrmse(chat[held],C[held]),'coefficient_decode_macs_per_expert':r*A+A*D*D,'optimizer_updates':0,'status_note':'exact native PCA alias'})
 full=pack(weights=weights,expert_ids=np.arange(T,dtype=np.int16));rows.append({'seed':SEED,'method':'independent_full_experts','rank':A,'serialized_bytes':len(full),'payload_sha256':hashlib.sha256(full).hexdigest(),'heldout_output_nrmse':0.,'heldout_coefficient_nrmse':0.,'coefficient_decode_macs_per_expert':D*D,'optimizer_updates':0,'status_note':'independent upper control'})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
