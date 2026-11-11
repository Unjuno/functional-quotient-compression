"""Few-shot code-only skill extension on a frozen synthetic motor basis."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=81301;D=8;R=2;NB=6;NS=32;NT=512

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);core=rng.normal(size=(D,D)).astype(np.float32)*.12;L=rng.normal(size=(R,D)).astype(np.float32);V=rng.normal(size=(R,D)).astype(np.float32);L/=np.linalg.norm(L,axis=1)[:,None];V/=np.linalg.norm(V,axis=1)[:,None]
 basecodes=rng.normal(0,.04,(NB,R)).astype(np.float32);newcode=rng.normal(0,.05,R).astype(np.float32);W=core+np.einsum('r,ri,rj->ij',newcode,L,V)
 xs=rng.normal(size=(NS,D)).astype(np.float32);ys=xs@W.T;xt=rng.normal(size=(NT,D)).astype(np.float32);yt=xt@W.T
 # Infer coefficient using only support: solve linear least squares on the frozen atom responses.
 feats=np.stack([xs@np.outer(L[k],V[k]).T for k in range(R)],axis=-1).reshape(-1,R);target=(ys-xs@core.T).reshape(-1);coef=np.linalg.lstsq(feats,target,rcond=None)[0].astype(np.float32)
 rec=core+np.einsum('r,ri,rj->ij',coef,L,V)
 # Fresh LoRA control rank2 via direct full matrix fit and SVD factorization.
 lr=np.linalg.lstsq(xs,ys-xs@core.T,rcond=None)[0].T;u,s,vt=np.linalg.svd(lr,full_matrices=False);A=(u[:,:R]*np.sqrt(s[:R])).astype(np.float32);B=(np.sqrt(s[:R,None])*vt[:R]).astype(np.float32);lora=core+A@B
 full=core+lr
 payloads=[('frozen_core_only',pack(core=core,base_codes=basecodes),core),('mirror_new_skill_code',pack(core=core,left_atoms=L,right_atoms=V,base_codes=basecodes,new_code=coef),rec),('native_shared_basis_code',pack(core=core,left_atoms=L,right_atoms=V,base_codes=basecodes,coefficient_code=coef),rec),('fresh_rank2_lora',pack(core=core,lora_A=A,lora_B=B),lora),('independent_full_new_skill',pack(new_skill_matrix=W,base_skill_codes=basecodes),W)]
 rows=[]
 for name,payload,M in payloads:
  p=xt@M.T;err=nrmse(p,yt);rt=float(np.max(np.abs(M-(core if name=='frozen_core_only' else M))))
  start=time.perf_counter();_=xt@M.T;wall=time.perf_counter()-start
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_new_skill_nrmse=err,support_examples=NS,adaptation_ops=NS*D*R,optimizer_updates=0,active_macs_per_action=D*D+2*D*R,payload_roundtrip_max_abs=rt,wall_time_s=wall,status_note='native shared-basis coefficient alias' if name=='mirror_new_skill_code' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
