"""Synthetic shared equilibrium forecaster with system-role linear views."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=81901;D=8;S=8;H=16;R=2

def pack(**kw):b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def orth(rng):u,_,vt=np.linalg.svd(rng.normal(size=(D,D)));return (u@vt*.9).astype(np.float32)
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);eq=rng.normal(size=D).astype(np.float32);core=orth(rng);L=rng.normal(size=(R,D)).astype(np.float32);V=rng.normal(size=(R,D)).astype(np.float32);L/=np.linalg.norm(L,axis=1)[:,None];V/=np.linalg.norm(V,axis=1)[:,None];codes=rng.normal(0,.015,(S,R)).astype(np.float32);trans=core[None]+np.einsum('sr,ri,rj->sij',codes,L,V)
 def forecast(x,M):
  y=x.copy();out=[]
  for _ in range(H):y=eq+(y-eq)@M.T;out.append(y.copy())
  return np.stack(out,axis=1)
 x=rng.normal(size=(S,64,D)).astype(np.float32);truth=np.stack([forecast(x[s],trans[s]) for s in range(S)]);ids=np.arange(S,dtype=np.int8)
 independent=pack(equilibrium=eq,system_transitions=trans,system_ids=ids,horizon=np.array(H,np.int8));shared=pack(equilibrium=eq,shared_transition=core,system_ids=ids,horizon=np.array(H,np.int8));native=pack(equilibrium=eq,shared_transition=core,role_left=L,role_right=V,system_embeddings=codes,system_ids=ids,horizon=np.array(H,np.int8));mirror=pack(equilibrium=eq,shared_transition=core,role_left=L,role_right=V,mirror_role_codes=codes,system_ids=ids,horizon=np.array(H,np.int8));rows=[]
 for name,payload,kind in [('independent_system_forecasters',independent,'ind'),('shared_equilibrium_only',shared,'shared'),('native_system_embedding',native,'factor'),('mirror_system_role_view',mirror,'factor')]:
  z=np.load(io.BytesIO(payload))
  if kind=='ind':rec=z['system_transitions']
  elif kind=='shared':rec=np.broadcast_to(z['shared_transition'],(S,D,D))
  else:
   c=z['mirror_role_codes'] if 'mirror_role_codes' in z else z['system_embeddings'];rec=z['shared_transition'][None]+np.einsum('sr,ri,rj->sij',c,z['role_left'],z['role_right'])
  pred=np.stack([forecast(x[s],rec[s]) for s in range(S)]);err=nrmse(pred[6:],truth[6:]);allerr=nrmse(pred,truth);rho=float(max(abs(np.linalg.eigvals(m)).max() for m in rec));rt=float(np.max(np.abs(rec-(z['system_transitions'] if kind=='ind' else rec))));st=time.perf_counter();_=np.stack([forecast(x[s],rec[s]) for s in range(S)]);wall=time.perf_counter()-st
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_system_rollout_nrmse=err,all_system_rollout_nrmse=allerr,horizon=H,system_count=S,heldout_systems=2,spectral_radius_max=rho,active_macs_per_step=D*D+2*D*R,payload_roundtrip_max_abs=rt,wall_time_s=wall,status_note='native embedding factorization alias' if name=='mirror_system_role_view' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
