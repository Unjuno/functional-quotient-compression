"""Synthetic one-step invertible action maps with per-skill Givens coordinates."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[1];SEED=80601;D=8;SKILLS=9;NTR=256;NTE=512;TOL=.02
PLANES=((0,1),(2,3))

def pack(**x):
 b=io.BytesIO();np.savez_compressed(b,**x);return b.getvalue()

def givens(theta,plane=(0,1),dim=D):
 i,j=plane;c=np.cos(theta);s=np.sin(theta);g=np.eye(dim,dtype=np.float32);g[i,i]=c;g[j,j]=c;g[i,j]=-s;g[j,i]=s;return g

def apply_givens(x,theta):
 y=x.copy()
 for r,plane in enumerate(PLANES):y=y@givens(float(theta[r]),plane).T
 return y

def polar(a):
 u,_,vt=np.linalg.svd(a,full_matrices=False);return (u@vt).astype(np.float32)

def nrmse(p,y):return float(np.sqrt(np.mean((p-y)**2)/(np.mean(y*y)+1e-12)))

def run():
 rng=np.random.default_rng(SEED);u,_,vt=np.linalg.svd(rng.normal(size=(D,D)));base=(u@vt).astype(np.float32)
 angles=rng.uniform(-.32,.32,(SKILLS,2)).astype(np.float32)
 xtr=rng.normal(size=(SKILLS,NTR,D)).astype(np.float32);xte=rng.normal(size=(SKILLS,NTE,D)).astype(np.float32)
 ytr=np.stack([apply_givens(xtr[k]@base.T,angles[k]) for k in range(SKILLS)])
 yte=np.stack([apply_givens(xte[k]@base.T,angles[k]) for k in range(SKILLS)])
 # Independent adapters use per-skill least-squares maps; shared code learns one base after undoing known skill rotations.
 independent=[];xcommon=[];ycommon=[]
 for k in range(SKILLS):
  w=np.linalg.lstsq(xtr[k],ytr[k],rcond=None)[0].T
  independent.append(polar(w))
  ycommon.append(apply_givens(ytr[k],-angles[k]))
  xcommon.append(xtr[k])
 independent=np.stack(independent);xcommon=np.concatenate(xcommon);ycommon=np.concatenate(ycommon)
 shared_base=polar(np.linalg.lstsq(xcommon,ycommon,rcond=None)[0].T)
 pooled_base=polar(np.linalg.lstsq(xtr.reshape(-1,D),ytr.reshape(-1,D),rcond=None)[0].T)
 mirror=np.stack([apply_givens(xte[k]@shared_base.T,angles[k]) for k in range(SKILLS)])
 shared_pred=np.stack([xte[k]@pooled_base.T for k in range(SKILLS)])
 independent_pred=np.stack([xte[k]@independent[k].T for k in range(SKILLS)])
 params=pack(shared_base=shared_base,skill_angles=angles,rotation_planes=np.asarray(PLANES,np.int8),skill_ids=np.arange(SKILLS,dtype=np.int8),dim=np.array(D,np.int8))
 indep_payload=pack(skill_matrices=independent,skill_ids=np.arange(SKILLS,dtype=np.int8),dim=np.array(D,np.int8))
 shared_payload=pack(policy_matrix=pooled_base,skill_ids=np.arange(SKILLS,dtype=np.int8),dim=np.array(D,np.int8))
 specs=[('independent_invertible_policies',independent_pred,indep_payload,'independent'),('shared_policy_no_skill_code',shared_pred,shared_payload,'shared'),
  ('mirror_one_step_skill_views',mirror,params,'mirror'),('native_givens_one_step_adapter',mirror,params,'mirror')]
 rows=[]
 for name,pred,payload,kind in specs:
  saved=np.load(io.BytesIO(payload))
  if kind=='independent':restored=saved['skill_matrices']
  elif kind=='shared':restored=np.broadcast_to(saved['policy_matrix'],(SKILLS,D,D))
  else:restored=np.stack([givens(float(saved['skill_angles'][k,1]),tuple(saved['rotation_planes'][1]),D)@givens(float(saved['skill_angles'][k,0]),tuple(saved['rotation_planes'][0]),D)@saved['shared_base'] for k in range(SKILLS)])
  if kind=='mirror': target_maps=np.stack([givens(float(angles[k,1]),PLANES[1])@givens(float(angles[k,0]),PLANES[0])@shared_base for k in range(SKILLS)])
  elif kind=='independent':target_maps=independent
  else:target_maps=np.broadcast_to(pooled_base,(SKILLS,D,D))
  rt=float(np.max(np.abs(restored-target_maps)))
  if rt>1e-5:raise AssertionError(f'{name} payload does not reconstruct evaluated policy: {rt}')
  def eval_policy(skill,x):
   if kind in ('independent','shared'):return x@restored[skill].T
   return apply_givens(x@saved['shared_base'].T,saved['skill_angles'][skill])
  start=time.perf_counter();pred_eval=np.stack([eval_policy(k,xte[k]) for k in range(SKILLS)]);wall=time.perf_counter()-start
  success=float(np.mean(np.sqrt(np.mean((pred_eval-yte)**2,axis=-1))<TOL))
  orth=float(max(np.linalg.norm(m@m.T-np.eye(D)) for m in restored))
  one_step=nrmse(pred_eval,yte)
  # A straight-line conditional flow sampler reaches the same action in 1 or 4 Euler subdivisions.
  noise=rng.normal(size=(SKILLS,NTE,D)).astype(np.float32);iterative=noise.copy();it_start=time.perf_counter()
  for _ in range(4):
   flow_targets=np.stack([eval_policy(k,xte[k]) for k in range(SKILLS)])
   velocity=flow_targets-noise;iterative+=velocity/4
  iterative_wall=time.perf_counter()-it_start
  iter_error=nrmse(iterative,yte)
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),
   one_step_action_nrmse=one_step,task_success_fraction=success,iterative4_action_nrmse=iter_error,
   max_orthogonality_error=orth,active_macs_per_action=(D*D+12 if kind=='mirror' else D*D),
   flow_evaluations_per_action=1,iterative_flow_evaluations=4,train_examples=SKILLS*NTR,test_examples=SKILLS*NTE,
   optimizer_updates=0,wall_time_s=wall,iterative4_wall_time_s=iterative_wall,payload_roundtrip_max_abs=rt,status_note='exact native Givens alias' if name=='mirror_one_step_skill_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows

if __name__=='__main__':
 for r in run():print(r)
