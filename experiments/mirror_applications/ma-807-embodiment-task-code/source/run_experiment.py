"""Synthetic embodiment x task factorization for one-step invertible policies."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[1];SEED=80701;D=8;NE=3;NT=3;NTR=256;NTE=512;TOL=.02
PLANES=((0,1),(2,3));PAIRS=[(e,t) for e in range(NE) for t in range(NT)];HELD=np.array([i for i,(e,t) in enumerate(PAIRS) if (e+t)%2==0]);TRAIN=np.array([i for i in range(9) if i not in HELD])

def pack(**x):
 b=io.BytesIO();np.savez_compressed(b,**x);return b.getvalue()
def givens(theta,plane):
 i,j=plane;c=np.cos(theta);s=np.sin(theta);g=np.eye(D,dtype=np.float32);g[i,i]=c;g[j,j]=c;g[i,j]=-s;g[j,i]=s;return g
def rotate(x,theta):
 y=x.copy()
 for r,p in enumerate(PLANES):y=y@givens(float(theta[r]),p).T
 return y
def policy(base,theta,x):return rotate(x@base.T,theta)
def polar(a):u,_,vt=np.linalg.svd(a,full_matrices=False);return (u@vt).astype(np.float32)
def nrmse(p,y):return float(np.sqrt(np.mean((p-y)**2)/(np.mean(y*y)+1e-12)))

def run():
 rng=np.random.default_rng(SEED);u,_,vt=np.linalg.svd(rng.normal(size=(D,D)));base=(u@vt).astype(np.float32)
 emb=rng.uniform(-.25,.25,(NE,2)).astype(np.float32);task=rng.uniform(-.25,.25,(NT,2)).astype(np.float32)
 xtr=rng.normal(size=(9,NTR,D)).astype(np.float32);xte=rng.normal(size=(9,NTE,D)).astype(np.float32)
 teacher=[];ytr=[];yte=[]
 for e,t in PAIRS:
  theta=emb[e]+task[t];teacher.append(givens(float(theta[1]),PLANES[1])@givens(float(theta[0]),PLANES[0])@base)
  ytr.append(policy(base,theta,xtr[len(teacher)-1]));yte.append(policy(base,theta,xte[len(teacher)-1]))
 teacher=np.stack(teacher);ytr=np.stack(ytr);yte=np.stack(yte)
 # Fit shared base using only observed pairs after undoing known embodiment/task coordinates.
 xc=[];yc=[]
 for idx in TRAIN:
  e,t=PAIRS[idx];yc.append(rotate(ytr[idx],-(emb[e]+task[t])));xc.append(xtr[idx])
 shared=polar(np.linalg.lstsq(np.concatenate(xc),np.concatenate(yc),rcond=None)[0].T)
 # Marginal controls get only one identity factor and fit a separate body- or task-level base.
 emb_bases=[];task_bases=[]
 for e in range(NE):
  ids=[i for i in TRAIN if PAIRS[i][0]==e]
  emb_bases.append(polar(np.linalg.lstsq(np.concatenate([xtr[i] for i in ids]),np.concatenate([ytr[i] for i in ids]),rcond=None)[0].T))
 for t in range(NT):
  ids=[i for i in TRAIN if PAIRS[i][1]==t]
  task_bases.append(polar(np.linalg.lstsq(np.concatenate([xtr[i] for i in ids]),np.concatenate([ytr[i] for i in ids]),rcond=None)[0].T))
 emb_bases=np.stack(emb_bases);task_bases=np.stack(task_bases)
 indep_payload=pack(pair_matrices=teacher,pair_ids=np.asarray(PAIRS,np.int8),dim=np.array(D,np.int8))
 view=pack(shared_base=shared,embodiment_angles=emb,task_angles=task,rotation_planes=np.asarray(PLANES,np.int8),pair_ids=np.asarray(PAIRS,np.int8),dim=np.array(D,np.int8))
 view_maps=np.stack([givens(float((emb[e]+task[t])[1]),PLANES[1])@givens(float((emb[e]+task[t])[0]),PLANES[0])@shared for e,t in PAIRS])
 specs=[('independent_pair_policies',teacher,indep_payload,'independent'),
  ('embodiment_only',emb_bases,pack(embodiment_matrices=emb_bases,embodiment_angles=emb,rotation_planes=np.asarray(PLANES,np.int8)), 'embodiment'),
  ('task_only',task_bases,pack(task_matrices=task_bases,task_angles=task,rotation_planes=np.asarray(PLANES,np.int8)),'task'),
  ('shared_policy_no_factor_code',np.broadcast_to(shared,(9,D,D)),pack(shared_base=shared,dim=np.array(D,np.int8)),'shared'),
  ('mirror_embodiment_task_views',view_maps,view,'mirror'),
  ('native_additive_givens_composition',view_maps,view,'mirror')]
 rows=[]
 for name,obj,payload,kind in specs:
  saved=np.load(io.BytesIO(payload));pred=[]
  if kind=='independent':restored=saved['pair_matrices'];pred=np.stack([xte[i]@restored[i].T for i in range(9)])
  elif kind=='embodiment':
   restored=saved['embodiment_matrices'];pred=np.stack([rotate(xte[i]@restored[PAIRS[i][0]].T,saved['embodiment_angles'][PAIRS[i][0]]) for i in range(9)])
  elif kind=='task':
   restored=saved['task_matrices'];pred=np.stack([rotate(xte[i]@restored[PAIRS[i][1]].T,saved['task_angles'][PAIRS[i][1]]) for i in range(9)])
  elif kind=='shared':
   restored=np.broadcast_to(saved['shared_base'],(9,D,D));pred=np.stack([xte[i]@restored[i].T for i in range(9)])
  else:
   angles=np.stack([saved['embodiment_angles'][e]+saved['task_angles'][t] for e,t in PAIRS])
   restored=np.stack([givens(float(a[1]),tuple(saved['rotation_planes'][1]))@givens(float(a[0]),tuple(saved['rotation_planes'][0]))@saved['shared_base'] for a in angles])
   pred=np.stack([rotate(xte[i]@saved['shared_base'].T,angles[i]) for i in range(9)])
  rt=float(np.max(np.abs(restored-obj)))
  if rt>1e-5:raise AssertionError(f'{name} payload reconstruction {rt}')
  success=float(np.mean(np.sqrt(np.mean((pred-yte)**2,axis=-1))<TOL));orth=float(max(np.linalg.norm(m@m.T-np.eye(D)) for m in restored))
  start=time.perf_counter()
  if kind=='independent':pred=np.stack([xte[i]@restored[i].T for i in range(9)])
  elif kind=='embodiment':pred=np.stack([rotate(xte[i]@restored[PAIRS[i][0]].T,saved['embodiment_angles'][PAIRS[i][0]]) for i in range(9)])
  elif kind=='task':pred=np.stack([rotate(xte[i]@restored[PAIRS[i][1]].T,saved['task_angles'][PAIRS[i][1]]) for i in range(9)])
  elif kind=='shared':pred=np.stack([xte[i]@restored[i].T for i in range(9)])
  else:pred=np.stack([rotate(xte[i]@saved['shared_base'].T,saved['embodiment_angles'][e]+saved['task_angles'][t]) for i,(e,t) in enumerate(PAIRS)])
  wall=time.perf_counter()-start
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),
   heldout_action_nrmse=nrmse(pred[HELD],yte[HELD]),all_pair_action_nrmse=nrmse(pred,yte),task_success_fraction=success,
   max_orthogonality_error=orth,active_macs_per_action=76 if kind in ('embodiment','task','mirror') else 64,angle_add_ops_per_action=2 if kind=='mirror' else 0,
   heldout_pair_count=len(HELD),train_examples=len(TRAIN)*NTR,test_examples=9*NTE,optimizer_updates=0,wall_time_s=wall,
   payload_roundtrip_max_abs=rt,status_note='exact native additive Givens alias' if name=='mirror_embodiment_task_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows

if __name__=='__main__':
 for r in run():print(r)
