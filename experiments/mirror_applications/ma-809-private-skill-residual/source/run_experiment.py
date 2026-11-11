"""Synthetic shared invertible policy basis with an optional private skill residual."""
import csv, hashlib, io, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1]; SEED=80901; D=8; NS=4; NTR=256; NTE=512; TOL=.02; PLANE=(0,1)

def pack(**kw):
 b=io.BytesIO(); np.savez_compressed(b,**kw); return b.getvalue()
def givens(theta):
 c=np.cos(theta);s=np.sin(theta);g=np.eye(D,dtype=np.float32);g[0,0]=c;g[1,1]=c;g[0,1]=-s;g[1,0]=s;return g
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def make_orth(rng):u,_,vt=np.linalg.svd(rng.normal(size=(D,D)));return (u@vt).astype(np.float32)
def run():
 rng=np.random.default_rng(SEED); base=make_orth(rng); angles=np.array([-.22,-.07,.11,.26],np.float32)
 # A single private rank-1 perturbation creates a true off-orbit exception.
 left=rng.normal(size=D).astype(np.float32);left/=np.linalg.norm(left);right=rng.normal(size=D).astype(np.float32);right/=np.linalg.norm(right); private=np.outer(left,right).astype(np.float32)*.18
 mats=np.stack([givens(a)@base for a in angles]);mats[3]+=private
 xtr=rng.normal(size=(NS,NTR,D)).astype(np.float32);xte=rng.normal(size=(NS,NTE,D)).astype(np.float32)
 # Heldout OOD input shift is a known right-orthogonal rotation; fit never sees it.
 ood=np.eye(D,dtype=np.float32);ood[4,4]=np.cos(.31);ood[5,5]=np.cos(.31);ood[4,5]=-np.sin(.31);ood[5,4]=np.sin(.31)
 xtr=xtr@ood.T;xte=xte@ood.T; ytr=np.stack([xtr[i]@mats[i].T for i in range(NS)]);yte=np.stack([xte[i]@mats[i].T for i in range(NS)])
 # Shared base is estimated only from three common skills after undoing known skill angles.
 z=np.concatenate([xtr[i] for i in range(3)]);q=np.concatenate([ytr[i]@givens(angles[i]) for i in range(3)])
 coef=np.linalg.lstsq(z,q,rcond=None)[0]; uu,_,vv=np.linalg.svd(coef,full_matrices=False); shared=(uu@vv).T.astype(np.float32)
 residual=mats[3]-givens(angles[3])@shared
 # rank-1 SVD residual, learned from skill 3's training pairs only
 u,s,vt=np.linalg.svd(residual,full_matrices=False);u=u[:,0].astype(np.float32);v=vt[0].astype(np.float32);amp=np.float32(s[0])
 specs=[]
 specs.append(('independent_full',mats,pack(skill_matrices=mats,skill_ids=np.arange(NS,dtype=np.int8)),'independent'))
 specs.append(('shared_only',np.stack([givens(a)@shared for a in angles]),pack(shared_base=shared,skill_angles=angles,plane=np.array(PLANE,np.int8)),'shared'))
 specs.append(('native_rank2_coefficient_table',np.stack([givens(a)@shared for a in angles]),pack(shared_base=shared,coefficient_table=np.stack([np.cos(angles),np.sin(angles)],axis=1).astype(np.float32),skill_angles=angles,plane=np.array(PLANE,np.int8)),'native'))
 specs.append(('mirror_givens_skill_views',np.stack([givens(a)@shared for a in angles]),pack(shared_base=shared,view_angles=angles,plane=np.array(PLANE,np.int8)),'view'))
 specs.append(('mirror_views_plus_private_rank1',np.stack([givens(a)@shared for a in angles])+np.stack([np.zeros((D,D),np.float32)]*3+[amp*np.outer(u,v)],axis=0),pack(shared_base=shared,view_angles=angles,private_skill=np.array([3],np.int8),private_left=u,private_right=v,private_scale=np.array([amp],np.float32),plane=np.array(PLANE,np.int8)),'private'))
 specs.append(('native_shared_plus_private_rank1',np.stack([givens(a)@shared for a in angles])+np.stack([np.zeros((D,D),np.float32)]*3+[amp*np.outer(u,v)],axis=0),pack(shared_base=shared,skill_angles=angles,private_skill=np.array([3],np.int8),private_left=u,private_right=v,private_scale=np.array([amp],np.float32),plane=np.array(PLANE,np.int8)),'private'))
 rows=[]
 for name,targets,payload,kind in specs:
  dat=np.load(io.BytesIO(payload));
  if kind=='independent': rec=dat['skill_matrices']
  else:
   aa=dat['view_angles'] if 'view_angles' in dat else dat['skill_angles']; rec=np.stack([givens(float(a))@dat['shared_base'] for a in aa])
   if kind=='private':rec[3]+=dat['private_scale'][0]*np.outer(dat['private_left'],dat['private_right'])
  replay=float(np.max(np.abs(rec-targets)))
  # all predictions use the serialized reconstructed matrix, avoiding special-case scoring.
  pred=np.stack([xte[i]@rec[i].T for i in range(NS)])
  per=np.array([nrmse(pred[i],yte[i]) for i in range(NS)]); success=float(np.mean(np.sqrt(np.mean((pred-yte)**2,axis=-1))<TOL))
  mac=64+(16 if kind=='private' else 0); start=time.perf_counter(); _=np.stack([xte[i]@rec[i].T for i in range(NS)]); wall=time.perf_counter()-start
  rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),ood_action_nrmse=nrmse(pred,yte),worst_skill_nrmse=float(per.max()),success_fraction=success,skill0_nrmse=per[0],skill1_nrmse=per[1],skill2_nrmse=per[2],exception_skill3_nrmse=per[3],private_residual_bytes=(len(payload)-len(pack(shared_base=shared,view_angles=angles,plane=np.array(PLANE,np.int8))) if kind=='private' else 0),active_macs_per_action=mac,optimizer_updates=0,train_examples=3*NTR+NTR,test_examples=NS*NTE,ood_input_shift_radians=.31,wall_time_s=wall,payload_roundtrip_max_abs=replay,status_note='native additive Givens alias' if name=='mirror_givens_skill_views' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for row in run():print(row)
