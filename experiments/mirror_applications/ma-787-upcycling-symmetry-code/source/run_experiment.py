"""Synthetic child-task symmetry breaking: shared code vs random perturbation and full updates."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=78701;NCH=8;D=10;NCAL=32;NTEST=256;UPDATES=400

def pack(**a):
 b=io.BytesIO();np.savez_compressed(b,**a);return b.getvalue()
def nrmse(p,y):return float(np.sqrt(np.mean((p-y)**2)/(np.mean(y**2)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);W0=rng.normal(0,.2,(D,D)).astype(np.float32);V=rng.normal(0,.08,(D,D)).astype(np.float32);V/=np.linalg.norm(V)
 random_dir=rng.normal(size=(D,D)).astype(np.float32);random_dir/=np.linalg.norm(random_dir)
 rows=[]
 for sigma in (0.,.05):
  c=rng.normal(0,.6,NCH).astype(np.float32);priv=rng.normal(0,sigma,(NCH,D,D)).astype(np.float32);teacher=W0[None]+c[:,None,None]*V[None]+priv
  xcal=rng.normal(size=(NCH,NCAL,D)).astype(np.float32);ycal=np.einsum('nbi,nji->nbj',xcal,teacher)
  xtest=rng.normal(size=(NCH,NTEST,D)).astype(np.float32);ytest=np.einsum('nbi,nji->nbj',xtest,teacher)
  # Learn shared rank-1 basis from train child IDs only.
  delta=teacher[:6]-W0;_,_,vh=np.linalg.svd(delta.reshape(6,-1),full_matrices=False);basis=vh[:1].reshape(1,D,D).astype(np.float32)
  codes=np.zeros(NCH,np.float32)
  for n in range(6):codes[n]=np.sum((teacher[n]-W0)*basis[0])/(np.sum(basis[0]**2)+1e-12)
  for n in (6,7):
   feat=np.einsum('bi,ji->bj',xcal[n],basis[0]);res=ycal[n]-xcal[n]@W0.T;codes[n]=np.sum(feat*res)/(np.sum(feat**2)+1e-12)
  randcodes=np.zeros(NCH,np.float32)
  for n in range(NCH):
   feat=xcal[n]@random_dir.T;res=ycal[n]-xcal[n]@W0.T;randcodes[n]=np.sum(feat*res)/(np.sum(feat**2)+1e-12)
  mirror=W0[None]+codes[:,None,None]*basis[0]
  random_pred=W0[None]+randcodes[:,None,None]*random_dir
  # Full continued-pretraining baseline: SGD on every child from duplicated W0.
  cpt=np.broadcast_to(W0,(NCH,D,D)).copy();cpt_updates=0;cpt_start=time.perf_counter()
  for n in range(NCH):
   w=cpt[n]
   for step in range(UPDATES):
    ids=rng.integers(0,NCAL,size=32);xb=xcal[n,ids];yb=ycal[n,ids];err=xb@w.T-yb;grad=err.T@xb/len(ids);w-=.04*grad;cpt_updates+=1
   cpt[n]=w
  cpt_wall=(time.perf_counter()-cpt_start)*1000
  methods=[('duplicated_identical',np.broadcast_to(W0,(NCH,D,D)).copy(),pack(parent=W0,child_ids=np.arange(NCH,dtype=np.int16)),0,D*D),
    ('random_perturbation_code',random_pred,pack(parent=W0,random_direction=random_dir,child_codes=randcodes,child_ids=np.arange(NCH,dtype=np.int16)),0,2*D*D),
    ('mirror_shared_basis_code',mirror,pack(parent=W0,basis=basis,child_codes=codes,child_ids=np.arange(NCH,dtype=np.int16)),0,2*D*D),
    ('native_rank1_lora',mirror,pack(parent=W0,basis=basis,child_codes=codes,child_ids=np.arange(NCH,dtype=np.int16)),0,2*D*D),
    ('continued_pretraining_full_children',cpt,pack(child_weights=cpt,child_ids=np.arange(NCH,dtype=np.int16)),cpt_updates,D*D),
    ('independent_full_children',teacher,pack(child_weights=teacher,child_ids=np.arange(NCH,dtype=np.int16)),0,D*D)]
  for name,weights,payload,updates,mac in methods:
   pred=np.einsum('nbi,nji->nbj',xtest,weights);rows.append({'seed':SEED,'heterogeneity_sigma':sigma,'method':name,'serialized_bytes':len(payload),'payload_sha256':hashlib.sha256(payload).hexdigest(),'heldout_child_nrmse':nrmse(pred[6:],ytest[6:]),'all_child_nrmse':nrmse(pred,ytest),'calibration_examples_per_child':NCAL,'optimizer_updates':updates,'active_macs_per_token':mac,'training_wall_ms':round(cpt_wall,6) if name=='continued_pretraining_full_children' else 0,'status_note':'exact native LoRA alias' if name=='mirror_shared_basis_code' else ''})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
