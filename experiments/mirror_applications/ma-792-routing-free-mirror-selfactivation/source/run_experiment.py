"""Synthetic routing-free self-activation and shared function Views."""
import csv,hashlib,io
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=79201;E=4;D=8;N=256;R=1

def pack(**x):
 b=io.BytesIO();np.savez_compressed(b,**x);return b.getvalue()
def nrmse(p,y):return float(np.sqrt(np.mean((p-y)**2)/(np.mean(y**2)+1e-12)))
def run():
 rng=np.random.default_rng(SEED);codes=np.array([-1.2,-.4,.4,1.2],np.float32);centers=np.zeros((E,D),np.float32);centers[:,0]=codes*2.0;centers[:,1]=codes*.6;W0=rng.normal(0,.18,(D,D)).astype(np.float32);basis=rng.normal(0,.05,(D,D)).astype(np.float32);basis/=np.linalg.norm(basis);rows=[]
 for sigma in (0.,.05):
  noise=rng.normal(0,sigma,(E,D,D)).astype(np.float32);teacher=W0[None]+codes[:,None,None]*basis[None]+noise
  x=np.concatenate([centers[e]+rng.normal(0,.3,(N,D)).astype(np.float32) for e in range(E)]);labels=np.repeat(np.arange(E),N)
  target=np.einsum('ni,nji->nj',x,teacher[labels]);scores=-((x[:,None]-centers[None])**2).sum(-1);route=np.argmax(scores,axis=1);accuracy=float(np.mean(route==labels));loads=np.bincount(route,minlength=E)/len(route)
  hard=np.broadcast_to(teacher.mean(0),(E,D,D)).copy()
  mirror=W0[None]+codes[:,None,None]*basis[None]
  basecent=(centers[0]+centers[3])/2;center_dir=centers[3]-centers[0];cc=(codes-codes.mean())/(codes[3]-codes[0]);cent_view=basecent[None]+cc[:,None]*center_dir[None]
  # The View code modulates both the expert transform and its self-activation centroid.
  methods=[('routing_free_independent',teacher,centers,pack(experts=teacher,activation_centers=centers,balance_bias=np.zeros(E,np.float32),expert_ids=np.arange(E,dtype=np.int16)),0),
   ('routing_free_hard_tied',hard,centers,pack(expert=teacher.mean(0),activation_centers=centers,balance_bias=np.zeros(E,np.float32),expert_ids=np.arange(E,dtype=np.int16)),0),
   ('central_top1_independent',teacher,centers,pack(experts=teacher,router_centers=centers,expert_ids=np.arange(E,dtype=np.int16)),0),
   ('mirror_selfactivation_views',mirror,cent_view,pack(parent=W0,weight_basis=basis,view_codes=codes,centroid_base=basecent,centroid_direction=center_dir,balance_bias=np.zeros(E,np.float32),expert_ids=np.arange(E,dtype=np.int16)),D*D*E),
   ('native_lowrank_plus_selfactivation',mirror,cent_view,pack(parent=W0,weight_basis=basis,view_codes=codes,centroid_base=basecent,centroid_direction=center_dir,balance_bias=np.zeros(E,np.float32),expert_ids=np.arange(E,dtype=np.int16)),D*D*E)]
  for name,ws,cens,payload,decode in methods:
   sc=-((x[:,None]-cens[None])**2).sum(-1);rr=np.argmax(sc,axis=1);ll=np.bincount(rr,minlength=E)/len(rr);acc=float(np.mean(rr==labels)
   )
   pred=np.einsum('ni,nji->nj',x,ws[rr]);rows.append({'seed':SEED,'heterogeneity_sigma':sigma,'method':name,'serialized_bytes':len(payload),'payload_sha256':hashlib.sha256(payload).hexdigest(),'mode_routing_accuracy':acc,'expert_load_min':float(ll.min()),'expert_load_max':float(ll.max()),'output_nrmse':nrmse(pred,target),'active_macs_per_token':E*D+D*D,'reconstruction_macs_total':decode,'optimizer_updates':0,'status_note':'exact native low-rank alias' if name=='mirror_selfactivation_views' else ''})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 for r in run():print(r)
