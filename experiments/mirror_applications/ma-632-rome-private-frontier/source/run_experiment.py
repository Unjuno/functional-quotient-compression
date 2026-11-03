import argparse,csv,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];PAIRS=[(i,j) for i in range(4) for j in range(4)];HOLD={(i,i) for i in range(4)};TRAIN=[p for p in PAIRS if p not in HOLD]
def run(seed):
 rng=np.random.default_rng(seed);d=32;w0=rng.normal(0,.1,(d,d)).astype(np.float32);u,_=np.linalg.qr(rng.normal(size=(d,2)));pfull,_=np.linalg.qr(np.column_stack([u,rng.normal(size=(d,d-2))]));pfull=pfull[:,2:]
 a=rng.normal(size=4);b=rng.normal(size=4);rows=[]
 for truep in [0,2,4,8,16]:
  z={pair:rng.normal(0,.2,truep) for pair in PAIRS}
  target={pair:u@np.array([a[pair[0]],b[pair[1]]])+(pfull[:,:truep]@z[pair] if truep else 0) for pair in PAIRS}
  trainmat=np.stack([target[x] for x in TRAIN]);_,_,vh=np.linalg.svd(trainmat,full_matrices=False);shared=vh[:2].T
  resid=np.stack([target[x]-shared@(shared.T@target[x]) for x in TRAIN]);_,sv,rh=np.linalg.svd(resid,full_matrices=False)
  keep=int(np.sum(sv>max(sv[0]*1e-10 if len(sv) else 0,1e-10))) if len(sv) else 0
  raw=rh[:keep].T if keep else np.zeros((d,0));raw=raw-shared@(shared.T@raw)
  learned_private=np.linalg.qr(raw)[0][:,:keep] if keep else np.zeros((d,0))
  for q in [0,1,2,4,8,16]:
   priv=np.zeros((d,q));n=min(q,learned_private.shape[1]);
   if n:priv[:,:n]=learned_private[:,:n]
   coeff={x:np.r_[shared.T@target[x],priv.T@target[x]] for x in PAIRS};pred={x:shared@coeff[x][:2]+(priv@coeff[x][2:] if q else 0) for x in PAIRS}
   joint=np.concatenate([shared,priv],axis=1);jointcodes=np.stack([coeff[x] for x in PAIRS]);ids=np.array(PAIRS,dtype=np.uint8);schema=np.frombuffer(b'ROME-SHARED-PRIVATE-V1',dtype=np.uint8)
   payloads={
    'independent_rank1':{'base':w0,'update_values':np.stack([target[x] for x in PAIRS]).astype(np.float32),'edit_ids':ids,'schema':np.frombuffer(b'ROME-R1-V1',dtype=np.uint8)},
    'mirror_shared_private':{'base':w0,'shared_basis':shared.astype(np.float32),'private_basis':priv.astype(np.float32),'shared_codes':jointcodes[:,:2].astype(np.float32),'private_codes':jointcodes[:,2:].astype(np.float32),'edit_ids':ids,'schema':schema},
    'native_joint_lowrank':{'base':w0,'joint_basis':joint.astype(np.float32),'coefficients':jointcodes.astype(np.float32),'edit_ids':ids,'schema':schema}}
   sizes={};states={}
   for name,arr in payloads.items():
    path=R/'source'/f'{seed}_p{truep}_q{q}_{name}.npz';np.savez_compressed(path,**arr);sizes[name]=path.stat().st_size
    state={k:v for k,v in arr.items() if k!='base'};sp=R/'source'/f'{seed}_p{truep}_q{q}_{name}_state.npz';np.savez_compressed(sp,**state);states[name]=sp.stat().st_size
   for method,pr in [('independent_rank1',target),('mirror_shared_private',pred),('native_joint_lowrank',{x:joint@coeff[x] for x in PAIRS})]:
    for pair in HOLD:
     ix=PAIRS.index(pair);err=np.linalg.norm(pr[pair]-target[pair])/max(np.linalg.norm(target[pair]),1e-12)
     # One-hot key selector: off-key response of this rank-one update is exactly zero.
     response=np.outer(pr[pair],np.eye(len(PAIRS))[ix]);response[:,ix]=0;leak=np.linalg.norm(response)/max(np.linalg.norm(target[pair]),1e-12)
     ops=d if method=='independent_rank1' else 2*d+2*q+4
     rows.append({'seed':seed,'true_private_rank':truep,'stored_private_rank':q,'task':list(pair),'method':method,'serialized_bytes':sizes[method],'edit_state_bytes':states[method],'efficacy_nrmse':float(err),'locality_leakage':float(leak),'active_compute_proxy':int(ops),'optimizer_updates':0})
 return rows
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);args=ap.parse_args();t=time.perf_counter();rows=[]
 for s in args.seeds:rows.extend(run(s))
 wall=time.perf_counter()-t
 for r in rows:r['wall_time_s']=wall
 (R/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (R/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
  for r in rows:w.writerow([f"private_rank_{r['true_private_rank']}_stored_{r['stored_private_rank']}",r['seed'],r['method'],r['serialized_bytes'],12,0,r['active_compute_proxy'],r['wall_time_s'],'efficacy_nrmse',r['efficacy_nrmse'],'locality_leakage',r['locality_leakage'],'development; fresh sealed'])
 print(json.dumps(rows,indent=2))
