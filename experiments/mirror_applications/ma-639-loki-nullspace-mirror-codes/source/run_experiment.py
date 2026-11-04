import argparse,csv,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];N=16;TRAIN=list(range(12));HELD=list(range(12,16))
def run(seed):
 rng=np.random.default_rng(seed);d=32;w0=rng.normal(0,.1,(d,d)).astype(np.float32);K=np.eye(d)[:,:16];P=np.eye(d)-K@K.T
 x=[]
 for i in range(N):
  v=rng.normal(size=16);v/=np.linalg.norm(v);prot=np.zeros(d);prot[i]=1.;x.append(.4*prot+np.sqrt(.84)*np.r_[np.zeros(16),v])
 x=np.asarray(x);projected=[]
 for v in x:
  z=P@v;projected.append(z/(z@v))
 projected=np.asarray(projected);raw=x/(np.sum(x*x,axis=1,keepdims=True));rows=[]
 for world,rtrue in [('aligned',2),('private',16)]:
  U,_=np.linalg.qr(rng.normal(size=(d,rtrue)));coeff=rng.normal(size=(N,rtrue));targets=coeff@U.T
  _,_,vh=np.linalg.svd(targets[TRAIN],full_matrices=False);learned=vh[:min(12,d)].T
  ids=np.arange(N,dtype=np.uint8);protected_ids=np.arange(16,dtype=np.uint8);schema=np.frombuffer(b'LOKI-NS-V1',dtype=np.uint8)
  for q in [0,1,2,4,8,16]:
   basis=np.zeros((d,q));n=min(q,learned.shape[1]);
   if n:basis[:,:n]=learned[:,:n]
   codes=targets@basis if q else np.zeros((N,0));approx=codes@basis.T if q else np.zeros_like(targets)
   # payload files
   payloads={
    'unprojected_full':{'base':w0,'protected_ids':protected_ids,'edit_ids':ids,'edit_keys':x.astype(np.float32),'right_factors':raw.astype(np.float32),'update_values':targets.astype(np.float32),'schema':schema},
    'loki_projected_full':{'base':w0,'protected_ids':protected_ids,'edit_ids':ids,'edit_keys':x.astype(np.float32),'right_factors':projected.astype(np.float32),'update_values':targets.astype(np.float32),'schema':schema},
    'mirror_projected_code':{'base':w0,'protected_ids':protected_ids,'edit_ids':ids,'edit_keys':x.astype(np.float32),'right_factors':projected.astype(np.float32),'value_basis':basis.astype(np.float32),'value_codes':codes.astype(np.float32),'schema':schema},
    'native_projected_pca':{'base':w0,'protected_ids':protected_ids,'edit_ids':ids,'edit_keys':x.astype(np.float32),'right_factors':projected.astype(np.float32),'value_basis':basis.astype(np.float32),'value_codes':codes.astype(np.float32),'schema':schema}}
   sizes={};states={}
   for name,arr in payloads.items():
    path=R/'source'/f'{seed}_{world}_q{q}_{name}.npz';np.savez_compressed(path,**arr);sizes[name]=path.stat().st_size
    sp=R/'source'/f'{seed}_{world}_q{q}_{name}_state.npz';np.savez_compressed(sp,**{k:v for k,v in arr.items() if k!='base'});states[name]=sp.stat().st_size
   for method,right,val in [('unprojected_full',raw,targets),('loki_projected_full',projected,targets),('mirror_projected_code',projected,approx),('native_projected_pca',projected,approx)]:
    efficacy=[];leak=[]
    for i in HELD:
     efficacy.append(np.linalg.norm(val[i]-targets[i])/max(np.linalg.norm(targets[i]),1e-12))
     delta=np.outer(val[i],right[i]);leak.append(np.linalg.norm(delta@K)/max(np.linalg.norm(targets[i]),1e-12))
    ops=d*(d-16)+(2*d*q if 'code' in method or 'pca' in method else 0)
    rows.append({'seed':seed,'world':world,'true_private_rank':rtrue,'stored_rank':q,'method':method,'serialized_bytes':sizes[method],'edit_state_bytes':states[method],'heldout_efficacy_nrmse':float(np.mean(efficacy)),'protected_key_leakage':float(np.mean(leak)),'active_compute_proxy':int(ops),'optimizer_updates':0})
 return rows
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);args=ap.parse_args();t=time.perf_counter();rows=[]
 for s in args.seeds:rows.extend(run(s))
 wall=time.perf_counter()-t
 for r in rows:r['wall_time_s']=wall
 (R/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (R/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
  for r in rows:w.writerow([f"{r['world']}_q{r['stored_rank']}",r['seed'],r['method'],r['serialized_bytes'],12,0,r['active_compute_proxy'],r['wall_time_s'],'heldout_efficacy_nrmse',r['heldout_efficacy_nrmse'],'protected_key_leakage',r['protected_key_leakage'],'development; fresh sealed'])
 print(json.dumps(rows,indent=2))
