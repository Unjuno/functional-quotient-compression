import argparse,csv,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D,T,L,R,N=24,12,6,4,96

def predict_pair(x,u,v,coef):
 z=x@v.T
 return np.einsum('nr,ro->no',z*coef[None,:],u)
def run(seeds):
 out=[]
 for seed in seeds:
  rng=np.random.default_rng(seed);u=(rng.normal(size=(L,R,D))/D**.5).astype('f');v=(rng.normal(size=(L,R,D))/D**.5).astype('f');tf=rng.normal(size=(T,R)).astype('f');lf=rng.normal(size=(L,R)).astype('f');coef=tf[:,None,:]*lf[None,:,:]
  x=rng.normal(size=(T,L,N,D)).astype('f');target=np.empty((T,L,N,D),'f')
  for t in range(T):
   for l in range(L):target[t,l]=predict_pair(x[t,l],u[l],v[l],coef[t,l])
  payloads={'independent':{'atom_left':u,'atom_right':v,'coefficients':coef},'mirror_product':{'atom_left':u,'atom_right':v,'task_factor':tf,'layer_factor':lf},'native_outer':{'atom_left':u.copy(),'atom_right':v.copy(),'task_left':tf.copy(),'layer_right':lf.copy()}}
  ids=[(t,l) for t in range(T) for l in range(L) if (t*7+l*11+seed)%4==0]
  for method,payload in payloads.items():
   path=ROOT/'source'/f'{seed}_{method}.npz';st=time.perf_counter();np.savez_compressed(path,**payload);ser=time.perf_counter()-st
   pred=[];begin=time.perf_counter()
   for t,l in ids:
    c=coef[t,l] if method=='independent' else tf[t]*lf[l]
    pred.append(predict_pair(x[t,l],u[l],v[l],c))
   wall=time.perf_counter()-begin;pred=np.stack(pred);truth=np.stack([target[t,l] for t,l in ids])
   err=float(np.sqrt(np.mean((pred-truth)**2)/max(float(np.mean(truth**2)),1e-12)))
   mac=int(len(ids)*(N*R*D*2+(R if method!='independent' else 0)))
   out.append({'seed':seed,'residual_rms':0.,'method':method,'serialized_bytes':path.stat().st_size,'output_nrmse':err,'examples':len(ids)*N,'optimizer_updates':0,'active_mac_proxy':mac,'inference_wall_time_s':wall,'serialization_wall_time_s':ser})
  assert np.array_equal(tf[:,None,:]*lf[None,:,:],payloads['native_outer']['task_left'][:,None,:]*payloads['native_outer']['layer_right'][None,:,:])
 (ROOT/'source/development.json').write_text(json.dumps(out,indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','serialization_wall_time_s','status_note'])
  for r in out:w.writerow(['heldout_task_layer_pairs',r['seed'],r['method'],r['serialized_bytes'],r['examples'],0,r['active_mac_proxy'],r['inference_wall_time_s'],'output_nrmse',r['output_nrmse'],'active_mac_proxy',r['active_mac_proxy'],r['serialization_wall_time_s'],'corrected rank-1 LoRA factors; development; fresh sealed'])
 return out
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--seeds',nargs='+',type=int,required=True);print(json.dumps(run(a.parse_args().seeds),indent=2))
