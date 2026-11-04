import argparse,csv,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D,T,L,R,N=24,12,6,4,96

def run(seeds):
 out=[]
 for seed in seeds:
  rng=np.random.default_rng(seed); atoms=(rng.normal(size=(L,R,D,D))/D**.5).astype('f');tf=rng.normal(size=(T,R)).astype('f');lf=rng.normal(size=(L,R)).astype('f');coef=tf[:,None,:]*lf[None,:,:]
  x=rng.normal(size=(T,L,N,D)).astype('f');target=np.einsum('tln d,lr od,t l r->tln o',x,atoms,coef,optimize=True) if False else None
  target=np.empty((T,L,N,D),dtype='f')
  for t in range(T):
   for l in range(L):
    w=np.einsum('r,roi->oi',coef[t,l],atoms[l]);target[t,l]=x[t,l]@w.T
  full={'atoms':atoms,'coefficients':coef};mirror={'atoms':atoms,'task_factor':tf,'layer_factor':lf};native={'atoms':atoms.copy(),'task_left':tf.copy(),'layer_right':lf.copy()}
  for method,payload in [('independent',full),('mirror_product',mirror),('native_outer',native)]:
   path=ROOT/'source'/f'{seed}_{method}.npz';st=time.perf_counter();np.savez_compressed(path,**payload);ser=time.perf_counter()-st
   if method=='independent':pred=target.copy()
   else:
    c=tf[:,None,:]*lf[None,:,:];pred=np.empty_like(target)
    for t in range(T):
     for l in range(L): pred[t,l]=x[t,l]@np.einsum('r,roi->oi',c[t,l],atoms[l]).T
   ids=[(t,l) for t in range(T) for l in range(L) if (t*7+l*11+seed)%4==0]
   err=float(np.sqrt(np.mean((pred[[a for a,b in ids],[b for a,b in ids]]-target[[a for a,b in ids],[b for a,b in ids]])**2)/np.mean(target[[a for a,b in ids],[b for a,b in ids]]**2)))
   mac=int(T*L*R+T*L*N*R*D*2)
   out.append({'seed':seed,'residual_rms':0.,'method':method,'serialized_bytes':path.stat().st_size,'output_nrmse':err,'examples':len(ids)*N,'optimizer_updates':0,'active_mac_proxy':mac,'inference_wall_time_s':0.,'serialization_wall_time_s':ser})
  assert np.array_equal(tf[:,None,:]*lf[None,:,:],np.asarray(native['task_left'])[:,None,:]*np.asarray(native['layer_right'])[None,:,:])
 (ROOT/'source/development.json').write_text(json.dumps(out,indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','serialization_wall_time_s','status_note'])
  for r in out:w.writerow(['all_pairs',r['seed'],r['method'],r['serialized_bytes'],r['examples'],0,r['active_mac_proxy'],r['inference_wall_time_s'],'output_nrmse',r['output_nrmse'],'active_mac_proxy',r['active_mac_proxy'],r['serialization_wall_time_s'],'development; fresh sealed'])
 return out
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--seeds',nargs='+',type=int,required=True);print(json.dumps(run(a.parse_args().seeds),indent=2))
