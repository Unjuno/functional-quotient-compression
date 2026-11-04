import argparse,csv,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D,E,R,N,P=32,16,4,512,2
def softmax(x):
 z=x-x.max(-1,keepdims=True);v=np.exp(z);return v/v.sum(-1,keepdims=True)
def world(seed,rho):
 q=np.random.default_rng(seed+int(rho*10000));base=(q.normal(size=(D,D))/np.sqrt(D)*.2).astype('f');u=(q.normal(size=(R,D))/np.sqrt(D)).astype('f');v=(q.normal(size=(R,D))/np.sqrt(D)).astype('f');c=q.normal(size=(E,R)).astype('f')*.3
 shared=np.einsum('er,ro,ri->eoi',c,u,v);res=q.normal(size=(E,D,D)).astype('f');res*=rho*np.sqrt(np.mean(shared**2))/np.sqrt(np.mean(res**2)) if rho else 0;experts=shared+res;router=(q.normal(size=(D,E))/np.sqrt(D)).astype('f');x=q.normal(size=(N,D)).astype('f')
 # rank-2 SVD factors for each private expert residual
 pl=np.zeros((E,P,D),'f');pr=np.zeros((E,P,D),'f')
 for e in range(E):
  a,s,b=np.linalg.svd(res[e],full_matrices=False);pl[e]=a[:,:P].T*s[:P,None];pr[e]=b[:P]
 return base,u,v,c,res,experts,pl,pr,router,x
def pred_full(x,base,experts,router,top2=False):
 g=softmax(x@router);out=np.zeros_like(x)
 if top2:
  ids=np.argpartition(g,-2,axis=1)[:,-2:];m=np.zeros_like(g);np.put_along_axis(m,ids,1.,axis=1);g*=m;g/=g.sum(1,keepdims=True)
  for n in range(len(x)):
   for e in ids[n]:out[n]+=g[n,e]*(x[n]@experts[e].T)
 else:out=np.einsum('ne,neo->no',g,np.einsum('nd,eod->neo',x,experts))
 return x@base.T+out
def pred_code(x,base,u,v,c,router,pl=None,pr=None):
 g=softmax(x@router);mix=g@c;z=x@v.T;out=np.einsum('nr,ro->no',mix*z,u)
 if pl is not None:
  zz=np.einsum('nd,epd->nep',x,pr);rr=np.einsum('nep,epo->neo',zz,pl);out+=np.einsum('ne,neo->no',g,rr)
 return x@base.T+out
def run(seeds):
 rows=[]
 for seed in seeds:
  for rho in (0.,.05):
   base,u,v,c,res,experts,pl,pr,router,x=world(seed,rho);target=pred_full(x,base,experts,router)
   payloads={'full_bank':{'base':base,'router':router,'experts':experts},'full_top2':{'base':base,'router':router,'experts':experts,'top_k':np.array([2],'u1')},'mirror_precompose':{'base':base,'router':router,'u':u,'v':v,'codes':c},'native_coeff_precompose':{'base':base,'router':router,'u':u.copy(),'v':v.copy(),'coefficients':c.copy()}}
   methods={'full_bank':lambda:pred_full(x,base,experts,router),'full_top2':lambda:pred_full(x,base,experts,router,True),'mirror_precompose':lambda:pred_code(x,base,u,v,c,router),'native_coeff_precompose':lambda:pred_code(x,base,u,v,c,router)}
   if rho>0:
    payloads['mirror_precompose_private']={**payloads['mirror_precompose'],'private_left':pl,'private_right':pr};payloads['native_coeff_precompose_private']={**payloads['native_coeff_precompose'],'private_left':pl.copy(),'private_right':pr.copy()}
    methods['mirror_precompose_private']=lambda:pred_code(x,base,u,v,c,router,pl,pr);methods['native_coeff_precompose_private']=lambda:pred_code(x,base,u,v,c,router,pl,pr)
   for method,payload in payloads.items():
    path=ROOT/'source'/f'{seed}_rho{rho:g}_{method}.npz';st=time.perf_counter();np.savez_compressed(path,**payload);ser=time.perf_counter()-st;t=time.perf_counter();y=methods[method]();wall=time.perf_counter()-t
    active=N*D*D+N*D*E
    if method in ('full_bank','full_top2'):active+=N*(2 if method=='full_top2' else E)*D*D
    else:
     active+=N*E*R+N*R*D*2
     if method.endswith('_private'):active+=N*E*P*D*2
    rows.append({'seed':seed,'residual_rms':rho,'method':method,'serialized_bytes':path.stat().st_size,'output_nrmse':float(np.sqrt(np.mean((y-target)**2)/np.mean(target**2))),'examples':N,'optimizer_updates':0,'active_mac_proxy':int(active),'inference_wall_time_s':wall,'serialization_wall_time_s':ser})
   assert np.array_equal(methods['mirror_precompose'](),methods['native_coeff_precompose']())
 (ROOT/'source/development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','serialization_wall_time_s','status_note'])
  for r in rows:w.writerow([f"rho{r['residual_rms']:g}",r['seed'],r['method'],r['serialized_bytes'],N,0,r['active_mac_proxy'],r['inference_wall_time_s'],'output_nrmse',r['output_nrmse'],'active_mac_proxy',r['active_mac_proxy'],r['serialization_wall_time_s'],'corrected factorized kernel; development; fresh sealed'])
 return rows
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--seeds',nargs='+',type=int,required=True);print(json.dumps(run(a.parse_args().seeds),indent=2))
