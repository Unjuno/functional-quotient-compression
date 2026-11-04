import argparse,csv,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D,E,R,N=32,16,4,512

def softmax(x):
 z=x-x.max(-1,keepdims=True);z=np.exp(z);return z/z.sum(-1,keepdims=True)
def world(seed,rho):
 q=np.random.default_rng(seed+int(rho*10000))
 base=(q.normal(size=(D,D))/np.sqrt(D)*.2).astype('f')
 u=(q.normal(size=(R,D))/np.sqrt(D)).astype('f');v=(q.normal(size=(R,D))/np.sqrt(D)).astype('f')
 coeff=q.normal(size=(E,R)).astype('f')*.3
 atoms=np.einsum('ro,ri->roi',u,v);shared=np.einsum('er,roi->eoi',coeff,atoms)
 residual=q.normal(size=(E,D,D)).astype('f');residual*=rho*np.sqrt(np.mean(shared**2))/np.sqrt(np.mean(residual**2)) if rho else 0
 experts=shared+residual
 router=(q.normal(size=(D,E))/np.sqrt(D)).astype('f');x=q.normal(size=(N,D)).astype('f')
 return base,u,v,coeff,experts,residual,router,x
def pred_full(x,base,experts,router,topk=None):
 g=softmax(x@router); 
 if topk:
  ids=np.argpartition(g,-topk,axis=1)[:,-topk:];m=np.zeros_like(g);np.put_along_axis(m,ids,1,axis=1);g*=m;g/=g.sum(1,keepdims=True)
 outs=np.einsum('nd,eod->neo',x,experts);return x@base.T+np.einsum('ne,neo->no',g,outs)
def pred_code(x,base,u,v,c,router):
 g=softmax(x@router);mix=g@c;z=x@v.T;return x@base.T+np.einsum('nr,ro,rn->no',mix,u,z) # corrected below to explicit stable contraction

def pred_code(x,base,u,v,c,router):
 g=softmax(x@router); mix=g@c; atom_out=np.einsum('nd,rod->nro',x,np.einsum('ro,ri->roi',u,v)); return x@base.T+np.einsum('nr,nro->no',mix,atom_out)
def run(seeds):
 rows=[]
 for seed in seeds:
  for rho in (0.,.05):
   base,u,v,c,experts,res,router,x=world(seed,rho);target=pred_full(x,base,experts,router)
   payloads={'full_bank':{'base':base,'router':router,'experts':experts},'mirror_precompose':{'base':base,'router':router,'u':u,'v':v,'codes':c},'native_coeff_precompose':{'base':base,'router':router,'u':u,'v':v,'coefficients':c},'full_top2':{'base':base,'router':router,'experts':experts,'top_k':np.array([2],dtype='u1')}}
   methods={'full_bank':lambda:pred_full(x,base,experts,router),'mirror_precompose':lambda:pred_code(x,base,u,v,c,router),'native_coeff_precompose':lambda:pred_code(x,base,u,v,c,router),'full_top2':lambda:pred_full(x,base,experts,router,2)}
   for method,p in payloads.items():
    path=ROOT/'source'/f'{seed}_rho{rho:g}_{method}.npz';t=time.perf_counter();np.savez_compressed(path,**p);serial=time.perf_counter()-t
    t=time.perf_counter();y=methods[method]();wall=time.perf_counter()-t
    active=N*D*D+N*D*E+(N*E*D*D if method in ('full_bank','full_top2') else N*E*R+N*R*D*2)
    if method=='full_top2':active=N*D*D+N*D*E+N*2*D*D
    rows.append({'seed':seed,'residual_rms':rho,'method':method,'serialized_bytes':path.stat().st_size,'output_nrmse':float(np.sqrt(np.mean((y-target)**2)/np.mean(target**2))),'examples':N,'optimizer_updates':0,'active_mac_proxy':int(active),'inference_wall_time_s':wall,'serialization_wall_time_s':serial})
   assert np.allclose(methods['mirror_precompose'](),methods['native_coeff_precompose'](),atol=0,rtol=0)
 (ROOT/'source/development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','serialization_wall_time_s','status_note'])
  for r in rows:w.writerow([f"rho{r['residual_rms']:g}",r['seed'],r['method'],r['serialized_bytes'],N,0,r['active_mac_proxy'],r['inference_wall_time_s'],'output_nrmse',r['output_nrmse'],'active_mac_proxy',r['active_mac_proxy'],r['serialization_wall_time_s'],'development; fresh sealed'])
 return rows
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--seeds',nargs='+',type=int,required=True);print(json.dumps(run(a.parse_args().seeds),indent=2))
