import argparse,csv,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=32;M=32;KP=8;R=4;N=32;RHOS=(0.,.05,.2);QRANKS=(0,1,2,4)
def orthogonal_complement(against,rng,n):
 vh=np.linalg.svd(against.T,full_matrices=True)[2];comp=vh[against.shape[1]:].T;z=comp@rng.normal(size=(comp.shape[1],n));q,_=np.linalg.qr(z);return q[:,:n]
def make_world(seed,rho):
 rng=np.random.default_rng(seed+int(rho*10000));k,_=np.linalg.qr(rng.normal(size=(D,KP)));protected=k.astype('f');null=np.linalg.svd(protected.T,full_matrices=True)[2][KP:].T.astype('f')
 ush,_=np.linalg.qr(rng.normal(size=(D,R)));ush=ush.astype('f');vsh,_=np.linalg.qr(null@rng.normal(size=(D-KP,R)));vsh=vsh.astype('f')
 pu=np.zeros((M,2,D),'f');pv=np.zeros((M,2,D),'f')
 ucomp=orthogonal_complement(ush,rng,2);vcomp=orthogonal_complement(np.concatenate([protected,vsh],axis=1),rng,2)
 for m in range(M):
  qu,_=np.linalg.qr(ucomp@rng.normal(size=(ucomp.shape[1],2)));qv,_=np.linalg.qr(vcomp@rng.normal(size=(vcomp.shape[1],2)));pu[m]=qu[:,:2].T;pv[m]=qv[:,:2].T
 codes=rng.normal(size=(M,R)).astype('f')*.5;pc=rng.normal(size=(M,2)).astype('f');basis=np.einsum('or,ir->roi',ush,vsh)
 shared=np.einsum('mr,roi->moi',codes,basis);private=rho*np.einsum('mq,mqo,mqi->moi',pc,pu,pv);deltas=shared+private
 x=rng.normal(size=(M,N,D)).astype('f');return protected,basis,deltas,x

def fit_shared(train_deltas):
 mat=train_deltas.reshape(len(train_deltas),-1);_,_,vh=np.linalg.svd(mat,full_matrices=False);basis=vh[:R].reshape(R,D,D).astype('f');codes=mat@vh[:R].T;return basis,codes.astype('f')
def factor_private(resid,q):
 left=np.zeros((M,q,D),'f');right=np.zeros((M,q,D),'f')
 if q:
  for m in range(M):
   u,s,vh=np.linalg.svd(resid[m],full_matrices=False);left[m]=u[:,:q].T*s[:q,None];right[m]=vh[:q]
 return left,right
def decode(p):
 if 'projected_deltas' in p:return p['projected_deltas']
 basis=p['basis'];codes=p['codes'] if 'codes' in p else p['coefficients'];out=np.einsum('mr,roi->moi',codes,basis)
 if 'private_left' in p:out+=np.einsum('mq,mqo,mqi->moi',np.ones((M,p['private_left'].shape[1]),'f'),p['private_left'],p['private_right'])
 return out
def eval_metrics(deltas,x,protected,held):
 target=np.einsum('moi,mni->mno',deltas[held],x[held]);t=time.perf_counter();pred=np.einsum('moi,mni->mno',deltas[held],x[held]);wall=time.perf_counter()-t
 err=float(np.sqrt(np.mean((pred-target)**2)/max(float(np.mean(target**2)),1e-12)))
 leak=float(np.max(np.linalg.norm(deltas[held]@protected,axis=(1,2))/np.maximum(np.linalg.norm(deltas[held],axis=(1,2)),1e-12)))
 return err,leak,wall
def run(seeds):
 rows=[]
 for seed in seeds:
  for rho in RHOS:
   protected,teacher,x=make_world(seed,rho);shared,codes=fit_shared(teacher[:24]);recon=np.einsum('mr,roi->moi',codes,shared);resid=teacher-recon;held=np.arange(24,32)
   payloads={'projected_full':{'protected_keys':protected,'projected_deltas':teacher},'mirror_q0':{'protected_keys':protected,'basis':shared,'codes':codes}}
   for q in QRANKS[1:]:
    pl,pr=factor_private(resid,q);payloads[f'mirror_q{q}']={'protected_keys':protected,'basis':shared,'codes':codes,'private_left':pl,'private_right':pr}
   for q in (0,2,4):
    key=f'mirror_q{q}';z=payloads[key];native={'protected_keys':protected.copy(),'basis':z['basis'].copy(),'coefficients':z['codes'].copy()}
    if q:native.update(private_left=z['private_left'].copy(),private_right=z['private_right'].copy())
    payloads[f'native_q{q}']=native
   for name,p in payloads.items():
    path=ROOT/'source'/f'{seed}_rho{rho:g}_{name}.npz';st=time.perf_counter();np.savez_compressed(path,**p);serial=time.perf_counter()-st
    with np.load(path) as z:decoded=decode(z)
    err,leak,wall=eval_metrics(decoded,x,protected,held);q=decoded.shape[0];rank=0 if 'projected_deltas' in p else (p['private_left'].shape[1] if 'private_left' in p else 0)
    mac=int(len(held)*N*(D*D if 'projected_deltas' in p else R*D*2+rank*D*2))
    rows.append({'seed':seed,'residual_rms':rho,'method':name,'serialized_bytes':path.stat().st_size,'output_nrmse':err,'examples':len(held)*N,'optimizer_updates':0,'active_mac_proxy':mac,'inference_wall_time_s':wall,'serialization_wall_time_s':serial,'protected_leakage':leak,'private_rank':rank})
 (ROOT/'source/development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','serialization_wall_time_s','status_note'])
  for r in rows:w.writerow([f"rho{r['residual_rms']:g}",r['seed'],r['method'],r['serialized_bytes'],r['examples'],0,r['active_mac_proxy'],r['inference_wall_time_s'],'output_nrmse',r['output_nrmse'],'protected_leakage/private_rank',f"{r['protected_leakage']}/{r['private_rank']}",r['serialization_wall_time_s'],'development; fresh sealed'])
 return rows
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--seeds',nargs='+',type=int,required=True);print(json.dumps(run(a.parse_args().seeds),indent=2))
