import argparse,csv,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D,K,UNSEEN,QCOUNT=32,4,16,128
SCHEMA=np.frombuffer(b"STORED-PROGRAM-CODE-V1",dtype=np.uint8)

def make_world(seed,n,rho):
 rng=np.random.default_rng(seed*1000+n);flat,_=np.linalg.qr(rng.normal(size=(D*D,K)));basis=flat.T.reshape(K,D,D).astype(np.float32)
 coeff=rng.normal(size=(n,K)).astype(np.float32);unseen=rng.normal(size=(UNSEEN,K)).astype(np.float32)
 raw=rng.normal(size=(n,D*D));raw-= (raw@flat)@flat.T
 programs0=coeff@flat.T
 scale=np.sqrt(np.mean(programs0**2,axis=1,keepdims=True));raw*=scale/np.maximum(np.sqrt(np.mean(raw**2,axis=1,keepdims=True)),1e-12)
 programs=(programs0+rho*raw).reshape(n,D,D).astype(np.float32)
 unseen_targets=(unseen@flat.T).reshape(UNSEEN,D,D).astype(np.float32)
 x=rng.normal(size=(n,QCOUNT,D)).astype(np.float32);xu=rng.normal(size=(UNSEEN,QCOUNT,D)).astype(np.float32)
 return basis,coeff,programs,unseen,unseen_targets,x,xu

def fit_metrics(pred,targets,x):
 yt=np.einsum('nqd,ndh->nqh',x,targets);yp=np.einsum('nqd,ndh->nqh',x,pred)
 e=np.mean((yp-yt)**2,axis=(1,2));den=np.mean(yt**2,axis=(1,2));return float(np.mean(np.sqrt(e/np.maximum(den,1e-12))))

def unique_fraction(mats):
 flat=mats.reshape(len(mats),-1);d=np.linalg.norm(flat[:,None]-flat[None,:],axis=-1);return float(np.mean((d+np.eye(len(mats)))>1e-5))

def save(seed,n,rho,name,payload):
 p=ROOT/'source'/f'{seed}_n{n}_rho{rho:g}_{name}.npz';t=time.perf_counter();np.savez_compressed(p,**payload);return p.stat().st_size,time.perf_counter()-t

def run_condition(seed,n,rho):
 basis,coeff,programs,unseen,unseen_targets,x,xu=make_world(seed,n,rho);flat=programs.reshape(n,-1)
 # Learned Mirror/native basis is the rank-4 PCA of stored controller functions.
 _,_,vh=np.linalg.svd(flat,full_matrices=False);pca=vh[:K].reshape(K,D,D).astype(np.float32);codes=flat@vh[:K].T
 # The explicit NSPM control stores the generating basis and coefficients.
 nsm_pred=np.einsum('nk,kdh->ndh',coeff,basis);mirror_pred=np.einsum('nk,kdh->ndh',codes,pca)
 # Non-interpolating one-hot program selection with a scalar magnitude.
 idx=np.argmax(np.abs(codes),axis=1);onehot_codes=np.zeros_like(codes);onehot_codes[np.arange(n),idx]=codes[np.arange(n),idx];onehot_pred=np.einsum('nk,kdh->ndh',onehot_codes,pca)
 unseen_pca=np.einsum('nk,kdh->ndh',unseen,pca);unseen_nsm=np.einsum('nk,kdh->ndh',unseen,basis)
 unseen_idx=np.argmax(np.abs(unseen),axis=1);uc=np.zeros_like(unseen);uc[np.arange(UNSEEN),unseen_idx]=unseen[np.arange(UNSEEN),unseen_idx];unseen_onehot=np.einsum('nk,kdh->ndh',uc,basis)
 payloads={
  'independent_full_programs':{'schema':SCHEMA,'program_ids':np.arange(n,dtype=np.uint16),'program_matrices':programs},
  'nsm_explicit_basis_codes':{'schema':SCHEMA,'program_ids':np.arange(n,dtype=np.uint16),'basis_controllers':basis,'program_codes':coeff},
  'mirror_pca_rank4':{'schema':SCHEMA,'program_ids':np.arange(n,dtype=np.uint16),'shared_function_basis':pca,'program_codes':codes.astype(np.float32)},
  'native_pca_rank4':{'schema':SCHEMA,'program_ids':np.arange(n,dtype=np.uint16),'shared_function_basis':pca.copy(),'program_codes':codes.astype(np.float32).copy()},
  'onehot_nearest_program':{'schema':SCHEMA,'program_ids':np.arange(n,dtype=np.uint16),'basis_controllers':basis,'selected_program_ids':idx.astype(np.uint8),'selected_scales':codes[np.arange(n),idx].astype(np.float32)},
 }
 predmap={'independent_full_programs':programs,'nsm_explicit_basis_codes':nsm_pred,'mirror_pca_rank4':mirror_pred,'native_pca_rank4':mirror_pred,'onehot_nearest_program':onehot_pred}
 unseennrmse={'independent_full_programs':None,'nsm_explicit_basis_codes':fit_metrics(unseen_nsm,unseen_targets,xu),'mirror_pca_rank4':fit_metrics(unseen_pca,unseen_targets,xu),'native_pca_rank4':fit_metrics(unseen_pca,unseen_targets,xu),'onehot_nearest_program':fit_metrics(unseen_onehot,unseen_targets,xu)}
 rows=[]
 for name,payload in payloads.items():
  size,wall=save(seed,n,rho,name,payload);decoded=predmap[name]
  rows.append({'seed':seed,'program_count':n,'private_function_residual_rms':rho,'method':name,'serialized_bytes':size,'stored_program_nrmse':fit_metrics(decoded,programs,x),'unseen_code_nrmse':unseennrmse[name],'unique_function_fraction':unique_fraction(decoded),'stored_query_examples':n*QCOUNT,'unseen_query_examples':UNSEEN*QCOUNT,'optimizer_updates':0,'basis_decode_mac_proxy':int(n*K*D*D if 'basis' in name or 'pca' in name or 'onehot' in name else 0),'program_execute_mac_proxy':int((n+UNSEEN)*QCOUNT*2*D*D),'offline_basis_svd_count':int(name in ('mirror_pca_rank4','native_pca_rank4')),'wall_time_s':wall})
 assert np.array_equal(payloads['mirror_pca_rank4']['shared_function_basis'],payloads['native_pca_rank4']['shared_function_basis'])
 assert np.array_equal(payloads['mirror_pca_rank4']['program_codes'],payloads['native_pca_rank4']['program_codes'])
 return rows

def run(seeds):
 rows=[x for s in seeds for n in (16,64) for rho in (0.,.05,.2) for x in run_condition(s,n,rho)]
 (ROOT/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
  for x in rows:w.writerow([f"n{x['program_count']}_rho{x['private_function_residual_rms']:g}",x['seed'],x['method'],x['serialized_bytes'],x['stored_query_examples']+x['unseen_query_examples'],0,x['basis_decode_mac_proxy']+x['program_execute_mac_proxy'],x['wall_time_s'],'stored_program_nrmse',x['stored_program_nrmse'],'unseen_code_nrmse',x['unseen_code_nrmse'],'development; fresh sealed'])
 return rows
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);a=ap.parse_args();print(json.dumps(run(a.seeds),indent=2))
