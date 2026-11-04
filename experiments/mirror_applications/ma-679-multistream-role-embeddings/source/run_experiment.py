import argparse,csv,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];S,V,D,K,N=4,64,16,1,128

def make_world(seed):
 q=np.random.default_rng(seed);base=(q.normal(size=(V,D))/D**.5).astype('f');atom=(q.normal(size=(K,V,D))/D**.5).astype('f');codes=q.normal(size=(S,K)).astype('f');teacher=base[None]+np.einsum('sk,kvd->svd',codes,atom)
 inputs=q.integers(0,V,size=(S,N));hold=np.arange(V-24,V,dtype=int);return base,atom,codes,teacher,inputs,hold

def lookup_mirror(ids,base,atom,codes):return base[ids]+np.einsum('sk,kd->sd',codes,atom[:,ids].transpose(1,0,2))

def run(seeds):
 rows=[]
 for seed in seeds:
  base,atom,codes,teacher,inputs,hold=make_world(seed);payloads={'independent':{'stream_embeddings':teacher},'hard_tied':{'base_embedding':base},'mirror_role_code':{'base_embedding':base,'role_atom':atom,'role_code':codes},'native_rank1_role':{'base_embedding':base.copy(),'role_atom':atom.copy(),'role_coefficients':codes.copy()}}
  for method,payload in payloads.items():
   path=ROOT/'source'/f'{seed}_{method}.npz';st=time.perf_counter();np.savez_compressed(path,**payload);serial=time.perf_counter()-st
   st=time.perf_counter()
   if method=='independent':pred=payload['stream_embeddings'][:,hold]
   elif method=='hard_tied':pred=np.broadcast_to(payload['base_embedding'][hold],(S,len(hold),D))
   else:pred=lookup_mirror(hold,payload['base_embedding'],payload['role_atom'],payload['role_code'] if 'role_code' in payload else payload['role_coefficients'])
   wall=time.perf_counter()-st;target=teacher[:,hold];err=float(np.sqrt(np.mean((pred-target)**2)/np.mean(target**2)))
   # Perturb every non-target stream's token IDs. Query outputs must remain bit-identical.
   test_ids=inputs[:,:16].copy();base_out=lookup_mirror(test_ids,base,atom,codes);perturbed=test_ids.copy()
   for s in range(S):
    for other in range(S):
     if other!=s:perturbed[other]=(perturbed[other]+17)%V
   changed=lookup_mirror(perturbed,base,atom,codes);leak=float(max(np.max(np.abs(base_out[s]-changed[s])) for s in range(S)))
   mac=int(S*len(hold)*D*(K if method in ('mirror_role_code','native_rank1_role') else 1))
   rows.append({'seed':seed,'residual_rms':0.,'method':method,'serialized_bytes':path.stat().st_size,'output_nrmse':err,'examples':S*len(hold),'optimizer_updates':0,'active_mac_proxy':mac,'inference_wall_time_s':wall,'serialization_wall_time_s':serial,'cross_stream_leakage':leak})
  assert np.array_equal(lookup_mirror(hold,base,atom,codes),lookup_mirror(hold,base,atom,codes.copy()))
 (ROOT/'source/development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','serialization_wall_time_s','status_note'])
  for r in rows:w.writerow(['heldout_tokens',r['seed'],r['method'],r['serialized_bytes'],r['examples'],0,r['active_mac_proxy'],r['inference_wall_time_s'],'output_nrmse',r['output_nrmse'],'cross_stream_leakage',r['cross_stream_leakage'],r['serialization_wall_time_s'],'development; fresh sealed'])
 return rows
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,required=True);print(json.dumps(run(p.parse_args().seeds),indent=2))
