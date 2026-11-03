import argparse,csv,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def run(seed):
 rng=np.random.default_rng(seed);n=128;kd=8;vd=16;keys=rng.normal(size=(n,kd));keys/=np.linalg.norm(keys,axis=1,keepdims=True);q,_=np.linalg.qr(rng.normal(size=(vd,4)));latent=rng.normal(size=(n,4));values=latent@q.T;base=rng.normal(0,.1,(vd,kd)).astype(np.float32);radii=np.full(n,.08,dtype=np.float32)
 train=96;_,_,vh=np.linalg.svd(values[:train],full_matrices=False);basis=vh[:4].T;codes=values@basis
 rngq=np.random.default_rng(seed+91);pos=[];posid=[];neg=[]
 for i in range(n):
  for _ in range(4):pos.append(keys[i]+rngq.normal(0,.01,kd));posid.append(i)
 while len(neg)<n*4:
  z=rngq.uniform(-2,2,kd)
  if np.min(np.linalg.norm(keys-z,axis=1))>.2:neg.append(z)
 pos=np.asarray(pos);posid=np.asarray(posid);neg=np.asarray(neg)
 def route(x):
  dist=np.linalg.norm(x[:,None,:]-keys[None,:,:],axis=2);idx=np.argmin(dist,axis=1);d=dist[np.arange(len(x)),idx];return idx,d<=radii[idx]
 payloads={'serac_explicit':{'base':base,'keys':keys.astype(np.float32),'radii':radii,'values':values.astype(np.float32),'edit_ids':np.arange(n,dtype=np.uint8),'schema':np.frombuffer(b'SERAC-EXPLICIT-V1',dtype=np.uint8)},
 'mirror_value_code':{'base':base,'keys':keys.astype(np.float32),'radii':radii,'value_basis':basis.astype(np.float32),'value_codes':codes.astype(np.float32),'edit_ids':np.arange(n,dtype=np.uint8),'schema':np.frombuffer(b'SERAC-VIEW-R4-V1',dtype=np.uint8)},
 'native_pca_value_code':{'base':base,'keys':keys.astype(np.float32),'radii':radii,'value_basis':basis.astype(np.float32),'value_codes':codes.astype(np.float32),'edit_ids':np.arange(n,dtype=np.uint8),'schema':np.frombuffer(b'SERAC-VIEW-R4-V1',dtype=np.uint8)}}
 paths={};states={}
 for name,arr in payloads.items():
  path=R/'source'/f'{seed}_{name}.npz';np.savez_compressed(path,**arr);paths[name]=path.stat().st_size
  sp=R/'source'/f'{seed}_{name}_state.npz';np.savez_compressed(sp,**{k:v for k,v in arr.items() if k!='base'});states[name]=sp.stat().st_size
 query=np.concatenate([pos,neg]);idx,active=route(query);rows=[]
 for method in payloads:
  start=time.perf_counter();
  if method=='serac_explicit':out=np.zeros((len(query),vd));out[active]=values[idx[active]]
  else:out=np.zeros((len(query),vd));out[active]=codes[idx[active]]@basis.T
  latency=time.perf_counter()-start
  posout=out[:len(pos)];nrmse=float(np.linalg.norm(posout-values[posid])/np.linalg.norm(values[posid]));hit=float(np.mean(active[:len(pos)]));fp=float(np.mean(active[len(pos):]));ops=int(len(query)*n*kd+int(np.sum(active))*vd*(1 if method=='serac_explicit' else 4))
  rows.append({'seed':seed,'method':method,'serialized_bytes':paths[method],'edit_state_bytes':states[method],'heldout_value_nrmse':nrmse,'positive_scope_hit_rate':hit,'negative_false_positive_rate':fp,'active_compute_proxy':ops,'retrieval_decode_latency_s':latency,'optimizer_updates':0})
 return rows
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);args=ap.parse_args();start=time.perf_counter();rows=[]
 for s in args.seeds:rows.extend(run(s))
 wall=time.perf_counter()-start
 for r in rows:r['wall_time_s']=wall
 (R/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (R/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
  for r in rows:w.writerow(['rank4_counterfactual_values',r['seed'],r['method'],r['serialized_bytes'],96,0,r['active_compute_proxy'],r['wall_time_s'],'heldout_value_nrmse',r['heldout_value_nrmse'],'negative_false_positive_rate',r['negative_false_positive_rate'],'development; fresh sealed'])
 print(json.dumps(rows,indent=2))
