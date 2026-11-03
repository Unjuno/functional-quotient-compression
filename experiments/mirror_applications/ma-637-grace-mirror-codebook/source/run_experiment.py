import argparse,csv,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def queries(keys,rng):
 pos=[];posid=[];neg=[]
 for i,k in enumerate(keys):
  for _ in range(4):pos.append(k+rng.normal(0,.01,len(k)));posid.append(i)
 while len(neg)<4*len(keys):
  x=rng.uniform(-2,2,keys.shape[1])
  if np.min(np.linalg.norm(keys-x,axis=1))>.2:neg.append(x)
 return np.array(pos),np.array(posid),np.array(neg)
def run(seed):
 rng=np.random.default_rng(seed);n=128;kd=8;vd=16;latentdim=4;Q,_=np.linalg.qr(rng.normal(size=(vd,latentdim)));cal=rng.normal(size=(16,latentdim))@Q.T;_,_,vh=np.linalg.svd(cal,full_matrices=False);basis=vh[:4].T
 keys=rng.normal(size=(n,kd));keys/=np.linalg.norm(keys,axis=1,keepdims=True);codes=rng.normal(size=(n,latentdim));values=codes@Q.T;base=rng.normal(0,.1,(vd,kd)).astype(np.float32);rows=[]
 for count in [16,32,64,128]:
  k=keys[:count];v=values[:count];c=v@basis;radii=np.full(count,.08,dtype=np.float32);ids=np.arange(count,dtype=np.uint8);age=np.arange(count,dtype=np.uint16);schema=np.frombuffer(b'GRACE-DKV-V1',dtype=np.uint8)
  methods={'grace_explicit':{'base':base,'keys':k.astype(np.float32),'values':v.astype(np.float32),'radii':radii,'edit_ids':ids,'age':age,'schema':schema},
   'mirror_basis_codes':{'base':base,'keys':k.astype(np.float32),'value_basis':basis.astype(np.float32),'value_codes':c.astype(np.float32),'radii':radii,'edit_ids':ids,'age':age,'schema':np.frombuffer(b'GRACE-VIEW-R4-V1',dtype=np.uint8)},
   'native_pca_codes':{'base':base,'keys':k.astype(np.float32),'value_basis':basis.astype(np.float32),'value_codes':c.astype(np.float32),'radii':radii,'edit_ids':ids,'age':age,'schema':np.frombuffer(b'GRACE-VIEW-R4-V1',dtype=np.uint8)}}
  sizes={};fixed={}
  for name,payload in methods.items():
   path=R/'source'/f'{seed}_n{count}_{name}.npz';np.savez_compressed(path,**payload);sizes[name]=path.stat().st_size
   state={x:y for x,y in payload.items() if x not in ['keys','values','value_codes','edit_ids','age']};sp=R/'source'/f'{seed}_n{count}_{name}_fixed.npz';np.savez_compressed(sp,**state);fixed[name]=sp.stat().st_size
  pos,posid,neg=queries(k,np.random.default_rng(seed+count*17));query=np.concatenate([pos,neg]);dist=np.linalg.norm(query[:,None,:]-k[None,:,:],axis=2);idx=np.argmin(dist,axis=1);active=dist[np.arange(len(query)),idx]<=radii[idx]
  for method in methods:
   t0=time.perf_counter()
   if method=='grace_explicit':out=np.zeros((len(query),vd));out[active]=v[idx[active]]
   else:out=np.zeros((len(query),vd));out[active]=c[idx[active]]@basis.T
   latency=time.perf_counter()-t0;positive=out[:len(pos)];err=float(np.linalg.norm(positive-v[posid])/np.linalg.norm(v[posid]));hit=float(np.mean(active[:len(pos)]));fp=float(np.mean(active[len(pos):]));ops=int(len(query)*count*kd+int(np.sum(active))*vd*(1 if method=='grace_explicit' else 4))
   rows.append({'seed':seed,'snapshot_edits':count,'method':method,'serialized_bytes':sizes[method],'fixed_state_bytes':fixed[method],'bytes_per_edit':sizes[method]/count,'retention_nrmse':err,'positive_scope_hit_rate':hit,'negative_false_positive_rate':fp,'active_compute_proxy':ops,'lookup_decode_latency_s':latency,'optimizer_updates':0})
 return rows
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);args=ap.parse_args();t=time.perf_counter();rows=[]
 for s in args.seeds:rows.extend(run(s))
 wall=time.perf_counter()-t
 for r in rows:r['wall_time_s']=wall
 (R/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (R/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
  for r in rows:w.writerow([f"edits_{r['snapshot_edits']}",r['seed'],r['method'],r['serialized_bytes'],16,0,r['active_compute_proxy'],r['wall_time_s'],'retention_nrmse',r['retention_nrmse'],'negative_false_positive_rate',r['negative_false_positive_rate'],'development; fresh sealed'])
 print(json.dumps(rows,indent=2))
