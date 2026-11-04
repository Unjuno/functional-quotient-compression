import argparse,csv,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def run(seed):
 rng=np.random.default_rng(seed);d=16;base=rng.normal(0,.1,(d,d)).astype(np.float32);u=rng.normal(size=d);u/=np.linalg.norm(u);v=rng.normal(size=d);v-=u*(u@v);v/=np.linalg.norm(v);rows=[]
 role_ids=np.array([0,1],dtype=np.uint8);schema=np.frombuffer(b'RRDA-DUAL-ROLE-V1',dtype=np.uint8)
 for deg in [0,15,30,60,90]:
  th=np.deg2rad(deg);targets=np.stack([u,-np.cos(th)*u+np.sin(th)*v]);_,_,vh=np.linalg.svd(targets,full_matrices=False)
  pca=vh[:2].T
  for rank in [1,2]:
   basis=pca[:,:rank];codes=targets@basis;decoded=codes@basis.T
   if rank==1:
    gate_basis=u[:,None];gate_codes=targets@gate_basis;gate_dec=gate_codes@gate_basis.T
   else:gate_basis=basis;gate_codes=codes;gate_dec=decoded
   methods={
    'independent_role_adapters':{'base_executor':base,'role_ids':role_ids,'role_values':targets.astype(np.float32),'router_schema':schema},
    f'mirror_shared_basis_r{rank}':{'base_executor':base,'role_ids':role_ids,'shared_basis':basis.astype(np.float32),'role_codes':codes.astype(np.float32),'router_schema':schema},
    'native_gate_rank1' if rank==1 else 'native_shared_lowrank_rank2':{'base_executor':base,'role_ids':role_ids,'shared_basis':gate_basis.astype(np.float32),'role_codes':gate_codes.astype(np.float32),'router_schema':schema}}
   for name,payload in methods.items():
    path=R/'source'/f'{seed}_theta{deg}_{name}.npz';np.savez_compressed(path,**payload);full_bytes=path.stat().st_size
    state={k:x for k,x in payload.items() if k!='base_executor'};sp=R/'source'/f'{seed}_theta{deg}_{name}_state.npz';np.savez_compressed(sp,**state);state_bytes=sp.stat().st_size
    if name=='independent_role_adapters':pred=targets;ops=2*d
    elif name=='native_gate_rank1':pred=gate_dec;ops=2*d
    elif rank==1:pred=decoded;ops=rank*d+2*rank
    else:pred=decoded;ops=rank*d+2*rank
    errs=np.linalg.norm(pred-targets,axis=1)/np.linalg.norm(targets,axis=1)
    rows.append({'seed':seed,'role_angle_degrees':deg,'method':name,'serialized_bytes':full_bytes,'role_state_bytes':state_bytes,'edit_role_nrmse':float(errs[0]),'suppress_role_nrmse':float(errs[1]),'active_compute_proxy':int(ops),'optimizer_updates':0})
 return rows
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);args=ap.parse_args();t=time.perf_counter();rows=[]
 for s in args.seeds:rows.extend(run(s))
 wall=time.perf_counter()-t
 for r in rows:r['wall_time_s']=wall
 (R/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (R/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
  for r in rows:w.writerow([f"role_angle_{r['role_angle_degrees']}",r['seed'],r['method'],r['serialized_bytes'],0,0,r['active_compute_proxy'],r['wall_time_s'],'edit_role_nrmse',r['edit_role_nrmse'],'suppress_role_nrmse',r['suppress_role_nrmse'],'development; fresh sealed'])
 print(json.dumps(rows,indent=2))
