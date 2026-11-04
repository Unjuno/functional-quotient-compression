import argparse,csv,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];S,D,L,B,K=4,16,12,32,1

def softmax(a):
 a=a-a.max(-1,keepdims=True);z=np.exp(a);return z/z.sum(-1,keepdims=True)
def attend(x,w,causal=True):
 q=x@w[0].T;k=x@w[1].T;v=x@w[2].T;s=q@k.transpose(0,2,1)/np.sqrt(D)
 if causal:s=np.where(np.triu(np.ones((L,L),dtype=bool),1),-1e9,s)
 return softmax(s)@v
def make_world(seed):
 r=np.random.default_rng(seed);base=(r.normal(size=(3,D,D))/D**.5*.2).astype('f');atom=(r.normal(size=(K,3,D,D))/D**.5*.2).astype('f');code=r.normal(size=(S,K)).astype('f');teacher=base[None]+np.einsum('sk,kqoi->sqoi',code,atom);x=r.normal(size=(S,B,L,D)).astype('f');return base,atom,code,teacher,x
def run(seeds):
 rows=[]
 for seed in seeds:
  base,atom,code,teacher,x=make_world(seed);payloads={'independent':{'qkv':teacher,'causal_mask':np.array([1],'u1')},'hard_tied':{'qkv_base':base,'causal_mask':np.array([1],'u1')},'mirror_view':{'qkv_base':base,'view_atoms':atom,'stream_code':code,'causal_mask':np.array([1],'u1')},'native_view':{'qkv_base':base.copy(),'view_atoms':atom.copy(),'role_coefficients':code.copy(),'causal_mask':np.array([1],'u1')}}
  for method,p in payloads.items():
   path=ROOT/'source'/f'{seed}_{method}.npz';st=time.perf_counter();np.savez_compressed(path,**p);serial=time.perf_counter()-st
   st=time.perf_counter()
   if method=='independent':pred=np.stack([attend(x[s],p['qkv'][s]) for s in range(S)])
   elif method=='hard_tied':pred=np.stack([attend(x[s],p['qkv_base']) for s in range(S)])
   else:
    co=p['stream_code'] if 'stream_code' in p else p['role_coefficients'];w=p['qkv_base'][None]+np.einsum('sk,kqoi->sqoi',co,p['view_atoms']);pred=np.stack([attend(x[s],w[s]) for s in range(S)])
   wall=time.perf_counter()-st;target=np.stack([attend(x[s],teacher[s]) for s in range(S)]);err=float(np.sqrt(np.mean((pred-target)**2)/np.mean(target**2)))
   # causal future-token perturbation
   future=x.copy();future[:,:,L//2:]+=1.7;future_out=np.stack([attend(future[s],teacher[s]) for s in range(S)]);future_leak=float(np.max(np.abs(target[:,:,:L//2]-future_out[:,:,:L//2])))
   # cross-stream isolation
   cross=x.copy();cross[1:]+=2.;cross_out=np.stack([attend(cross[s],teacher[s]) for s in range(S)]);cross_leak=float(np.max(np.abs(target[0]-cross_out[0])))
   mac=int(S*B*(3*L*D*D+2*L*L*D)+(S*K*3*D*D if method in ('mirror_view','native_view') else 0))
   rows.append({'seed':seed,'residual_rms':0.,'method':method,'serialized_bytes':path.stat().st_size,'output_nrmse':err,'examples':S*B*L,'optimizer_updates':0,'active_mac_proxy':mac,'inference_wall_time_s':wall,'serialization_wall_time_s':serial,'future_leakage':future_leak,'cross_stream_leakage':cross_leak})
  assert np.array_equal(teacher,base[None]+np.einsum('sk,kqoi->sqoi',code,atom))
 (ROOT/'source/development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','serialization_wall_time_s','status_note'])
  for r in rows:w.writerow(['synthetic_causal_attention',r['seed'],r['method'],r['serialized_bytes'],r['examples'],0,r['active_mac_proxy'],r['inference_wall_time_s'],'output_nrmse',r['output_nrmse'],'future_and_cross_leakage',max(r['future_leakage'],r['cross_stream_leakage']),r['serialization_wall_time_s'],'development; fresh sealed'])
 return rows
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--seeds',nargs='+',type=int,required=True);print(json.dumps(run(a.parse_args().seeds),indent=2))
