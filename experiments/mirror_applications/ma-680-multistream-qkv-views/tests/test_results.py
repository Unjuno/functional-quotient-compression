import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'));import run_experiment as run

def test_serialized_qkv_replay_and_causality():
 rows=json.loads((ROOT/'source/development.json').read_text());assert len(rows)==8;d={(r['seed'],r['method']):r for r in rows}
 for seed in (68001,68002):
  base,atom,code,teacher,x=run.make_world(seed);target=np.stack([run.attend(x[s],teacher[s]) for s in range(run.S)])
  loaded={m:np.load(ROOT/'source'/f'{seed}_{m}.npz') for m in ('independent','hard_tied','mirror_view','native_view')}
  for method,z in loaded.items():
   path=ROOT/'source'/f'{seed}_{method}.npz';row=d[(seed,method)];assert path.stat().st_size==row['serialized_bytes']
   if method=='independent':pred=np.stack([run.attend(x[s],z['qkv'][s]) for s in range(run.S)])
   elif method=='hard_tied':pred=np.stack([run.attend(x[s],z['qkv_base']) for s in range(run.S)])
   else:
    c=z['stream_code'] if 'stream_code' in z else z['role_coefficients'];w=z['qkv_base'][None]+np.einsum('sk,kqoi->sqoi',c,z['view_atoms']);pred=np.stack([run.attend(x[s],w[s]) for s in range(run.S)])
   err=float(np.sqrt(np.mean((pred-target)**2)/np.mean(target**2)));assert abs(err-row['output_nrmse'])<1e-7
  z=loaded['mirror_view'];future=x.copy();future[:,:,run.L//2:]+=1.7;future_out=[]
  for s in range(run.S):
   w=z['qkv_base']+np.einsum('k,kqoi->qoi',z['stream_code'][s],z['view_atoms']);future_out.append(run.attend(future[s],w))
  assert max(float(np.max(abs(target[s,:,:run.L//2]-future_out[s][:,:run.L//2]))) for s in range(run.S))==0.
  for z in loaded.values():z.close()
