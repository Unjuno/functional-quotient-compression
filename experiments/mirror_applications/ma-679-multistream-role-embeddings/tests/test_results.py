import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'));import run_experiment as run

def test_payload_replay_and_stream_isolation():
 rows=json.loads((ROOT/'source/development.json').read_text());assert len(rows)==8;d={(r['seed'],r['method']):r for r in rows}
 for seed in (67901,67902):
  base,atom,codes,teacher,inputs,hold=run.make_world(seed);ids=hold
  loaded={m:np.load(ROOT/'source'/f'{seed}_{m}.npz') for m in ('independent','hard_tied','mirror_role_code','native_rank1_role')}
  gold=loaded['independent']['stream_embeddings'][:,ids]
  for method,z in loaded.items():
   row=d[(seed,method)];path=ROOT/'source'/f'{seed}_{method}.npz';assert path.stat().st_size==row['serialized_bytes']
   if method=='independent':pred=z['stream_embeddings'][:,ids]
   elif method=='hard_tied':pred=np.broadcast_to(z['base_embedding'][ids],(run.S,len(ids),run.D))
   else:pred=run.lookup_mirror(ids,z['base_embedding'],z['role_atom'],z['role_code'] if 'role_code' in z else z['role_coefficients'])
   err=float(np.sqrt(np.mean((pred-gold)**2)/np.mean(gold**2)));assert abs(err-row['output_nrmse'])<1e-7
  z=loaded['mirror_role_code'];probe=inputs[:,:16];orig=run.lookup_mirror(probe,z['base_embedding'],z['role_atom'],z['role_code']);leak=0.
  for s in range(run.S):
   p=probe.copy()
   for other in range(run.S):
    if other!=s:p[other]=(p[other]+17)%run.V
   changed=run.lookup_mirror(p,z['base_embedding'],z['role_atom'],z['role_code']);leak=max(leak,float(np.max(np.abs(orig[s]-changed[s]))))
  assert leak==d[(seed,'mirror_role_code')]['cross_stream_leakage']==0.
  for z in loaded.values():z.close()
