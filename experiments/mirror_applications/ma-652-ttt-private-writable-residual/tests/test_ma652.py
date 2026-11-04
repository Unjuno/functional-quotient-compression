import json
from pathlib import Path
import numpy as np
from run_experiment import ROOT,make_world,online_full,online_basis,choose_mode,RANK

def test_matrix_and_payload_byte_replay():
 rows=json.loads((ROOT/'source'/'development.json').read_text());assert len(rows)==40
 for row in rows:
  path=ROOT/'source'/f"{row['seed']}_r{row['private_residual_rank']}_{row['method']}.npz"
  assert path.stat().st_size==row['serialized_state_bytes']

def test_mirror_native_payloads_exactly_match():
 for seed in (65201,65202):
  for r in (0,2,4,8):
   a=ROOT/'source'/f'{seed}_r{r}_mirror_shared_private.npz';b=ROOT/'source'/f'{seed}_r{r}_native_dictionary_private.npz'
   with np.load(a) as x,np.load(b) as y:
    for key in ('base_initialization','shared_dictionary','stream_codes','private_bases','writable_shared_private_codes'):
     assert np.array_equal(x[key],y[key])

def test_metric_replay_for_full_shared_and_private():
 rows=json.loads((ROOT/'source'/'development.json').read_text())
 for row in rows:
  if row['method'] not in ('full_ttt_linear','shared_only','mirror_shared_private','independent_private'):continue
  seed,r,method=row['seed'],row['private_residual_rank'],row['method'];base,d,streams=make_world(seed,r);loss=[]
  p=ROOT/'source'/f'{seed}_r{r}_{method}.npz'
  with np.load(p) as payload:
   codes=payload['stream_codes'] if 'stream_codes' in payload else None
   for i,s in enumerate(streams):
    if method=='full_ttt_linear': _,m=online_full(base,s)
    elif method=='shared_only':
     code=int(codes[i]);u=d[:,code*RANK:(code+1)*RANK];_,m=online_basis(base,s,u)
    elif method=='mirror_shared_private':
     code=int(codes[i]);u=d[:,code*RANK:(code+1)*RANK];basis=np.concatenate([u,s['v']],axis=1);_,m=online_basis(base,s,basis)
    else:
     basis=np.concatenate([s['u'],s['v']],axis=1);_,m=online_basis(base,s,basis)
    loss.append(m)
  loss=np.mean(loss,axis=0)
  assert abs(float(loss[0])-row['mean_prequential_nll'])<1e-10
  assert abs(float(loss[2])-row['mean_final32_mse'])<1e-10
