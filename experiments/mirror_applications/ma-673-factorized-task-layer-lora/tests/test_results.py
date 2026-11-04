import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D,T,L,R,N=24,12,6,4,96
def test_payload_replay_bytes_and_native_alias():
 rows=json.loads((ROOT/'source/development.json').read_text());assert len(rows)==6
 by={(x['seed'],x['method']):x for x in rows}
 for seed in (67301,67302):
  rng=np.random.default_rng(seed);x=rng.normal(size=(T,L,N,D)).astype('f')
  loaded={m:np.load(ROOT/'source'/f'{seed}_{m}.npz') for m in ('independent','mirror_product','native_outer')}
  for m,z in loaded.items():
   row=by[(seed,m)];assert (ROOT/'source'/f'{seed}_{m}.npz').stat().st_size==row['serialized_bytes']
   preds=[];truth=[]
   if m=='independent': c=z['coefficients']
   elif m=='mirror_product': c=z['task_factor'][:,None,:]*z['layer_factor'][None,:,:]
   else:c=z['task_left'][:,None,:]*z['layer_right'][None,:,:]
   for t in range(T):
    for l in range(L):
     if (t*7+l*11+seed)%4: continue
     out=np.einsum('nr,ro->no',(x[t,l]@z['atom_right'][l].T)*c[t,l][None,:],z['atom_left'][l])
     preds.append(out);truth.append(np.einsum('nr,ro->no',(x[t,l]@loaded['independent']['atom_right'][l].T)*loaded['independent']['coefficients'][t,l][None,:],loaded['independent']['atom_left'][l]))
   preds=np.stack(preds);truth=np.stack(truth);err=float(np.sqrt(np.mean((preds-truth)**2)/np.mean(truth**2)))
   assert abs(err-row['output_nrmse'])<1e-7
  assert np.array_equal(loaded['mirror_product']['task_factor'],loaded['native_outer']['task_left'])
  assert np.array_equal(loaded['mirror_product']['layer_factor'],loaded['native_outer']['layer_right'])
  for z in loaded.values():z.close()
