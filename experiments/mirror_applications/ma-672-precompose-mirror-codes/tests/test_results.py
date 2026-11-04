import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'));import run_experiment as run

def test_serialized_factor_payload_replay():
 rows=json.loads((ROOT/'source/development.json').read_text());assert len(rows)==20;d={(x['seed'],x['residual_rms'],x['method']):x for x in rows}
 for seed in (67201,67202):
  for rho in (0.,.05):
   methods=['full_bank','full_top2','mirror_precompose','native_coeff_precompose']+(['mirror_precompose_private','native_coeff_precompose_private'] if rho else [])
   paths={m:ROOT/'source'/f'{seed}_rho{rho:g}_{m}.npz' for m in methods}
   with np.load(paths['full_bank']) as f:
    base=f['base'];router=f['router'];target=run.pred_full(np.random.default_rng(seed+int(rho*10000)).normal(size=(run.N,run.D)).astype('f'),base,f['expert_left'],f['expert_right'],router)
   q=np.random.default_rng(seed+int(rho*10000));_base,u,v,c,res,experts,pl,pr,_router,x=run.world(seed,rho)
   # Inputs and teacher are deterministic; replay each saved payload against full-factor teacher.
   with np.load(paths['full_bank']) as f: target=run.pred_full(x,f['base'],f['expert_left'],f['expert_right'],f['router'])
   for name,path in paths.items():
    row=d[(seed,rho,name)];assert path.stat().st_size==row['serialized_bytes']
    with np.load(path) as z:
     if name.startswith('full_'):pred=run.pred_full(x,z['base'],z['expert_left'],z['expert_right'],z['router'],name=='full_top2')
     else:pred=run.pred_code(x,z['base'],z['u'],z['v'],z['codes'] if 'codes' in z else z['coefficients'],z['router'],z['private_left'] if 'private_left' in z else None,z['private_right'] if 'private_right' in z else None)
     err=float(np.sqrt(np.mean((pred-target)**2)/np.mean(target**2)));assert abs(err-row['output_nrmse'])<1e-7
   assert d[(seed,rho,'mirror_precompose')]['output_nrmse']==d[(seed,rho,'native_coeff_precompose')]['output_nrmse']
