import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import run_experiment as run

def test_serialized_payload_replay_and_alias():
 rows=json.loads((ROOT/'source/development.json').read_text());assert len(rows)==20
 d={(x['seed'],x['residual_rms'],x['method']):x for x in rows}
 for seed in (67201,67202):
  for rho in (0.,.05):
   base,u,v,c,res,experts,pl,pr,router,x=run.world(seed,rho);target=run.pred_full(x,base,experts,router)
   methods=['full_bank','full_top2','mirror_precompose','native_coeff_precompose']+(['mirror_precompose_private','native_coeff_precompose_private'] if rho else [])
   for name in methods:
    p=ROOT/'source'/f'{seed}_rho{rho:g}_{name}.npz';row=d[(seed,rho,name)];assert p.stat().st_size==row['serialized_bytes']
    with np.load(p) as z:
     if name.startswith('full_'): pred=run.pred_full(x,z['base'],z['experts'],z['router'],name=='full_top2')
     else: pred=run.pred_code(x,z['base'],z['u'],z['v'],z['codes'] if 'codes' in z else z['coefficients'],z['router'],z['private_left'] if 'private_left' in z else None,z['private_right'] if 'private_right' in z else None)
     err=float(np.sqrt(np.mean((pred-target)**2)/np.mean(target**2)))
     assert abs(err-row['output_nrmse'])<1e-7
   a=d[(seed,rho,'mirror_precompose')];b=d[(seed,rho,'native_coeff_precompose')];assert a['output_nrmse']==b['output_nrmse']
