import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def test_payload_sizes_and_native_alias():
 rows=json.loads((ROOT/'source/development.json').read_text());assert len(rows)==16
 d={(r['seed'],r['residual_rms'],r['method']):r for r in rows}
 for r in rows:
  p=ROOT/'source'/f"{r['seed']}_rho{r['residual_rms']:g}_{r['method']}.npz"
  assert p.stat().st_size==r['serialized_bytes']
 for seed in (67201,67202):
  for rho in (0.,.05):
   a=d[(seed,rho,'mirror_precompose')];b=d[(seed,rho,'native_coeff_precompose')]
   assert a['output_nrmse']==b['output_nrmse']
   with np.load(ROOT/'source'/f'{seed}_rho{rho:g}_mirror_precompose.npz') as x, np.load(ROOT/'source'/f'{seed}_rho{rho:g}_native_coeff_precompose.npz') as y:
    for k in ('base','router','u','v'): assert np.array_equal(x[k],y[k])
    assert np.array_equal(x['codes'],y['coefficients'])
