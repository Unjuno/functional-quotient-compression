import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def test_payloads_and_native_exact_alias():
 rows=json.loads((ROOT/'source/development.json').read_text());assert len(rows)==20
 d={(r['seed'],r['method']):r for r in rows}
 for r in rows: assert (ROOT/'source'/f"{r['seed']}_{r['method']}.npz").stat().st_size==r['serialized_bytes']
 for seed in (67401,67402):
  for k in (2,4,8,16):
   assert d[(seed,f'mirror_top{k}')]['output_nrmse']==d[(seed,f'native_top{k}')]['output_nrmse']
   with np.load(ROOT/'source'/f'{seed}_mirror_top{k}.npz') as a,np.load(ROOT/'source'/f'{seed}_native_top{k}.npz') as b:
    for keya,keyb in [('atom_left','atom_left'),('atom_right','atom_right'),('expert_codes','expert_scales')]:assert np.array_equal(a[keya],b[keyb])
