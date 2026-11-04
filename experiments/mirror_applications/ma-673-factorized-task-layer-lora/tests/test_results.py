import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def test_bytes_and_factor_identity():
 rows=json.loads((ROOT/'source/development.json').read_text()); assert len(rows)==6
 by={(x['seed'],x['method']):x for x in rows}
 for x in rows: assert (ROOT/'source'/f"{x['seed']}_{x['method']}.npz").stat().st_size==x['serialized_bytes']
 for seed in (67301,67302):
  assert by[(seed,'mirror_product')]['output_nrmse']==0
  with np.load(ROOT/'source'/f'{seed}_mirror_product.npz') as a,np.load(ROOT/'source'/f'{seed}_native_outer.npz') as b:
   assert np.array_equal(a['atoms'],b['atoms'])
   assert np.array_equal(a['task_factor'],b['task_left'])
   assert np.array_equal(a['layer_factor'],b['layer_right'])
