import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma350',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_mirror_direct_and_independent_modes_are_functionally_equal():
 a=m.world(35021)
 for split in ('iid','ood'):
  p=m.predictions('mirror_phase',a,split);q=m.predictions('direct_coefficients',a,split);r=m.predictions('independent_vectors',a,split)
  assert np.allclose(p,q,atol=1e-7) and np.allclose(p,r,atol=1e-7)
def test_payload_roundtrip_charges_mode_code_and_weights():
 a=m.world(35022)
 x=m.pack('mirror_phase',a);meta,arr=m.load(x)
 assert len(x)==int(10+len(__import__('json').dumps(meta,sort_keys=True,separators=(',',':')))+sum(z.nbytes for z in arr))
 assert arr[1].shape==(32,1) and arr[-1].shape==(32,)
